import os

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from alpha_clip import tokenize as alpha_clip_tokenizer

from utils.backbone_loader import BackboneLoader

class FilteringMergingModule:
    def __init__(
        self,
        alpha_clip_model: nn.Module,
        img_transforms: transforms.Compose,
        mask_transforms: transforms.Compose,
        alpha: float,
        static_threshold: float,
        dynamic_threshold: float,
        device
    ):
        self.alpha_clip_model = alpha_clip_model
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.alpha = alpha
        self.static_threshold = static_threshold
        self.dynamic_threshold = dynamic_threshold
        self.device = device
    
    def compute(
        self,
        query_img: torch.Tensor,
        mask_proposals: torch.Tensor,
        support_mask: torch.Tensor,
        cost_matrix: torch.Tensor,
        patch_features_spatial_dimension: int,
        vva: torch.Tensor,
        vta: torch.Tensor,
        text: list[str]
    ) -> torch.Tensor:
        scored_masks = self._score_proposals(
            query_img=query_img,
            mask_proposals=mask_proposals,
            support_mask=support_mask,
            cost_matrix=cost_matrix,
            patch_features_spatial_dimension=patch_features_spatial_dimension,
            vva=vva,
            vta=vta,
            text=text
        )
        
        return self._merge_masks(scored_masks)
    
    def _score_proposals(
        self,
        query_img: torch.Tensor,
        mask_proposals: torch.Tensor,
        support_mask: torch.Tensor,
        cost_matrix: torch.Tensor,
        patch_features_spatial_dimension: int,
        vva: torch.Tensor,
        vta: torch.Tensor,
        text: list[str]
    ) -> list[tuple[torch.Tensor, float]]:
        vta = vta.detach().cpu().numpy()
        vva = vva.detach().cpu().numpy()
        
        pooled_support_mask = F.adaptive_max_pool2d(
            support_mask.permute(1, 0, 2, 3).float(), 
            (patch_features_spatial_dimension, patch_features_spatial_dimension)
        ) # (ns x 1 x h_embed x w_embed) bcs bsz is set equal to 1
        mask_union = (torch.sum(mask_proposals, dim=0) > 0).float()
        pooled_mask_union = F.adaptive_max_pool2d(
            mask_union.unsqueeze(0), 
            (patch_features_spatial_dimension, patch_features_spatial_dimension)
        ).squeeze(0).cpu().numpy() > 0
        
        # Storing mask proposals scores
        emd_scores = []
        alphaclip_scores = []
        pvv_scores = []
        pvt_scores = []
        
        # Pre-computing alphaclip text features
        text_feats = self._compute_alphaclip_text_feats(text)
        
        # Batched computation of alphaclip image features
        img_feats = self._compute_alphaclip_vis_feats(
            query_img[0], 
            mask_proposals, 
        )
        alphaclip_scores = list((img_feats @ text_feats.T).cpu().numpy())
        
        for m_p in mask_proposals:
            pooled_m_p = F.adaptive_max_pool2d(
                m_p.unsqueeze(0).float(), 
                (patch_features_spatial_dimension, patch_features_spatial_dimension)
            ).squeeze(0)
            coverage_m_p = np.sum(pooled_m_p.cpu().numpy() > 0) / (1e-7 + np.sum(pooled_mask_union))
            m_p_alignment_pvv = np.sum(vva[pooled_m_p.cpu().numpy() > 0]) / (1e-7 + np.sum(pooled_m_p.cpu().numpy() > 0))
            m_p_alignment_pvt = np.sum(vta[pooled_m_p.cpu().numpy() > 0]) / (1e-7 + np.sum(pooled_m_p.cpu().numpy() > 0))
            
            emd_score = self._compute_emd(
                pooled_support_mask.cpu(),
                pooled_m_p,
                cost_matrix
            )
            
            pvv_score = self.alpha * m_p_alignment_pvv + (1 - self.alpha) * coverage_m_p
            pvt_score = self.alpha * m_p_alignment_pvt + (1 - self.alpha) * coverage_m_p
            
            emd_scores.append(emd_score)
            pvv_scores.append(pvv_score)
            pvt_scores.append(pvt_score)
        
        # Min-max scaling of EMD and Alpha-CLIP scores
        min_emd = min(emd_scores)
        max_emd = max(emd_scores)
        min_alphaclip = min(alphaclip_scores)
        max_alphaclip = max(alphaclip_scores)
        
        emd_scores = [(s - min_emd) / (1e-7 + max_emd - min_emd) for s in emd_scores]
        alphaclip_scores = [(s - min_alphaclip) / (1e-7 + max_alphaclip - min_alphaclip) for s in alphaclip_scores]
        
        mask_scores = []
        for i, m_p in enumerate(mask_proposals):
            mask_scores.append((m_p, (emd_scores[i] + alphaclip_scores[i] + pvv_scores[i] + pvt_scores[i]) / 4))
        
        mask_scores = sorted(mask_scores, key=lambda x: x[1], reverse=True)
        
        return mask_scores
    
    def _compute_emd(
        self,
        support_mask: torch.Tensor,
        mask_proposal: torch.Tensor,
        cost_matrix: torch.Tensor
    ) -> float:
        """Computes the Earth Mover's Distance (EMD) between two masks.

        :param ref_mask: mask of the reference image. 
        :type ref_mask: torch.Tensor
        :param query_mask: mask proposal of the query image
        :type query_mask: torch.Tensor
        :param cost_matrix: cost matrix for the EMD computation. 
        It represents the cost of moving a unit of mass from one pixel to another.
        :type cost_matrix: torch.Tensor
        :return: EMD score
        :rtype: float
        """
        emd_cost_pool = cost_matrix[support_mask.flatten().bool(), :][:, mask_proposal.flatten().bool()]
        
        emd = ot.emd2(
            a=[1. / emd_cost_pool.shape[0] for i in range(emd_cost_pool.shape[0])],
            b=[1. / emd_cost_pool.shape[1] for i in range(emd_cost_pool.shape[1])],
            M=emd_cost_pool.cpu().numpy()
        )
        emd_score = 1 - emd
        
        return emd_score
    
    def _compute_alphaclip_text_feats(
        self,
        text: list[str],
    ):
        tokenized_text = alpha_clip_tokenizer(text).to(self.device)
        
        with torch.no_grad():
            text_features = self.alpha_clip_model.encode_text(tokenized_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def _compute_alphaclip_vis_feats(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        image_np = image.permute(1,2,0).cpu().numpy()
        image_alpha_clip = self.img_transforms(image_np).unsqueeze(0).half().to(self.device)
        
        # masks is a tensor of shape (n_masks, h, w). To compute in a single pass
        # the features of all masks, we need to have the images stacked in a single tensor.
        # Note that for each mask proposal the image is the same.
        image_alpha_clip = image_alpha_clip.repeat(masks.shape[0], 1, 1, 1) # n_masks x 3 x h x w

        alpha = torch.stack([self.mask_transforms((mask.cpu().numpy() * 255).astype(np.uint8)) for mask in masks]) # n_masks x 1 x h x w
        alpha = alpha.half().to(self.device) # .unsqueeze(dim=0) 

        with torch.no_grad():
            image_features = self.alpha_clip_model.visual(image_alpha_clip, alpha)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # n_masks x embed_dim
        
        return image_features
    
    def _merge_masks(
        self,
        ranked_masks: list[tuple[torch.Tensor, float]],
    ) -> torch.Tensor:
        if ranked_masks[0][1] < self.static_threshold:
            lower_bound = self.dynamic_threshold * ranked_masks[0][1]
            ranked_masks = [m[0] for m in ranked_masks if m[1] >= lower_bound]
        else:
            ranked_masks = [m[0] for m in ranked_masks if m[1] >= self.static_threshold]
        
        merged_mask = (torch.sum(torch.stack(ranked_masks), dim=0) > 0).float()
        
        return merged_mask
    
def build_filtering_and_merging_module(args):
    print('[FilteringMergingModule] - Loading Filtering and Merging Module...')
    alphaclip_model, alphaclip_transforms = BackboneLoader.load_backbone(
        backbone_name='alphaclip',
        backbone_size='ViT-L/14@336px',
        device=args.device,
        backbone_weights_path=os.path.join(args.models_path, 'clip_l14_336_grit_20m_4xe.pth'),
        encoder_kwargs={'device': args.device, 'download_root': args.models_path}
    )

    alphaclip_img_transforms = alphaclip_transforms[0]
    alphaclip_mask_transforms = alphaclip_transforms[1]
    
    print('[FilteringMergingModule] - Filtering and Merging Module loaded.')
    
    return FilteringMergingModule(
        alpha_clip_model=alphaclip_model,
        img_transforms=alphaclip_img_transforms,
        mask_transforms=alphaclip_mask_transforms,
        alpha=args.alpha_coverage,
        static_threshold=args.static_threshold,
        dynamic_threshold=args.dynamic_threshold,
        device=args.device
    )