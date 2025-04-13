import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from mars.components.PriorInformationRefinementModule import PriorInformationRefinementModule
from utils.backbone_loader import BackboneLoader

class VisualVisualAlignmentModule:
    def __init__(
        self,
        model: nn.Module,
        model_transforms: transforms.Compose,
        model_patch_size: int,
        model_embedding_spatial_dimensions: int,
        model_num_regs: int,
        vva_refinement_box_threshold: float,
        last_n_attention_maps_for_refinement: int,
    ):
        self.model = model
        self.model_transforms = model_transforms
        self.model_patch_size = model_patch_size
        self.model_embedding_spatial_dimensions = model_embedding_spatial_dimensions
        self.model_num_regs = model_num_regs
    
        self.pir = PriorInformationRefinementModule(
            box_threshold=vva_refinement_box_threshold,
            last_n_attention_maps_for_refinement=last_n_attention_maps_for_refinement,
            num_regs=model_num_regs
        )
        
        self.similarity_matrix = None
        self.cost_matrix = None
        
    def compute(
        self,
        support_imgs: torch.Tensor,
        support_masks: torch.Tensor,
        query_img: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the visual-visual alignment between the support image and the query image.

        :param support_imgs: the support images. It has shape [BS, NumShots, C, H, W] 
        :type support_imgs: torch.Tensor
        :param support_masks: the support masks. It has shape [BS, NumShots, H, W]
        :type support_masks: torch.Tensor
        :param query_img: the query image. It has shape [BS, C, H, W]
        :type query_img: torch.Tensor
        :return: Returns the refined visual-visual alignment, with shape [H, W]
        :rtype: torch.Tensor
        """
        # Extracting visual patch features and matching them to produce a 
        # similarity matrix. 
        visual_support_feats = self._extract_patch_features(support_imgs[0])
        visual_query_feats = self._extract_patch_features(query_img)
        visual_attention_maps = list(self.model.get_last_self_attention(self.model_transforms(query_img[0]).unsqueeze(0).cuda()))
        self.similarity_matrix = torch.matmul(visual_support_feats, visual_query_feats.T).cpu()
        self.cost_matrix = (1 - similarity_matrix) / 2
        
        pooled_support_mask = F.adaptive_max_pool2d(
            support_masks.permute(1, 0, 2, 3).float(), 
            (self.model_embedding_spatial_dimensions, self.model_embedding_spatial_dimensions)
        ) # support mask from (bs x ns x h x w) to (ns x bs x h x w)
        relevant_support_feats = visual_support_feats[pooled_support_mask.flatten().bool()]
        irrelevant_support_feats = visual_support_feats[~pooled_support_mask.flatten().bool()]
        similarity_matrix = torch.matmul(relevant_support_feats, visual_query_feats.T)
        anti_similarity_matrix = torch.matmul(irrelevant_support_feats, visual_query_feats.T)
        
        # Computing foreground VVA
        vva_max = similarity_matrix.max(dim=0).values.reshape(
            (self.model_embedding_spatial_dimensions, self.model_embedding_spatial_dimensions)
        )
        vva_mean = similarity_matrix.mean(dim=0).reshape(
            (self.model_embedding_spatial_dimensions, self.model_embedding_spatial_dimensions)
        )
        vva = vva_mean * vva_max
        
        # Computing background VVA
        vva_max_bg = anti_similarity_matrix.max(dim=0).values.reshape(
            (self.model_embedding_spatial_dimensions, self.model_embedding_spatial_dimensions)
        )
        vva_mean_bg = anti_similarity_matrix.mean(dim=0).reshape(
            (self.model_embedding_spatial_dimensions, self.model_embedding_spatial_dimensions)
        )
        vva_bg = vva_mean_bg * vva_max_bg
        vva -= vva_bg
        vva = (vva - vva.min()) / (1e-7 + vva.max() - vva.min())
        
        # Refining the VVA
        vva_refined = self.pir.compute(
            prior=vva, 
            attn_maps=visual_attention_maps, 
        )
        scaled_vva_refined = (vva_refined - vva_refined.min()) / (1e-7 + vva_refined.max() - vva_refined.min())
        
        return scaled_vva_refined
        
    def _extract_patch_features(self, imgs):
        imgs = torch.cat(
            [
                self.model_transforms(i).unsqueeze(0).cuda() for i in imgs
            ],
            dim=0
        )
        
        with torch.no_grad():
            feats = self.model.forward_features(imgs)['x_prenorm'][:, 1+self.model_num_regs:]
        
        feats = feats.reshape(-1, self.model.embed_dim)
        feats = F.normalize(feats, p=2, dim=1)

        return feats
    
def build_visual_visual_alignment_component(args):
    # DINOv2, Image Encoder
    dino_kwargs = dict(
        img_size=args.input_size,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        num_register_tokens=args.num_regs,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    
    vvp_model, vvp_transforms = BackboneLoader.load_backbone(
        backbone_name='dinov2',
        backbone_size=args.dino_backbone,
        device=args.device,
        backbone_weights_path=os.path.join(args.models_path, 'dinov2_vitl14_reg4_pretrain.pth') if num_regs == 4 else os.path.join(args.models_path, 'dinov2_vitl14_pretrain.pth'),
        encoder_kwargs=dino_kwargs
    )
    vvp_model_patch_size = dino_kwargs['patch_size']
    embedding_spatial_dimensions = args.input_size // dino_kwargs['patch_size']
    
    return VisualVisualAlignmentModule(
        model=vvp_model,
        model_transforms=vvp_transforms,
        model_patch_size=vvp_model_patch_size,
        model_embedding_spatial_dimensions=embedding_spatial_dimensions,
        model_num_regs=args.num_regs,
        vva_refinement_box_threshold=args.vva_refinement_box_threshold,
        last_n_attention_maps_for_refinement=args.last_n_attn_for_vva_refinement,
        use_vva_mix=args.use_vva_mix,
        use_negative_prior=args.use_negative_prior
    )
