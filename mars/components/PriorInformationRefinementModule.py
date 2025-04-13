import cv2
import numpy as np
import torch

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

class PriorInformationRefinementModule:
    def __init__(
        self,
        box_threshold: float,
        last_n_attention_maps_for_refinement: int,
        num_regs: int = 0,
    ):
        self.threshold = box_threshold
        self.last_n_attention_maps_for_refinement = last_n_attention_maps_for_refinement
        self.num_regs = num_regs
    
    def compute(
        self,
        prior: torch.Tensor,
        attn_maps: list,
    ) -> torch.Tensor:
        prior = prior.cpu().numpy()
        original_prior_shape = prior.shape

        # Discarding the attention weights from/to cls token and possible register
        # tokens, and keeping only the last self.last_n_attention_maps_for_refinement
        # attention blocks.
        attention_maps = torch.stack(
            [aw[0, :, 1+self.num_regs:, 1+self.num_regs:] for aw in attn_maps], 
            dim=0
        )[-self.last_n_attention_maps_for_refinement:]
        
        # Taking the mean of the retained attention blocks and among all the heads
        # TODO: might change between VVA and RVA. This code should work for VVA
        attention_maps = torch.mean(attention_maps, dim=(0, 1))
        attention_maps = attention_maps.float()

        # Calculating the matrix B. As PI-CLIP/CLIP-ES paper suggest
        # the matrix B is obtained by threhsolding the initial GradCAM.
        # In this case the GradCAM is thresholded at self.threshold and then 
        # bounding boxes are drawn around all the connected components. 
        # The binary mask is constructed by putting 1s inside the 
        # BBs and 0 outside.
        box, cnt = self._scoremap2bbox(
            scoremap=prior, 
            multi_contour_eval=True
        )
        B = torch.zeros(
            (prior.shape[0], prior.shape[1])
        ).cuda()
        
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            B[y0_:y1_, x0_:x1_] = 1

        B = B.view(1, prior.shape[0] * prior.shape[1])

        # Performing the Sinkhorn normalization on attn_weight matrix
        aff_mat = attention_maps
        
        D = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        D = D / torch.sum(D, dim=1, keepdim=True)

        # Finally getting the refinement matrix R
        DDT = D @ D.t()
        R = torch.max(D, DDT)

        # Additional refinement step as introduced in CLIP-ES
        for _ in range(1):
            R = torch.matmul(R, R)

        R_B = R * B
        
        # Refining the initial prior
        prior = torch.FloatTensor(prior).cuda()
        refined_prior = torch.matmul(
            R_B, prior.view(-1, 1)
        ).reshape(original_prior_shape)

        return refined_prior
    
    def _scoremap2bbox(
        self,
        scoremap: np.ndarray,
        multi_contour_eval: bool = False
    ) -> tuple[np.ndarray, int]:
        height, width = scoremap.shape
        scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(self.threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)
