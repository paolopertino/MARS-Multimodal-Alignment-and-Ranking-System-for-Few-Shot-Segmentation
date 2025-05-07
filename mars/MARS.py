"""Definition of the MARS pipeline: Multimodal Alignment and Ranking System for Few-Shot Segmentation"""

from typing import Optional

import torch
import torch.nn.functional as F

from mars.components.TextRetrieverModule import build_text_retriever_component, TextRetrieverModule
from mars.components.VisualTextAlignmentModule import build_visual_text_alignment_component, VisualTextAlignmentModule
from mars.components.VisualVisualAlignmentModule import build_visual_visual_alignment_component, VisualVisualAlignmentModule
from mars.components.FilteringMergingModule import build_filtering_and_merging_module, FilteringMergingModule

class MARS:
    def __init__(
        self,
        text_retriever_component: TextRetrieverModule,
        visual_text_alignment_component: VisualTextAlignmentModule,
        visual_visual_alignment_component: VisualVisualAlignmentModule,
        filtering_merging_component: FilteringMergingModule,
        mask_generator = None
    ):
        self.text_retriever_component = text_retriever_component
        self.visual_text_alignment_component = visual_text_alignment_component
        self.visual_visual_alignment_component = visual_visual_alignment_component
        self.filtering_merging_component = filtering_merging_component
        self.mask_generator = mask_generator
        
    def predict(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_image: torch.Tensor,
        mask_proposals: Optional[torch.Tensor] = None
    ):
        # If no mask proposals are passed and no mask generator
        # has been set, MARS cannot work.
        assert (mask_proposals != None or self.mask_generator != None)
        
        # If the mask generator is set, then we use it to extract mask proposals.
        if self.mask_generator is not None:
            mask_proposals = self.mask_generator.generate(
                support_images, 
                support_masks, 
                query_image
            )
        
        # Extracting the class name of the object of interest.
        entity_of_interest_name, entity_of_interest_description = self.text_retriever_component.get_conceptual_information(
            support_images=support_images,
            support_masks=support_masks,
        )
        
        # Extracting the local-visual information.
        vva = self.visual_visual_alignment_component.compute(
            support_imgs=support_images,
            support_masks=support_masks,
            query_img=query_image
        )
        
        # Extracting the local-conceptual information.
        vta = self.visual_text_alignment_component.compute(
            query_image=query_image,
            fg_label=entity_of_interest_name,
            bg_labels=[]
        )
        # The vta and vva dimensions should be the same.
        # The vta extraction component does not apply the final
        # min-max scale so we apply it here
        vta = F.interpolate(
            vta.unsqueeze(0).unsqueeze(0),
            vva.shape,
            mode='nearest'
        ).squeeze(0).squeeze(0)
        vta = (vta - vta.min()) / (1e-7 + vta.max() - vta.min())
        
        # Preparing the text for global-conceptual score computation.
        text_alpha_clip = []
        if entity_of_interest_description == '':
            text_alpha_clip = [f'a {entity_of_interest_name}.']
        else:
            text_alpha_clip = [f'a {entity_of_interest_name}, {entity_of_interest_description}.']
        
        # Scoring, filtering, and merging the mask proposals
        return self.filtering_merging_component.compute(
            query_img=query_image,
            mask_proposals=mask_proposals,
            support_mask=support_masks,
            cost_matrix=self.visual_visual_alignment_component.cost_matrix,
            patch_features_spatial_dimension=self.visual_visual_alignment_component.model_embedding_spatial_dimensions,
            vva=vva,
            vta=vta,
            text=text_alpha_clip
        )


def build_MARS_fss(args):
    return MARS(
        text_retriever_component=build_text_retriever_component(args=args),
        visual_text_alignment_component=build_visual_text_alignment_component(args=args),
        visual_visual_alignment_component=build_visual_visual_alignment_component(args=args),
        filtering_merging_component=build_filtering_and_merging_module(args=args),
    )