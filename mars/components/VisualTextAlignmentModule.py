import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from mars.components.PriorInformationRefinementModule import PriorInformationRefinementModule
from mars.components.SoftmaxGradCAM import SoftmaxGradCAM
from utils.backbone_loader import BackboneLoader


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
warnings.filterwarnings("ignore")

class VisualTextAlignmentModule:
    def __init__(
        self,
        model: nn.Module,
        model_transforms: transforms.Compose,
        model_patch_size: int,
        model_embedding_spatial_dimensions: int,
        model_num_regs: int,
        vta_refinement_box_threshold: float,
        last_n_attention_maps_for_refinement: int,
        device
    ):
        self.model = model
        self.model_transforms = model_transforms
        self.model_patch_size = model_patch_size
        self.model_embedding_spatial_dimensions = model_embedding_spatial_dimensions
        self.model_num_regs = model_num_regs
        self.device = device
    
        self.pir = PriorInformationRefinementModule(
            box_threshold=vta_refinement_box_threshold,
            last_n_attention_maps_for_refinement=last_n_attention_maps_for_refinement,
            device=device,
            num_regs=model_num_regs
        )
        
    def compute(
        self,
        query_image: torch.Tensor,
        fg_label: str,
        bg_labels: list[str],
        
    ) -> torch.Tensor:
        target_layers = [self.model.visual.transformer.resblocks[-1].ln_1]
        cam = SoftmaxGradCAM(
            model=self.model,
            model_preprocess=self.model_transforms,
            target_layers=target_layers
        )
        
        unrefined_cam, attn_maps_list = cam(
            image=query_image[0],
            foreground_label=fg_label,
            all_labels=bg_labels
        )
        
        refined_cam = self.pir.compute(
            prior=torch.Tensor(unrefined_cam),
            attn_maps=attn_maps_list
        )
        
        return refined_cam

def build_visual_text_alignment_component(args):
    print("[VTA] - Loading VTA module...")
    vtp_model, _ = BackboneLoader.load_backbone(
        backbone_name='clip',
        backbone_size=args.vta_backbone,
        encoder_kwargs={'device': args.device, 'download_root': args.models_path}
    )
    vtp_model_patch_size = int(args.vta_backbone[-2:])
    embedding_spatial_dimensions = int(np.ceil(int(args.input_size) / vtp_model_patch_size)* vtp_model_patch_size)
    vtp_transforms = transforms.Compose([
        transforms.Resize((embedding_spatial_dimensions, embedding_spatial_dimensions), interpolation=BICUBIC),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    print(f"[VTA] - VTA model loaded")
    
    return VisualTextAlignmentModule(
        model=vtp_model,
        model_transforms=vtp_transforms,
        model_patch_size=vtp_model_patch_size,
        model_embedding_spatial_dimensions=embedding_spatial_dimensions,
        model_num_regs=0,
        vta_refinement_box_threshold=args.vta_refinement_box_threshold,
        last_n_attention_maps_for_refinement=args.last_n_attn_for_vta_refinement,
        device=args.device
    )
