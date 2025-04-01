import torch
import alpha_clip
import clip

from torchvision import transforms

import dinov2.utils.utils as dinov2_utils

from dinov2.models import vision_transformer as vits
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from mars.data.transforms import MaybeToPILImage


class BackboneLoader:
    AVAILABLE_BACKBONES = {"dinov2", "resnet101", "alphaclip", "clip"}

    @classmethod
    def load_backbone(
        cls,
        backbone_name: str,
        backbone_size: str = None,
        backbone_weights_path: str = None,
        device: str = 'cuda',
        encoder_kwargs: dict = None
    ):
        """
        Load a specific backbone model for feature extraction.

        Parameters:
        - backbone_name (str): The name of the backbone model to load. Must be one of the available backbones.
        - backbone_size (str): The size of the backbone model (e.g., 'base' for ViT-B/16).
        - backbone_weights_path (str): The path to the pre-trained weights for the backbone model.
        - device (str): The device to load the model on. Default is 'cuda'.
        - encoder_kwargs (dict): Additional keyword arguments for the backbone model.

        Returns:
        - encoder_loader: An instance of the loaded backbone model.
        """
        assert backbone_name in cls.AVAILABLE_BACKBONES, f"Backbone {backbone_name} not available. Choose from {cls.AVAILABLE_BACKBONES}"

        if backbone_name == "dinov2":
            encoder_loader = DinoV2(
                backbone_size, backbone_weights_path, encoder_kwargs)
        elif backbone_name == "resnet101":
            encoder_loader = Resnet(backbone_size)
        elif backbone_name == "alphaclip":
            encoder_loader = AlphaCLIP(
                backbone_size, backbone_weights_path, encoder_kwargs)
        elif backbone_name == "clip":
            encoder_loader = CLIP(backbone_size, encoder_kwargs)

        return encoder_loader.encoder.to(device), encoder_loader.encoder_transform


class DinoV2():
    def __init__(self, size: str, weights_path, kwargs):
        """
        Initialize a DINOv2 backbone model.

        Parameters:
        - size (str): The size of the DINOv2 model (e.g., 'base' for ViT-B/16).
        - weights_path (str): The path to the pre-trained weights for the DINOv2 model.
        - kwargs (dict): Additional keyword arguments for the DINOv2 model.
        """
        self.model = vits.__dict__[size](**kwargs)
        dinov2_utils.load_pretrained_weights(
            self.model, weights_path, "teacher")
        self.model.eval()

        # Setting custom parameters to the model
        # Name of the family of models
        setattr(self.model, "family", "vits_dino2")

        self.dino_transforms = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @property
    def encoder_transform(self):
        """
        Return the transform applied to the input data for the DINOv2 model.

        Returns:
        - dino_transforms: The transform applied to the input data.
        """
        return self.dino_transforms

    @property
    def encoder(self):
        """
        Return the DINOv2 model.

        Returns:
        - model: The DINOv2 model.
        """
        return self.model


class Resnet():
    def __init__(self, model_size) -> None:
        """
        Initialize a ResNet backbone model.

        Parameters:
        - model_size (str): The size of the ResNet model (e.g., 'resnet101').
        """
        assert model_size in ["resnet18", "resnet34", "resnet50", "resnet101",
                              "resnet152"], f"Resnet model size {model_size} not available. Choose from ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']"
        self.backbone = torch.hub.load(
            'pytorch/vision:v0.10.0', model_size, pretrained=True)
        self.model = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.model.eval()

        # Setting custom parameters to the model
        # Name of the family of models
        setattr(self.model, "family", "convnets_resnet")

        # Channel dimension of the embedding space
        setattr(self.model, "embed_dim", 2048)

        # "Patch size" of the model. Patch size is not the proper terminology for ConvNets,
        # but it is used to indicate the downsampling factor of the model in this case.
        setattr(self.model, "patch_size", 32)

        self.resnet_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
        ])

    @property
    def encoder(self):
        """
        Return the ResNet model.

        Returns:
        - model: The ResNet model.
        """
        return self.model

    @property
    def encoder_transform(self):
        """
        Return the transform applied to the input data for the ResNet model.

        Returns:
        - encoder_transform: The transform applied to the input data.
        """
        return self.resnet_transform


class AlphaCLIP():
    def __init__(self, model_size, weights_path, kwargs):
        """
        Initialize an AlphaCLIP backbone model.

        Parameters:
        - model_size (str): The size of the AlphaCLIP model (e.g., 'base').
        - weights_path (str): The path to the pre-trained weights for the AlphaCLIP model.
        - kwargs (dict): Additional keyword arguments for the AlphaCLIP model.
        """
        self.model, _ = alpha_clip.load(
            model_size,
            alpha_vision_ckpt_pth=weights_path,
            download_root=kwargs.get('download_root', 'models'),
            device=kwargs.get('device', 'cuda:0')
        )
        self.model.eval()

        self.image_transforms = transforms.Compose([
            MaybeToPILImage(),
            transforms.Resize(
                336, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(336),
            MaybeToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.mask_transforms = transforms.Compose([
            MaybeToTensor(),
            transforms.Resize((336, 336)),
            transforms.Normalize(0.5, 0.26)
        ])

    @property
    def encoder_transform(self):
        """
        Return the transform applied to the input data for the AlphaCLIP model.

        Returns:
        - alphaclip_transforms: The transform applied to the input data.
        """
        return (self.image_transforms, self.mask_transforms)

    @property
    def encoder(self):
        """
        Return the AlphaCLIP model.

        Returns:
        - model: The AlphaCLIP model.
        """
        return self.model

class CLIP():
    def __init__(self, model_size, kwargs):
        self.model, self.clip_image_transforms = clip.load(
            model_size,
            download_root=kwargs.get('download_root', 'models'),
            device=kwargs.get('device', 'cuda:0')
        )
        
    @property
    def encoder_transform(self):
        """
        Return the transform applied to the input data for the DINOv2 model.

        Returns:
        - dino_transforms: The transform applied to the input data.
        """
        return self.clip_image_transforms

    @property
    def encoder(self):
        """
        Return the DINOv2 model.

        Returns:
        - model: The DINOv2 model.
        """
        return self.model