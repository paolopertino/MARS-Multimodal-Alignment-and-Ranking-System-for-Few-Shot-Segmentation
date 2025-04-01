import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from clip.clip import tokenize
from clip.clip_text import new_class_names_coco
from mars.utils.coco_prompts import classes, coco_templates
from pytorch_grad_cam import GradCAM

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

class ClipOutputTarget:
    def __init__(self, target_idx):
        # Indicate the idx of the target output on top of which compute the gradient
        self.target_idx = target_idx

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.target_idx]
        return model_output[:, self.target_idx]
    
def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    height = width = int((tensor.size(1)-1)**0.5)
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class SoftmaxGradCAM:
    def __init__(
        self,
        model: nn.Module,
        model_preprocess,
        target_layers
    ) -> None:
        if hasattr(model, 'visual'):
            if hasattr(model.visual, 'patch_size'):
                self.patch_size = model.visual.patch_size
        elif hasattr(model, 'patch_size'):
            self.patch_size = model.patch_size
        else:
            raise ValueError('The model does not have the attribute patch_size')
        
        self.model = model
        self.model_preprocess = model_preprocess
        self.target_layers = target_layers
        
    def compute_text_feats(
        self, 
        labels: list,
        use_multiple_prompts: bool = False    
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the set of class names of objects present in an image,
        it computes the embeddings of the prompts. For foreground objects

        :param labels: list of labels representing the class names of the objects in an image.
        :type labels: list
        :param use_multiple_prompts: if True, a set of 15 prompts will be used for foreground prompts. Optional, defaults to False
        :type use_multiple_prompts: bool, optional
        :return: the text features for both the foreground and background prompts
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            foreground_text_feats = []
            background_text_feats = []
            
            for label in labels:
                if use_multiple_prompts:
                    foregorund_texts = [template.format(label) for template in coco_templates]
                else:
                    foregorund_texts = [f'a photo of {label}.']
                background_texts = [f'a photo without {label}.']
                
                foregorund_texts = tokenize(foregorund_texts).to(device)
                background_texts = tokenize(background_texts).to(device)
                
                foreground_text_embeddings = self.model.encode_text(foregorund_texts)
                background_text_embeddings = self.model.encode_text(background_texts)
                
                foreground_text_embeddings /= foreground_text_embeddings.norm(dim=-1, keepdim=True)
                background_text_embeddings /= background_text_embeddings.norm(dim=-1, keepdim=True)
                foreground_text_embeddings = foreground_text_embeddings.mean(dim=0)
                background_text_embeddings = background_text_embeddings.mean(dim=0)
                foreground_text_embeddings /= foreground_text_embeddings.norm()
                background_text_embeddings /= background_text_embeddings.norm()
                
                foreground_text_feats.append(foreground_text_embeddings)
                background_text_feats.append(background_text_embeddings)
        
        foreground_text_feats = torch.stack(foreground_text_feats, dim = 1).to(device) # 512 x n_labels
        background_text_feats = torch.stack(background_text_feats, dim = 1).to(device) # 512 x n_labels
        
        return foreground_text_feats.t(), background_text_feats.t()
    
    def scoremap2bbox(self, scoremap, threshold, multi_contour_eval=False):
        height, width = scoremap.shape
        scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
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
    
    def scale_cam_image(self, cam, target_size=None) -> list:
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result
    
    def __call__(
        self, 
        image: torch.Tensor, 
        foreground_label: str,
        all_labels: list[str],
        last_n_attn: int = 8,
        use_multiple_prompts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the Softmax-GradCAM for the given image and target label,
        given all the labels of the object classes present in the image.

        :param image: image to compute the GradCAM on
        :type image: torch.Tensor
        :param foreground_label: name of the object class to compute the GradCAM for
        :type foreground_label: str
        :param all_labels: list of all the object classes present in the image except the foreground_label
        :type all_labels: list[str]
        :param last_n_attn: last n attention layers to use to refine the original CAM, defaults to 8
        :type last_n_attn: int, optional
        :param use_multiple_prompts: if True, a set of 15 prompts will be used for foreground prompts. Optional, defaults to False
        :type use_multiple_prompts: bool, optional
        :return: the refined CAM, the unrefined CAM and the transition matrix H to possibly use for custom refinement.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        original_height, original_width = image.shape[-2], image.shape[-1]
        processed_image = self.model_preprocess(image)
        
        processed_image = processed_image.unsqueeze(0)
        h, w = processed_image.shape[-2], processed_image.shape[-1]
        processed_image = processed_image.cuda()
        
        # Note that model.encode_image has an additional `extract` parameter
        # If not set the feats and attn_weights of the second last layer are returned.
        image_features, attn_weight_list = self.model.encode_image(processed_image, h, w)
        
        # Setting the label of interest in the first position
        label_list = [foreground_label] + all_labels
        foreground_text_feats, background_text_feats = self.compute_text_feats(
            label_list,
            use_multiple_prompts=use_multiple_prompts
        )
        fg_bg_text_feats = torch.cat([foreground_text_feats, background_text_feats], dim=0)
        input_tensor = [image_features, fg_bg_text_feats, h, w]
        # As mentioned previously, the label of interest is placed in the first position
        targets = [ClipOutputTarget(0)]
        
        cam = GradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=reshape_transform)
        grayscale_cam, _, attn_weight_last_layer = cam(
            input_tensor = input_tensor,
            targets=targets,
            target_size=None
        )
        grayscale_cam = grayscale_cam[0, :]
        
        attn_weight_list.append(attn_weight_last_layer)
        
        # Discarding the attention weights from/to cls token
        attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
        
        # Keeping only the last_n attention blocks
        attn_weight = torch.stack(attn_weight, dim=0)[-last_n_attn:]
        
        # Taking the mean of the attention blocks. Note that the CLIP model
        # has MultiHead attention blocks, however the implementation already
        # perform the mean across the multiple heads, so for each block we have
        # a single attention map.
        attn_weight = torch.mean(attn_weight, dim=0)
        attn_weight = attn_weight[0].detach()
        attn_weight = attn_weight.float()
        
        # Calculating the matrix B. As PI-CLIP/CLIP-ES paper suggest
        # the matrix B is obtained by threhsolding the initial GradCAM.
        # In this case the GradCAM is thresholded at 0.4 and then bounding
        # boxes are drawn around all the connected components. The binary mask
        # is constructed by putting 1s inside the BBs and 0 outside.
        box, cnt = self.scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
        B = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            B[y0_:y1_, x0_:x1_] = 1
        B = B.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        
        # Performing the Sinkhorn normalization on attn_weight matrix
        aff_mat = attn_weight
        
        D = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        D = D / torch.sum(D, dim=1, keepdim=True)
        
        # Finally getting the refinement matrix R
        DDT = D @ D.t()
        R = torch.max(D, DDT)
        
        # Additional refinement step as introduced in CLIP-ES
        for _ in range(1):
            R = torch.matmul(R, R)
        
        R_B = R * B
        
        # Refining the CAM
        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)
        
        refined_cam = torch.matmul(R_B, cam_to_refine).reshape(h // self.patch_size, w // self.patch_size)
        refined_cam = refined_cam.cpu().numpy().astype(np.float32)
        cam_to_refine = cam_to_refine.reshape(h // self.patch_size, w // self.patch_size).cpu().numpy().astype(np.float32)

        high_res_refined_cam = self.scale_cam_image(
            [refined_cam.copy()],
            (original_width, original_height)
        )[0]
        high_res_unrefined_cam = self.scale_cam_image(
            [cam_to_refine],
            (original_width, original_height)
        )[0]
        
        return (
            (torch.tensor(high_res_refined_cam).cuda(), torch.tensor(refined_cam).cuda()),
            (torch.tensor(high_res_unrefined_cam).cuda(), torch.tensor(cam_to_refine).cuda()),
            R
        )