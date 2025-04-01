import os

import torch
import torch.nn as nn
import numpy as np
import alpha_clip

from torchvision import transforms
from transformers import VipLlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from utils.backbone_loader import BackboneLoader
from mars.components.TextFetcher import COCOTextFetcher
from mars.components.PromptGenerator import PromptGenerator
from mars.components.MaskMerger import MaskMerger
from mars.components.VLM import VLM, EnsambleConfig, VLMGenerationConfig
from mars.components.VisualPromptGenerator import MaskGenerator, BoundingBoxGenerator, MaskContourGenerator, EllipseGenerator
from matcher.common.logger import CometLogger


class CLIPPipeline():
    def __init__(
        self,
        alphaclip_model: nn.Module,
        alphaclip_transforms: transforms.Compose,
        prompt_generator: PromptGenerator,
        mask_merger: MaskMerger,
        generator: SamAutomaticMaskGenerator,
        vlm: VLM = None,
        input_size: int = 518,
        min_mask_region_area: int = 0,
        containment_threshold: float = 0.0,
        visualize: bool = False,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ):
        # AlphaClip parameters
        self.alphaclip_model = alphaclip_model
        self.alphaclip_img_transforms = alphaclip_transforms[0]
        self.alphaclip_mask_transforms = alphaclip_transforms[1]
        self.prompt_generator = prompt_generator
        self.mask_merger = mask_merger

        # SAM generator parameters
        self.generator = generator

        # VLM parameters
        self.vlm = vlm
        if vlm is not None:
            self.prompt_generator.text_fetcher.set_vlm(vlm)

        # Input parameters
        self.reference_img = None
        self.reference_mask = None
        self.target_img = None
        self.class_name = None
        self.description = None

        # Misc parameters
        self.candidate_mask = None
        self.input_size = input_size
        self.min_mask_region_area = min_mask_region_area
        self.containment_threshold = containment_threshold
        self.prompts = []
        self.visualize = visualize
        self.device = device
        self.predicted_mask = None
        self.logger = None

    def set_reference(self, reference_image, reference_mask):
        """Fetches and store informations about the reference (support) images and masks.

        :param imgs: Reference/support image(s) of the current iteration. 1 x ns x 3 x h x w
        :type imgs: torch.Tensor
        :param masks: masks of the reference/support images. 1 x ns x h x w
        :type masks: torch.Tensor
        """

        def reference_masks_verification(masks):
            """If there is no mask with value different from 0, then a square of size 14x14 is set to 1
            in the center of the mask.

            :param masks: masks of the reference/support images.
            :type masks: torch.Tensor
            :return: returns the eventually adjusted masks
            :rtype: torch.Tensor
            """
            if masks.sum() == 0:
                _, _, sh, sw = masks.shape
                masks[..., (sh // 2 - 7):(sh // 2 + 7),
                      (sw // 2 - 7):(sw // 2 + 7)] = 1
            return masks

        reference_img = reference_image.flatten(
            0, 2).permute(1, 2, 0).cpu().numpy()
        reference_img_size = reference_img.shape[0]

        assert reference_img_size == self.input_size

        reference_mask = reference_masks_verification(reference_mask)
        reference_mask = reference_mask.permute(1, 0, 2, 3)

        self.reference_img = reference_img
        self.reference_mask = reference_mask

    def set_target(self, target_image):
        """Fetches and store informations about the target image.

        :param target_image: target image of the current iteration. 1 x 3 x h x w
        :type target_image: torch.Tensor
        """
        target_image = target_image.flatten(
            0, 1).permute(1, 2, 0).cpu().numpy()
        target_image_size = target_image.shape[0]

        assert target_image_size == self.input_size

        self.target_img = target_image

    def set_class_name(self, class_name):
        """Set the class name of the target object."""
        self.class_name = class_name

    def fetch_class_name(self):
        """Fetch the class name of the target object from the support image."""
        return self.vlm.fetch_class_name(self.reference_img, self.reference_mask.squeeze(0).squeeze(0).cpu().numpy())

    def process_masks(self, text: list[str]) -> tuple[list[dict], dict]:
        """Given the image path, model, generator and text, return the generated masks and similarity pairs with the text.

        :param text: list of COCO prompts for the target object
        :type text: list[str]
        :return: generated masks and similarity pairs with the text
        :rtype: tuple[list[dict], dict]
        """
        image_np = self.target_img
        image_alpha_clip = self.alphaclip_img_transforms(
            self.target_img).unsqueeze(0).half().to(self.device)
        text = alpha_clip.tokenize(text).to(self.device)

        generated_masks = self.generator.generate(image_np)

        if self.candidate_mask is not None and len(self.candidate_mask) > 0:
            # generated_masks = []
            for c_mask in self.candidate_mask:
                generated_masks.append(
                    {'segmentation': c_mask, 'type': 'matcher'})

        mask_sim_pairs = {}

        with torch.no_grad():
            text_features = self.alphaclip_model.encode_text(text)
            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)

        for idx, g_mask in enumerate(generated_masks):
            alpha = self.alphaclip_mask_transforms(
                (g_mask['segmentation'] * 255).astype(np.uint8))
            alpha = alpha.half().to(self.device).unsqueeze(dim=0)

            with torch.no_grad():
                image_features = self.alphaclip_model.visual(
                    image_alpha_clip, alpha)

            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)

            similarity = 100.0 * image_features @ text_features.T
            mask_sim_pairs[idx] = similarity.cpu().numpy()

        return generated_masks, mask_sim_pairs

    def set_candidate_mask(self, mask):
        self.candidate_mask = mask

    def predict(self):
        # Fetching the class name of the target object from the support image
        if self.class_name is None:
            if self.vlm is None:
                raise ValueError(
                    "If the class name is not set, the VLM object must be set.")

            self.class_name = self.fetch_class_name().lower()
            self.prompt_generator.text_fetcher.set_img(self.reference_img)
            self.prompt_generator.text_fetcher.set_mask(
                self.reference_mask.squeeze(0).squeeze(0).cpu().numpy())

        # Free cache memory
        torch.cuda.empty_cache()

        # Given the class name, fetch the COCO prompts for the target object
        print(f"Used class name: {self.class_name}")
        self.prompts = self.prompt_generator.prompt_generation(self.class_name)

        # Process the masks and get the similarity pairs with the text
        generated_masks, mask_sim_pairs = self.process_masks(self.prompts)

        # Merging the masks based on the mask merging strategy
        self.predicted_mask = self.mask_merger.merge_masks(
            masks=generated_masks,
            mask_sim_pairs=mask_sim_pairs,
            prompts=self.prompts,
            show=self.visualize
        )

        return self.predicted_mask['merged_mask']

    def set_logger(self, logger: CometLogger):
        """Set the logger object for ClipPipeline and all its components.

        :param logger: logger object
        :type logger: CometLogger
        """
        self.logger = logger

        # Setting the logger for the PromptGenerator object,
        # the MaskMerger object, and the VLM object
        if self.prompt_generator is not None:
            self.prompt_generator.set_logger(logger)

        if self.mask_merger is not None:
            self.mask_merger.set_logger(logger)

        if self.vlm is not None:
            self.vlm.set_logger(logger)

    def reset_clip_pipeline(self):
        """Reset the CLIPPipeline object parameters."""
        self.reference_img = None
        self.reference_mask = None
        self.target_img = None
        self.class_name = None
        self.description = None
        self.prompts = []
        self.predicted_mask = None
        self.generator.reset_stored_features()
        self.candidate_mask = None


def build_CLIPpipeline_oss(args):
    # Alpha-CLIP
    alphaclip_model, alphaclip_transforms = BackboneLoader.load_backbone(
        backbone_name=args.backbone,
        backbone_size=args.backbone_size,
        device=args.device,
        backbone_weights_path=args.backbone_weights,
        encoder_kwargs={'device': args.device}
    )

    # Text-Fetcher
    text_fetcher_cfg = {
        'llm_key': args.llm_key,
        'description_db_path': args.description_db_path,
        'use_openai_llm': args.use_openai_llm,
        'use_llama_llm': args.use_llama_llm,
        'use_wordnet': args.use_wordnet,
        'use_wordnet_gt_mapping': args.use_wordnet_gt_mapping,
        'use_wordnet_openai_fallback': args.use_wordnet_openai_fallback,
    }

    # Add here more datasets if needed
    if args.benchmark == 'coco':
        text_fetcher = COCOTextFetcher(text_fetcher_cfg)
    else:
        raise NotImplementedError(
            f"Dataset {args.benchmark} is not supported.")

    # Prompt-Generator
    prompt_usage_cfg = {
        'use_class_name_only': args.use_class_name_only,
        'use_synonyms': args.use_synonyms,
        'use_meronyms': args.use_meronyms,
        'use_descriptions': args.use_descriptions,
        'use_wordnet': args.use_wordnet,
    }
    if not PromptGenerator.prompt_usage_cfg_sanity_checks(prompt_usage_cfg):
        raise ValueError("Prompt usage configuration is not valid.")
    prompt_generator = PromptGenerator(text_fetcher, prompt_usage_cfg)

    # SAM
    sam = sam_model_registry[args.sam_size](checkpoint=args.sam_weights)
    sam.to(device=args.device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        sel_stability_score_thresh=args.sel_stability_score_thresh,
        sel_pred_iou_thresh=args.iou_filter,
        box_nms_thresh=args.box_nms_thresh,
        # sel_output_layer=args.output_layer,
        # output_layer=args.dense_multimask_output,
        # dense_pred=args.use_dense_mask,
        # multimask_output=args.dense_multimask_output > 0,
        # sel_multimask_output=args.multimask_output > 0,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )

    # VLM
    if not args.use_gt_class_names:
        device_map = {"": 1} if torch.cuda.is_available(
        ) and torch.cuda.device_count() > 1 else "auto"
        vlm_ensamble_config = EnsambleConfig(
            ensamble_prompts=args.ensamble_prompts,
            ensamble_zoom=args.ensamble_zoom,
            ensamble_colors=args.ensamble_colors,
            prompt_types=args.ensamble_prompts_list,
            zoom_percentages=args.ensamble_zoom_list,
            colors=args.ensamble_colors_list,
        )
        vlm_prompt_generation_config = VLMGenerationConfig(
            prompt_type=args.prompt_type,
            zoom_pctg=args.zoom_percentage,
            color=args.color,
            thickness=args.thickness,
            alpha=args.alpha
        )
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        vlm_model = VipLlavaForConditionalGeneration.from_pretrained(
            "llava-hf/vip-llava-7b-hf",
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )
        vlm_processor = AutoProcessor.from_pretrained(
            "llava-hf/vip-llava-7b-hf")
        prompt_generators = {
            'mask': MaskGenerator(),
            'bb': BoundingBoxGenerator(),
            'contour': MaskContourGenerator(),
            'ellipse': EllipseGenerator()
        }
        vlm = VLM(
            model=vlm_model,
            processor=vlm_processor,
            prompt_generators=prompt_generators,
            generation_config=vlm_prompt_generation_config,
            ensamble_config=vlm_ensamble_config,
        )
    else:
        vlm = None

    # Parameters for mask merging strategies
    mask_merge_cfg = {
        'mask_merging_mode': args.mask_merging_mode,  # max, avg, top_per_prompt
        'similarity_threshold': args.similarity_threshold,  # can be -1 to avoid filtering
        'percentile': args.percentile,  # 95 is a good value
        # 3 to 5 possible range. If set, the percentage is ignored
        'keep_top_k_masks': args.keep_top_k_masks,
        # defines the minimum number of masks to use (if the topK masks are less than this number, the topK is increased). Suggestion: at least 3.
        'min_mask_number': args.min_mask_number,
        # 0.0 to 0.5 possible range. Used only for top_per_prompt.
        'penalization_weight': args.penalization_weight,
    }
    if not MaskMerger.mask_merge_cfg_sanity_checks(mask_merge_cfg):
        raise ValueError("Mask merging configuration is not valid.")
    mask_merger = MaskMerger(mask_merge_cfg)

    return CLIPPipeline(
        alphaclip_model=alphaclip_model,
        alphaclip_transforms=alphaclip_transforms,
        prompt_generator=prompt_generator,
        mask_merger=mask_merger,
        input_size=args.img_size,
        min_mask_region_area=args.min_mask_region_area,
        containment_threshold=args.containment_threshold,
        generator=generator,
        vlm=vlm,
        visualize=args.visualize != 0,
        device=args.device
    )
