from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import VipLlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from mars.components.VisualPromptGenerator import MaskGenerator, BoundingBoxGenerator, MaskContourGenerator, EllipseGenerator
from mars.components.helpers.prompts import SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA, COLORS, VISUAL_PROMPTS_VLM_VIP_LLAVA, VISUAL_PROMPTS_DESCRIPTIONS_VLM_VIP_LLAVA

VISUAL_PROMPT_GENERATORS = {
    'mask': MaskGenerator(),
    'bb': BoundingBoxGenerator(),
    'contour': MaskContourGenerator(),
    'ellipse': EllipseGenerator()
}

class TextRetrieverModule:
    def __init__(
        self,
        vlm,
        vlm_processor,
        prompt_generator,
        vlm_ensamble_config,
        vlm_prompt_generation_config,
    ):
        self.vlm = vlm
        self.vlm_processor = vlm_processor
        
        # Visual Prompt Generator
        self.visual_prompt_generator = prompt_generator
        
        # Configurations
        self.vlm_ensamble_config = vlm_ensamble_config
        self.vlm_prompt_generation_config = vlm_prompt_generation_config
    
    def get_conceptual_information(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> tuple[str, str]:
        predicted_names = []
        
        # For each support image-mask pair in the support set
        # extract the class name. The class name to use will be
        # extracted through majority voting.
        for s_img, s_mask in zip(support_images[0], support_masks[0]):
            support_img_numpy = np.array(transforms.ToPILImage()(s_img).convert("RGB"))
            support_mask_numpy = s_mask.numpy()
            if not self.vlm_ensamble_config.is_ensamble():
                # Draw Visual Prompt on top of the support image.
                prompted_image_np = self.visual_prompt_generator.draw(
                    image=s_img,
                    mask=s_mask, 
                    color=COLORS[self.vlm_prompt_generation_config.color],
                    alpha=self.vlm_prompt_generation_config.alpha,
                    thickness=self.vlm_prompt_generation_config.thickness,
                    zoom_percent=self.vlm_prompt_generation_config.zoom_pctg
                )
                
                # Constructing text prompts to feed to the VLM
                prompt = VISUAL_PROMPTS_VLM_VIP_LLAVA[self.vlm_prompt_generation_config.prompt_type].format(self.vlm_prompt_generation_config.color)
                prompt = SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(prompt)
                prompted_image = self.vlm_processor(
                    text=prompt, 
                    images=Image.fromarray(prompted_image_np), 
                    return_tensors="pt"
                ).to(self.vlm.device)
                
                # Predicting the class name
                res = self.vlm.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = self.vlm_processor.decode(
                    res[0][len(prompted_image['input_ids'][0]):], 
                    skip_special_tokens=True
                )
                
                predicted_names.append(pred_class_name)
            else:
                pred_class_names = self._get_ensamble_predictions(
                    support_img_numpy, 
                    support_mask_numpy
                )
                counter_pred_class_names = Counter(pred_class_names)
                pred_class_name = max(
                    counter_pred_class_names, 
                    key=counter_pred_class_names.get
                )
                
                predicted_names.append(pred_class_name)
        
        # Applying majority voting for choosing the class name
        # (for k-shot we are choosing among k generated class names)
        counter_predicted_names = Counter(predicted_names)
        result_name = max(counter_predicted_names, key=counter_predicted_names.get)
        
        # Fetching a definition for the extracted entity of interest
        pred_description = None
        prompted_image_np = self.visual_prompt_generator.draw(
            image=s_img,
            mask=s_mask, 
            color=COLORS[self.vlm_prompt_generation_config.color],
            alpha=self.vlm_prompt_generation_config.alpha,
            thickness=self.vlm_prompt_generation_config.thickness,
            zoom_percent=self.vlm_prompt_generation_config.zoom_pctg
        )
        prompt = VISUAL_PROMPTS_DESCRIPTIONS_VLM_VIP_LLAVA[self.vlm_prompt_generation_config.prompt_type].format(result_name, self.vlm_prompt_generation_config.color)
        prompt = SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(prompt)
        prompted_image = self.vlm_processor(
            text=prompt, 
            images=Image.fromarray(prompted_image_np), 
            return_tensors="pt"
        ).to(self.vlm.device)
        res = self.vlm.generate(**prompted_image, min_new_tokens=20, max_new_tokens=50)
        pred_description = self.vlm_processor.decode(
            res[0][len(prompted_image['input_ids'][0]):], 
            skip_special_tokens=True
        )
        
        # Given the predicted class name and definition,
        # the wordnet description is extracted.
        wn_synset_for_class_name = self._get_synset(
            class_name=result_name,
            vlm_description=pred_description
        )
        if (wn_synset_for_class_name is not None):
            wn_description = wn.synset(wn_synset_for_class_name).definition()
        else:
            wn_description = ''
            
        return result_name, wn_description
    
    def _get_synset(self, class_name: str, vlm_description: str):
        lower_class_name = class_name.strip().lower()
        stop_words = set(stopwords.words('english'))

        # If the class name is composed by multiple words,
        # to match the synset most often the synset is matched
        # by changing the spaces with underscores.
        synsets = []
        synsets += wn.synsets(lower_class_name.replace(' ', '_'), pos=wn.NOUN)

        # In other cases the synset is matched by the class name
        # with the spaces removed
        if len(synsets) == 0:
            synsets += wn.synsets(lower_class_name.replace(' ',
                                    ''), pos=wn.NOUN)

        # If no synset is still found, then try to match the subwords of the class name
        if len(synsets) == 0:
            for word in lower_class_name.split():
                synsets += wn.synsets(word.strip(), pos=wn.NOUN)

        # If no synset is found, return None
        if len(synsets) == 0:
            return None

        # If a single synset is found, return the name of the synset
        if len(synsets) == 1:
            return synsets[0].name()

        # If multiple synsets are found, the synset is matched
        # using the description of the object.
        print(f"[TextRetrieverModule] - Multiple synsets found for {class_name}. Matching using description.")
        best_synset = None
        max_overlap = 0
        description_tokens = set(word_tokenize(vlm_description.lower())) - stop_words

        for synset in synsets:
            definition_tokens = set(word_tokenize(
                synset.definition().lower())) - stop_words
            # Count overlapping words
            overlap = len(description_tokens & definition_tokens)

            if overlap > max_overlap:
                max_overlap = overlap
                best_synset = synset

        return best_synset.name() if best_synset else None
        
    def _get_ensamble_predictions(
        self,
        support_img: np.array, 
        support_mask: np.array, 
        args
    ) -> list:
        if self.vlm_ensamble_config.is_ensamble_color_only():
            pred_class_names = []
            for color in self.vlm_ensamble_config.colors:
                prompted_image_np = self.visual_prompt_generator.draw(
                    image=support_img, 
                    mask=support_mask, 
                    color=COLORS[color],  
                    alpha=self.vlm_prompt_generation_config.alpha, 
                    thickness=self.vlm_prompt_generation_config.thickness,
                    zoom_percent=self.vlm_prompt_generation_config.zoom_pctg
                )
                prompted_image = self.vlm_processor(
                    text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(
                        VISUAL_PROMPTS_VLM_VIP_LLAVA[args.prompt_type].format(color)
                    ), 
                    images=Image.fromarray(prompted_image_np), 
                    return_tensors="pt"
                ).to(self.vlm.device)
                res = self.vlm.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = self.vlm_processor.decode(
                    res[0][len(prompted_image['input_ids'][0]):], 
                    skip_special_tokens=True
                )
                pred_class_names.append(pred_class_name)

            return pred_class_names
        
        if self.vlm_ensamble_config.is_ensamble_zoom_only():
            pred_class_names = []
            for zoom_percentage in self.vlm_ensamble_config.zoom_percentages:
                prompted_image_np = self.visual_prompt_generator.draw(
                    image=support_img,
                    mask=support_mask,
                    color=COLORS[self.vlm_prompt_generation_config.color],
                    alpha=self.vlm_prompt_generation_config.alpha, 
                    thickness=self.vlm_prompt_generation_config.thickness,
                    zoom_percentage=self.vlm_prompt_generation_config.zoom_percent
                )
                prompt = VISUAL_PROMPTS_VLM_VIP_LLAVA[self.vlm_prompt_generation_config.prompt_type].format(self.vlm_prompt_generation_config.color)
                prompt = SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(prompt)
                
                prompted_image = self.vlm_processor(
                    text=prompt, 
                    images=Image.fromarray(prompted_image_np), 
                    return_tensors="pt"
                ).to(self.vlm.device)
                res = self.vlm.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = self.vlm_processor.decode(
                    res[0][len(prompted_image['input_ids'][0]):], 
                    skip_special_tokens=True
                )
                pred_class_names.append(pred_class_name)
                
            return pred_class_names
        
        if self.vlm_ensamble_config.is_ensamble_prompt_color():
            pred_class_names = []
            for prompt_type in self.vlm_ensamble_config.prompt_types:
                for color in self.vlm_ensamble_config.colors:
                    prompted_image_np = VISUAL_PROMPT_GENERATORS[prompt_type].draw(
                        image=support_img, 
                        mask=support_mask, 
                        color=COLORS[color], 
                        alpha=self.vlm_prompt_generation_config.alpha, 
                        thickness=self.vlm_prompt_generation_config.thickness,
                        zoom_percentage=self.vlm_prompt_generation_config.zoom_percent
                    )
                    prompt = VISUAL_PROMPTS_VLM_VIP_LLAVA[prompt_type].format(color)
                    prompt = SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(prompt)
                
                    prompted_image = self.vlm_processor(
                        text=prompt, 
                        images=Image.fromarray(prompted_image_np), 
                        return_tensors="pt"
                    ).to(self.vlm.device)
                    
                    res = self.vlm.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = self.vlm_processor.decode(
                        res[0][len(prompted_image['input_ids'][0]):], 
                        skip_special_tokens=True
                    )
                    pred_class_names.append(pred_class_name)
                            
            return pred_class_names
        
        if self.vlm_ensamble_config.is_ensamble_color_zoom():
            pred_class_names = []
            for color in self.vlm_ensamble_config.colors:
                for zoom_percentage in self.vlm_ensamble_config.zoom_percentages:
                    prompted_image_np = self.visual_prompt_generator.draw(
                        image=support_img, 
                        mask=support_mask, 
                        color=COLORS[color], 
                        alpha=self.vlm_prompt_generation_config.alpha, 
                        thickness=self.vlm_prompt_generation_config.thickness,
                        zoom_percentage=zoom_percentage
                    )
                    prompt = VISUAL_PROMPTS_VLM_VIP_LLAVA[self.vlm_prompt_generation_config.prompt_type].format(color)
                    prompt = SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(prompt)
                    
                    prompted_image = self.vlm_processor(
                        text=prompt, 
                        images=Image.fromarray(prompted_image_np), 
                        return_tensors="pt"
                    ).to(self.vlm.device)
                    res = self.vlm.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = self.vlm_processor.decode(
                        res[0][len(prompted_image['input_ids'][0]):], 
                        skip_special_tokens=True
                    )
                    pred_class_names.append(pred_class_name)

            return pred_class_names
        
        if self.vlm_ensamble_config.is_ensamble_prompt_zoom():
            pred_class_names = []
            for prompt_type in self.vlm_ensamble_config.prompt_types:
                for zoom_percentage in self.vlm_ensamble_config.zoom_percentages:
                    prompted_image_np = VISUAL_PROMPT_GENERATORS[prompt_type].draw(
                        image=support_img, 
                        mask=support_mask, 
                        color=COLORS[self.vlm_prompt_generation_config.color], 
                        alpha=self.vlm_prompt_generation_config.alpha, 
                        thickness=self.vlm_prompt_generation_config.thickness,
                        zoom_percentage=zoom_percentage
                    )
                    prompt = VISUAL_PROMPTS_VLM_VIP_LLAVA[prompt_type].format(self.vlm_prompt_generation_config.color)
                    prompt = SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(prompt)
                    
                    prompted_image = self.vlm_processor(
                        text=prompt, 
                        images=Image.fromarray(prompted_image_np), 
                        return_tensors="pt"
                    ).to(self.vlm.device)
                    
                    res = self.vlm.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = self.vlm_processor.decode(
                        res[0][len(prompted_image['input_ids'][0]):], 
                        skip_special_tokens=True
                    )
                    pred_class_names.append(pred_class_name)
                    
            return pred_class_names
        
        if self.vlm_ensamble_config.is_full_ensamble():
            pred_class_names = []
            for prompt_type in self.vlm_ensamble_config.prompt_types:
                for color in self.vlm_ensamble_config.colors:
                    for zoom_percentage in self.vlm_ensamble_config.zoom_percentages:
                        prompted_image_np = VISUAL_PROMPT_GENERATORS[prompt_type].draw(
                            image=support_img, 
                            mask=support_mask, 
                            color=COLORS[color], 
                            alpha=self.vlm_prompt_generation_config.alpha, 
                            thickness=self.vlm_prompt_generation_config.thickness,
                            zoom_percentage=zoom_percentage
                        )
                        prompt = VISUAL_PROMPTS_VLM_VIP_LLAVA[prompt_type].format(self.vlm_prompt_generation_config.color)
                        prompt = SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(prompt)
                        
                        prompted_image = self.vlm_processor(
                            text=prompt, 
                            images=Image.fromarray(prompted_image_np), 
                            return_tensors="pt"
                        ).to(self.vlm.device)
                        
                        res = self.vlm.generate(**prompted_image, max_new_tokens=20)
                        pred_class_name = self.vlm_processor.decode(
                            res[0][len(prompted_image['input_ids'][0]):], 
                            skip_special_tokens=True
                        )
                        pred_class_names.append(pred_class_name)
                        
            return pred_class_names
        
class VLMGenerationConfig:
    def __init__(
        self,
        prompt_type: str = "contour",
        zoom_pctg: float = 0.,
        color: str = "red",
        thickness: int = 2,
        alpha: float = 0.5
    ) -> None:
        self.prompt_type = prompt_type
        self.zoom_pctg = zoom_pctg
        self.color = color
        self.thickness = thickness
        self.alpha = alpha

class EnsambleConfig:
    def __init__(
            self, 
            ensamble_prompts: bool = False,
            ensamble_zoom: bool = False,
            ensamble_colors: bool = False,
            prompt_types: list = ["bb", "contour", "ellipse"], 
            zoom_percentages: list = [0, 30, 50], 
            colors: list = ["red", "green", "blue"]
        ):
        self.ensamble_prompts = ensamble_prompts
        self.ensamble_zoom = ensamble_zoom
        self.ensamble_colors = ensamble_colors
        self.prompt_types = prompt_types
        self.zoom_percentages = zoom_percentages
        self.colors = colors
    
    def is_ensamble(self) -> bool:
        if self.ensamble_zoom or self.ensamble_colors:
            return True
        
        if self.ensamble_prompts and not self.ensamble_zoom and not self.ensamble_colors:
            print("[WARNING] Ensamble prompts is enabled but no other ensamble option is enabled. Using default prompt w/o ensamble.")
            return False
        
        return False
    
    def is_ensamble_color_only(self) -> bool:
        if self.ensamble_colors and not self.ensamble_zoom and not self.ensamble_prompts:
            return True
        
        return False
    
    def is_ensamble_zoom_only(self) -> bool:
        if self.ensamble_zoom and not self.ensamble_colors and not self.ensamble_prompts:
            return True
        
        return False
    
    def is_ensamble_prompt_color(self) -> bool:
        if self.ensamble_prompts and self.ensamble_colors and not self.ensamble_zoom:
            return True
        
        return False
    
    def is_ensamble_prompt_zoom(self) -> bool:
        if self.ensamble_prompts and self.ensamble_zoom and not self.ensamble_colors:
            return True
        
        return False
    
    def is_ensamble_color_zoom(self) -> bool:
        if self.ensamble_colors and self.ensamble_zoom and not self.ensamble_prompts:
            return True
        
        return False
    
    def is_full_ensamble(self) -> bool:
        if self.ensamble_prompts and self.ensamble_zoom and self.ensamble_colors:
            return True
        
        return False
    
def build_text_retriever_component(args):
    print("[TextRetrieverModule] - Loading Text Retriever Module...")
    # Mapping the VLM on the second GPU if more than 1 are available, otherwise choose automatically.
    device_map = {"": 1} if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "auto"
    
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
        alpha=args.alpha_blending
    )
    
    # Loading the VLM possibly quantized at either 4 or 8 bits
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.vlm4bit,
        load_in_8bit=args.vlm8bit,
    )
    vlm_model = VipLlavaForConditionalGeneration.from_pretrained(
        "llava-hf/vip-llava-7b-hf",
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    vlm_processor = AutoProcessor.from_pretrained(
        "llava-hf/vip-llava-7b-hf")
    
    print("[TextRetrieverModule] - Text Retriever Module loaded.")
    
    return TextRetrieverModule(
        vlm=vlm_model,
        vlm_processor=vlm_processor,
        prompt_generator=VISUAL_PROMPT_GENERATORS[args.prompt_type],
        vlm_ensamble_config=vlm_ensamble_config,
        vlm_prompt_generation_config=vlm_prompt_generation_config,
        
    )