import os

from collections import Counter

import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import transforms
from transformers import AutoProcessor
from PIL import Image

from matcher.common.logger import CometLogger, Logger
from mars.components.helpers.prompts import SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA, VISUAL_PROMPTS_VLM_VIP_LLAVA, COLORS, VISUAL_PROMPTS_DESCRIPTIONS_VLM_VIP_LLAVA

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

class VLM:
    def __init__(
        self, 
        model: nn.Module, 
        processor: AutoProcessor,
        prompt_generators: dict,
        generation_config: VLMGenerationConfig,
        ensamble_config: EnsambleConfig,
    ) -> None:
        self.model = model
        self.processor = processor
        self.prompt_generators = prompt_generators
        self.generation_config = generation_config
        self.ensamble_config = ensamble_config
        self.logger = None
    
    @torch.no_grad()
    def fetch_class_name(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
    ) -> str:
        # Ensure that the image and mask are in the correct format
        image = np.array(transforms.ToPILImage()(torch.tensor(image).permute(2, 0, 1)))
        image = np.array(Image.fromarray(image).convert("RGB"))
        mask = np.array(Image.fromarray(mask).convert("L"))
        
        # If the ensamble config is not enabled, use the default configs
        if not self.ensamble_config.is_ensamble():
            prompted_image_np = self.prompt_generators.get(self.generation_config.prompt_type).draw(
                image, 
                mask, 
                color=COLORS[self.generation_config.color], 
                alpha=self.generation_config.alpha, 
                thickness=self.generation_config.thickness,
                zoom_percent=self.generation_config.zoom_pctg
            )
            
            prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_VLM_VIP_LLAVA[self.generation_config.prompt_type].format(self.generation_config.color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
            res = self.model.generate(**prompted_image, max_new_tokens=20)
            pred_class_name = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
        else:
            pred_class_names = self.get_ensamble_predictions(image, mask)
            counter_pred_class_names = Counter(pred_class_names)
            pred_class_name = max(counter_pred_class_names, key=counter_pred_class_names.get)
        
        return pred_class_name
    
    @torch.no_grad()
    def fetch_definition(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        class_name: str,
    ) -> str:
        """Fetch the description of the target object.

        :param image: image of the target object
        :type image: np.ndarray
        :param mask: binary mask of the target object
        :type mask: np.ndarray
        :param class_name: class name of the target object
        :type class_name: str
        :return: definition of the target object
        :rtype: str
        """
        image = np.array(transforms.ToPILImage()(torch.tensor(image).permute(2, 0, 1)))
        image = np.array(Image.fromarray(image).convert("RGB"))
        mask = np.array(Image.fromarray(mask).convert("L"))
        
        prompted_image_np = self.prompt_generators.get(self.generation_config.prompt_type).draw(
            image, 
            mask, 
            color=COLORS[self.generation_config.color], 
            alpha=self.generation_config.alpha, 
            thickness=self.generation_config.thickness,
            zoom_percent=self.generation_config.zoom_pctg
        )
        
        prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_DESCRIPTIONS_VLM_VIP_LLAVA[self.generation_config.prompt_type].format(class_name, self.generation_config.color, class_name, class_name)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
        res = self.model.generate(**prompted_image, min_new_tokens=15, max_new_tokens=50)
        pred_description = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
        
        return pred_description
    
    def get_ensamble_predictions(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> list:
        # Ensamble only colors. Prompt and zoom are fixed and provided by the user
        if self.ensamble_config.is_ensamble_color_only():
            pred_class_names = []
            for color in self.ensamble_config.colors:
                prompted_image_np = self.prompt_generators.get(self.generation_config.prompt_type).draw(
                    image, 
                    mask,
                    color=COLORS[color],
                    alpha=self.generation_config.alpha,
                    thickness=self.generation_config.thickness,
                    zoom_percent=self.generation_config.zoom_pctg
                )
                prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_VLM_VIP_LLAVA[self.generation_config.prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
                res = self.model.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                pred_class_names.append(pred_class_name)

            return pred_class_names
        
        # Ensamble only zoom percentages. Prompt and Color are fixed and provided by the user
        if self.ensamble_config.is_ensamble_zoom_only():
            pred_class_names = []
            for zoom_percentage in self.ensamble_config.zoom_percentages:
                prompted_image_np = self.prompt_generators.get(self.generation_config.prompt_type).draw(
                    image, 
                    mask, 
                    color=COLORS[self.generation_config.color],
                    alpha=self.generation_config.alpha,
                    thickness=self.generation_config.thickness,
                    zoom_percent=zoom_percentage
                ), 
                prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_VLM_VIP_LLAVA[self.generation_config.prompt_type].format(self.generation_config.color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
                res = self.model.generate(**prompted_image, max_new_tokens=20)
                pred_class_name = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                pred_class_names.append(pred_class_name)
                
            return pred_class_names
        
        # Ensamble prompt and colors. Zoom percentage is fixed and provided by the user
        if self.ensamble_config.is_ensamble_prompt_color():
            pred_class_names = []
            for prompt_type in self.ensamble_config.prompt_types:
                for color in self.ensamble_config.colors:
                    prompted_image_np = self.prompt_generators.get(prompt_type).draw(
                        image, 
                        mask, 
                        color=COLORS[color], 
                        alpha=self.generation_config.alpha,
                        thickness=self.generation_config.thickness,
                        zoom_percent=self.generation_config.zoom_pctg
                    )
                    prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_VLM_VIP_LLAVA[prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
                    res = self.model.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                    pred_class_names.append(pred_class_name)
                            
            return pred_class_names
        
        # Ensamble colors and zoom percentages. Prompt is fixed and provided by the user
        if self.ensamble_config.is_ensamble_color_zoom():
            pred_class_names = []
            for color in self.ensamble_config.colors:
                for zoom_percentage in self.ensamble_config.zoom_percentages:
                    prompted_image_np = self.prompt_generators.get(self.generation_config.prompt_type).draw(
                        image, 
                        mask, 
                        color=COLORS[color], 
                        alpha=self.generation_config.alpha, 
                        thickness=self.generation_config.thickness,
                        zoom_percent=zoom_percentage
                    )
                    prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_VLM_VIP_LLAVA[self.generation_config.prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
                    res = self.model.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                    pred_class_names.append(pred_class_name)

            return pred_class_names
        
        # Ensamble prompts and zoom percentages. Color is fixed and provided by the user
        if self.ensamble_config.is_ensamble_prompt_zoom():
            pred_class_names = []
            for prompt_type in self.ensamble_config.prompt_types:
                for zoom_percentage in self.ensamble_config.zoom_percentages:
                    prompted_image_np = self.prompt_generators.get(prompt_type).draw(
                        image, 
                        mask, 
                        color=COLORS[self.generation_config.color], 
                        alpha=self.generation_config.alpha, 
                        thickness=self.generation_config.thickness,
                        zoom_percent=zoom_percentage
                    )
                    prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_VLM_VIP_LLAVA[prompt_type].format(self.generation_config.color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
                    res = self.model.generate(**prompted_image, max_new_tokens=20)
                    pred_class_name = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                    pred_class_names.append(pred_class_name)
                    
            return pred_class_names
        
        # Ensamble both prompts, colors and zoom percentages
        if self.ensamble_config.is_full_ensamble():
            pred_class_names = []
            for prompt_type in self.ensamble_config.prompt_types:
                for color in self.ensamble_config.colors:
                    for zoom_percentage in self.ensamble_config.zoom_percentages:
                        prompted_image_np = self.prompt_generators.get(prompt_type).draw(
                            image, 
                            mask, 
                            color=COLORS[color], 
                            alpha=self.generation_config.alpha, 
                            thickness=self.generation_config.thickness,
                            zoom_percent=zoom_percentage
                        )
                        prompted_image = self.processor(text=SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA.format(VISUAL_PROMPTS_VLM_VIP_LLAVA[prompt_type].format(color)), images=Image.fromarray(prompted_image_np), return_tensors="pt").to(self.model.device)
                        res = self.model.generate(**prompted_image, max_new_tokens=20)
                        pred_class_name = self.processor.decode(res[0][len(prompted_image['input_ids'][0]):], skip_special_tokens=True)
                        pred_class_names.append(pred_class_name)
                        
            return pred_class_names
    
    def set_logger(self, logger: Logger) -> None:
        self.logger = logger