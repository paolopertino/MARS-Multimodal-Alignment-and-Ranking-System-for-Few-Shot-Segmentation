r""" Visualize model predictions """
import os

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms as transforms

from . import utils

class Visualizer:

    @classmethod
    def initialize(cls, visualize, ds_name: str = None, exp_name:str = None, fold_num: int = 0):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = './vis/'
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)
        if ds_name is not None:
            cls.vis_path += ds_name + '/' + exp_name + '/'
            if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)
            cls.vis_path += f'fold{fold_num}/'
            if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx, fold_num, idx_to_classname, iou_b=None, comet_logger = None):
        spt_img_b = utils.to_cpu(spt_img_b)
        spt_mask_b = utils.to_cpu(spt_mask_b)
        qry_img_b = utils.to_cpu(qry_img_b)
        qry_mask_b = utils.to_cpu(qry_mask_b)
        pred_mask_b = utils.to_cpu(pred_mask_b)
        cls_id_b = utils.to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b)):
            iou = iou_b[sample_idx] if iou_b is not None else None
            class_path = cls.vis_path + f'{cls_id}_{idx_to_classname[cls_id.item()]}/'
            if not os.path.exists(class_path): os.makedirs(class_path)
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, idx_to_classname[cls_id.item()], fold_num, iou, class_path, comet_logger)

    @classmethod
    def to_numpy(cls, tensor, type, unnormalize=True):
        if type == 'img':
            if unnormalize:
                return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
            else:
                return np.array(cls.to_pil(tensor)).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, label, fold_num, iou=None, save_path=None, comet_logger=None):

        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs_unnormalized = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_imgs = [cls.to_numpy(spt_img, 'img', unnormalize=False) for spt_img in spt_imgs]
        spt_pils_unnormalized = [cls.to_pil(spt_img) for spt_img in spt_imgs_unnormalized]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks_unnormalized = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masks = [cls.to_numpy(spt_mask, 'mask', unnormalize=False) for spt_mask in spt_masks]
        spt_masked_pils_unnormalized = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs_unnormalized, spt_masks_unnormalized)]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]

        qry_img_unnormalized = cls.to_numpy(qry_img, 'img')
        qry_img = cls.to_numpy(qry_img, 'img', unnormalize=False)
        qry_pil_unnormalized = cls.to_pil(qry_img_unnormalized)
        qry_pil = cls.to_pil(qry_img)
        qry_mask_unnormalized = cls.to_numpy(qry_mask, 'mask')
        qry_mask = cls.to_numpy(qry_mask, 'mask', unnormalize=False)
        pred_mask_unnormalized = cls.to_numpy(pred_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask', unnormalize=False)
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        pred_masked_pil_unnormalized = Image.fromarray(cls.apply_mask(qry_img_unnormalized.astype(np.uint8), pred_mask_unnormalized.astype(np.uint8), pred_color))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))
        qry_masked_pil_unnormalized = Image.fromarray(cls.apply_mask(qry_img_unnormalized.astype(np.uint8), qry_mask_unnormalized.astype(np.uint8), qry_color))

        titles = ["Support image with mask", "Query image", "Prediction", "GT"]
        images_with_titles = cls.add_titles_to_images(spt_masked_pils + [qry_pil, pred_masked_pil, qry_masked_pil], titles)
        images_unnormalized = cls.merge_image_pair(spt_masked_pils_unnormalized + [qry_pil_unnormalized, pred_masked_pil_unnormalized, qry_masked_pil_unnormalized], title="Unnormalized images")

        merged_pil = cls.merge_image_pair(images_with_titles, title=f"Image class: {cls_id} | IoU: {iou:.2f}" if iou else f"Image class: {cls_id}")
        merged_pil = cls.merge_image_pair_vertically([merged_pil, images_unnormalized])

        iou = iou.item() if iou else 0.0
        
        if save_path is None:
            merged_pil.save(cls.vis_path + 'fold%d_%d_%d_class-%d-%s_iou-%.2f' % (fold_num, batch_idx, sample_idx, cls_id, label, iou) + '.jpg')
        else:
            merged_pil.save(save_path + 'fold%d_%d_%d_class-%d-%s_iou-%.2f' % (fold_num, batch_idx, sample_idx, cls_id, label, iou) + '.jpg')
            
        # Comet logging
        if comet_logger is not None:
            comet_logger.log_image(
                image_data=merged_pil, 
                name=f'fold{fold_num}_{batch_idx}_{sample_idx}_class-{cls_id}-{label}_iou-{iou:.2f}'
            )

    @classmethod
    def merge_image_pair(cls, pil_imgs, title):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs]) + 30  # Adding space for titles
        canvas = Image.new('RGB', (canvas_width, canvas_height + 30), "white")

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        # Draw the main title
        title_width, title_height = draw.textbbox((0, 0), title, font=font)[2:]
        draw.text(((canvas_width - title_width) / 2, 10), title, fill="black", font=font)
        
        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 40))  # Adjust y-position to account for the main title
            xpos += pil.size[0]

        return canvas
    
    @classmethod
    def merge_image_pair_vertically(cls, pil_imgs):
        r""" Vertically aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = max([pil.size[0] for pil in pil_imgs])
        canvas_height = sum([pil.size[1] for pil in pil_imgs]) + 30
        canvas = Image.new('RGB', (canvas_width, canvas_height), "white")
        
        ypos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (0, 40 + ypos))
            ypos += pil.size[1]
        
        return canvas

    @classmethod
    def add_titles_to_images(cls, images, titles):
        """
        Add titles to a list of PIL images and return the new images with titles added.
        """
        titled_images = []
        for image, title in zip(images, titles):
            width, height = image.size
            canvas = Image.new('RGB', (width, height + 20), 'white')  # Create a new image with extra space for title
            canvas.paste(image, (0, 20))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            text_width, text_height = draw.textbbox((0, 0), title, font=font)[2:]
            draw.text(((width - text_width) / 2, 0), title, fill="black", font=font)
            titled_images.append(canvas)
        return titled_images

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img
