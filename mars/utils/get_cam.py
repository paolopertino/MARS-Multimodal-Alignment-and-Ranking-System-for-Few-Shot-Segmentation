# pylint: disable=locally-disabled, consider-using-enumerate, import-error, unused-import
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from lxml import etree

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from clip.clip_text import class_names, new_class_names, new_class_names_coco

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


class ClipOutputTarget:
    def __init__(self, category):
        self.category = category # At which index of the label_list list the label of interest is located

    def __call__(self, model_output):
        # print(f"Model output shape: {model_output.shape}")
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
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


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform_resize(h, w):
    return Compose([
        Resize((h, w), interpolation=BICUBIC),
        # _convert_image_to_rgb,
        # ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def img_ms_and_flip(que_img, ori_height, ori_width, scales=[1.0], patch_size=16):

    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size)
                                       * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        # que_img = que_img.cpu().detach().numpy().astype(np.uint8)
        # que_img = Image.fromarray(que_img.transpose(1, 2, 0))
        image = preprocess(que_img)
    return image

# que_img is a tensor of shape (batch_size, 3, H, W)
# tmp_que_name is a list of strings (names of the files in the batch for the query images)
# que_class is a list of integers (class labels for the query images)


def get_img_cam(que_img, tmp_que_name, que_class, model, bg_text_features, fg_text_features, cam, annotation_root, flag=None, patch_size=16, last_n_attn=8):
    model = model.cuda()
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()
    refined_cam_all_scales = []
    unrefined_cam_all_scales = []
    transition_matrices = []
    for i in range(0, len(tmp_que_name)):
        que_name = tmp_que_name[i]
        if 'VOC' in annotation_root:
            xmlfile = os.path.join(annotation_root, str(que_name))
            xmlfile = xmlfile.replace('.jpg', '.xml')
            with open(xmlfile) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]

            ori_width = int(data['size']['width'])
            ori_height = int(data['size']['height'])

            label_list = []
            label_id_list = []
            for obj in data["object"]:
                obj["name"] = new_class_names[class_names.index(obj["name"])]
                if obj["name"] not in label_list:
                    label_list.append(obj["name"])
                    label_id_list.append(new_class_names.index(obj["name"]))
        else:
            ori_height, ori_width = np.asarray(
                que_img[i].cpu().detach()).shape[1:]
            if flag == False:
                # tmp_label_img_path = os.path.join(annotation_root, 'val2014')
                label_img_path = os.path.join(
                    annotation_root, que_name).replace('jpg', 'png')
            else:
                tmp_label_img_path = os.path.join(annotation_root, 'train2014')
                label_img_path = os.path.join(
                    tmp_label_img_path, que_name).replace('jpg', 'png')
            label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
            label_id_list = np.unique(label_img).tolist()
            if 0 in label_id_list:
                label_id_list.remove(0)
            if 255 in label_id_list:
                label_id_list.remove(255)
            label_id_list = [x - 1 for x in label_id_list]

            label_list = []
            for lid in label_id_list:
                label_list.append(new_class_names_coco[int(lid)])

        if que_class[i] not in label_id_list:
            # returns a tensor of shape (64,64) filled with 255
            return [torch.full((64, 64), 255).float().cuda()]

        image = img_ms_and_flip(
            que_img[i], ori_height, ori_width, scales=[1.0], patch_size=patch_size)
        
        # plt.imshow(image.numpy().transpose(1, 2, 0))
        
        image = image.unsqueeze(0)
        h, w = image.shape[-2], image.shape[-1]
        image = image.cuda()
        image_features, attn_weight_list = model.encode_image(image, h, w)

        bg_features_temp = bg_text_features[label_id_list].cuda()
        fg_features_temp = fg_text_features[label_id_list].cuda()
        text_features_temp = torch.cat(
            [fg_features_temp, bg_features_temp], dim=0)
        input_tensor = [image_features, text_features_temp, h, w]

        # print(f"Label list: {label_list} - label id list: {label_id_list}")
        for idx, label in enumerate(label_list):
            if 'VOC' in annotation_root:
                label_id = new_class_names.index(label)
            else:
                label_id = new_class_names_coco.index(label)
            if label_id == que_class[i]:
                # print(f"Processing label {label} - {label_list.index(label)}")
                # keys.append(label_id)
                targets = [ClipOutputTarget(label_list.index(label))]
                
                # torch.cuda.empty_cache()
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                        targets=targets,
                                                                        target_size=None)

                # print(f"Grayscale CAM shape: {grayscale_cam.shape}")
                grayscale_cam = grayscale_cam[0, :]
                
                # plt.imshow(grayscale_cam)

                # if idx == 0:
                attn_weight_list.append(attn_weight_last)
                attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
                attn_weight = torch.stack(attn_weight, dim=0)[-last_n_attn:]
                attn_weight = torch.mean(attn_weight, dim=0)
                attn_weight = attn_weight[0].detach()
                attn_weight = attn_weight.float()

                box, cnt = scoremap2bbox(
                    scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                aff_mask = torch.zeros(
                    (grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1

                aff_mask = aff_mask.view(
                    1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / \
                    torch.sum(trans_mat, dim=1, keepdim=True)

                H_trans_mat = trans_mat @ trans_mat.t()
                trans_mat = torch.max(trans_mat, H_trans_mat)

                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                trans_mat_final = trans_mat * aff_mask

                cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
                cam_to_refine = cam_to_refine.view(-1, 1)

                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(
                    trans_mat_final, cam_to_refine).reshape(h // patch_size, w // patch_size)
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_to_refine = cam_to_refine.reshape(h // patch_size, w // patch_size).cpu().numpy().astype(np.float32)

        # print(f"Refined CAM shape: {cam_refined.shape}")
        cam_refined_highres = scale_cam_image(
            [cam_refined.copy()], (ori_width, ori_height))[0]
        cam_unrefined_highres = scale_cam_image(
            [cam_to_refine], (ori_width, ori_height))[0]
        refined_cam_all_scales.append((torch.tensor(cam_refined_highres).cuda(), torch.tensor(cam_refined).cuda()))
        unrefined_cam_all_scales.append((torch.tensor(cam_unrefined_highres).cuda(), torch.tensor(cam_to_refine).cuda()))
        transition_matrices.append(trans_mat)
    return refined_cam_all_scales, unrefined_cam_all_scales, transition_matrices