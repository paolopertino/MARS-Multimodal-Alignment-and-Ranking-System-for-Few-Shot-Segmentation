"""
Module containing reader to parse pascal_5i dataset from SBD and VOC2012. Adapted for Matcher
"""
import os

from typing import List, Tuple

import numpy as np
import torch.nn.functional as F
import torch
import torchvision

from scipy.io import loadmat
from PIL import Image
from tqdm import trange


class DatasetPASCAL5i(torchvision.datasets.vision.VisionDataset):
    """
    PASCAL5i dataset. This class reads PASCAL5i dataset from SBD and VOC2012.

    Parameters:
        - datapath:  root to data folder containing SBD and VOC2012 dataset. See README.md for details
        - fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf). Possible folds: `{0, 1, 2, 3}`
        - transform: set of transformations to apply to the images
        - split:  split of the dataset. Possible splits: `{train, val}`
        - shot:  number of support images to use in few-shot learning. Possible shots: `{1, 5}`
        - use_original_imgsize:  whether to use the original image size or resize to a fixed size.
    """

    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        super(DatasetPASCAL5i, self).__init__(datapath, None, None, None)
        assert fold >= 0 and fold <= 3
        self.nfold = 4
        self.nclass = 20
        self.split = 'trn' if split == 'train' else 'val'
        self.train = split == 'train'
        self.benchmark = 'pascal5i'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        # Define base to SBD and VOC2012
        sbd_base = os.path.join(datapath, 'PASCAL5i', 'sbd')
        voc_base = os.path.join(datapath, 'PASCAL5i', 'VOCdevkit', 'VOC2012')

        # Define path to relevant txt files
        sbd_train_list_path = os.path.join(
            datapath, 'PASCAL5i', 'sbd', 'train.txt')
        sbd_val_list_path = os.path.join(
            datapath, 'PASCAL5i', 'sbd', 'val.txt')
        voc_train_list_path = os.path.join(
            voc_base, 'ImageSets', 'Segmentation', 'train.txt')
        voc_val_list_path = os.path.join(
            voc_base, 'ImageSets', 'Segmentation', 'val.txt')

        # Use np.loadtxt to load all train/val sets
        sbd_train_list = list(np.loadtxt(sbd_train_list_path, dtype="str"))
        sbd_val_list = list(np.loadtxt(sbd_val_list_path, dtype="str"))
        voc_train_list = list(np.loadtxt(voc_train_list_path, dtype="str"))
        voc_val_list = list(np.loadtxt(voc_val_list_path, dtype="str"))

        # Following PANet, we use images in SBD validation for training
        sbd_train_list = sbd_train_list + sbd_val_list

        # Remove overlapping images in SBD/VOC2012 from SBD train
        sbd_train_list = [i for i in sbd_train_list if i not in voc_val_list]
        
        # Generate self.images and self.targets
        if self.train:
            # If an image occur in both SBD and VOC2012, use VOC2012 annotation
            sbd_train_list = [
                i for i in sbd_train_list if i not in voc_train_list]

            # Generate image/mask full paths for SBD dataset
            sbd_train_img_list = [os.path.join(
                sbd_base, 'img', i + '.jpg') for i in sbd_train_list]
            sbd_train_target_list = [os.path.join(
                sbd_base, 'cls', i + '.mat') for i in sbd_train_list]

            # Generate image/mask full paths for VOC2012 segmentation training task
            voc_train_img_list = [os.path.join(
                voc_base, 'JPEGImages', i + '.jpg') for i in voc_train_list]
            voc_train_target_list = [os.path.join(
                voc_base, "SegmentationClass", i + '.png') for i in voc_train_list]

            # FINAL: Merge these two datasets
            self.images = sbd_train_img_list + voc_train_img_list
            self.targets = sbd_train_target_list + voc_train_target_list
        else:
            # Generate image/mask full paths for VOC2012 semantation validation task
            # Following PANet, only VOC2012 validation set is used for validation
            self.images = [os.path.join(
                voc_base, 'JPEGImages', i + '.jpg') for i in voc_val_list]
            self.targets = [os.path.join(
                voc_base, "SegmentationClass", i + '.png') for i in voc_val_list]

        # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
        # Given fold number, define L_{test}
        self.val_label_set = [i for i in range(fold * 5 + 1, fold * 5 + 6)]
        self.train_label_set = [i for i in range(
            1, 21) if i not in self.val_label_set]
        if self.train:
            self.label_set = self.train_label_set
            self.class_ids = self.train_label_set
        else:
            self.label_set = self.val_label_set
            self.class_ids = self.val_label_set

        assert len(self.images) == len(self.targets)
        self.transforms = transform

        # Find subset of image. This is actually faster than hist
        folded_images = []
        folded_targets = []

        # Given a class, this dict returns list of images containing the class
        self.class_img_map = {}
        for label_id, x in enumerate(self.label_set):
            self.class_img_map[x] = []

        # Given an index of an image, this dict returns list of classes in the image
        self.img_class_map = {}

        saved_metadata_path = os.path.join(
            voc_base, f"dataset_{fold}_{self.train}.pt")

        # If metadata exists, load it, otherwise create it.
        # Metadata contains the mapping between images and classes and vice versa
        # so that we can quickly find images containing a class or classes in an image
        if os.path.exists(saved_metadata_path):
            print('Using saved class mapping')
            d = torch.load(saved_metadata_path)
            self.img_class_map = d['icm']
            self.class_img_map = d['cim']
            folded_images = d['fi']
            folded_targets = d['ft']
        else:
            print('Creating Dataset')
            for i in trange(len(self.images)):
                mask = self.load_seg_mask(self.targets[i])
                appended_flag = False
                for label_id, x in enumerate(self.label_set):
                    if x in mask:
                        if not appended_flag:
                            # contain at least one pixel in L_{train}
                            folded_images.append(self.images[i])
                            folded_targets.append(self.targets[i])
                            appended_flag = True
                        cur_img_id = len(folded_images) - 1
                        cur_class_id = x
                        # This image must be the latest appended image
                        self.class_img_map[cur_class_id].append(cur_img_id)
                        if cur_img_id in self.img_class_map:
                            self.img_class_map[cur_img_id].append(cur_class_id)
                        else:
                            self.img_class_map[cur_img_id] = [cur_class_id]
            torch.save({
                'icm': self.img_class_map,
                'cim': self.class_img_map,
                'fi': folded_images,
                'ft': folded_targets
            }, saved_metadata_path)

        self.images = folded_images     # names of the files of the images in this fold
        self.targets = folded_targets   # names of the files of the masks in this fold

    def load_seg_mask(self, file_path):
        """
        Load seg_mask from file_path (supports .mat and .png).

        Target masks in SBD are stored as matlab .mat; while those in VOC2012 are .png

        Parameters:
            - file_path: path to the segmenation file

        Return: a numpy array of dtype long and element range(0, 21) containing segmentation mask
        """
        if file_path.endswith('.mat'):
            mat = loadmat(file_path)
            target = Image.fromarray(mat['GTcls'][0]['Segmentation'][0])
        else:
            target = Image.open(file_path)
        target_np = np.array(target, dtype=np.int_)

        # Annotation in VOC contains 255
        # Handled in evaluator.
        # target_np[target_np > 20] = 0
        return target_np

    def set_bg_pixel(self, target_np):
        """
        Following OSLSM, we mask pixels not in current label set as 0. e.g., when
        self.train = True, pixels whose labels are in L_{test} are masked as background

        Parameters:
            - target_np: segmentation mask (usually returned array from self.load_seg_mask)

        Return:
            - Offseted and masked segmentation mask
        """
        if self.train:
            for x in self.val_label_set:
                target_np[target_np == x] = 0
            max_val_label = max(self.val_label_set)
            target_np[target_np >
                      max_val_label] = target_np[target_np > max_val_label] - 5
        else:
            label_mask_idx_map = []
            for x in self.val_label_set:
                label_mask_idx_map.append(target_np == x)
            target_np = np.zeros_like(target_np)
            for i in range(len(label_mask_idx_map)):
                target_np[label_mask_idx_map[i]] = i + 1
        return target_np

    def get_img_containing_class(self, class_id):
        """
        Given a class label id (e.g., 2), return a list of all images in
        the dataset containing at least one pixel of the class.

        Parameters:
            - class_id: an integer representing class

        Return:
            - a list of all images in the dataset containing at least one pixel of the class
        """
        return self.class_img_map[class_id]

    def get_class_in_an_image(self, img_idx):
        """
        Given an image idx (e.g., 123), return the list of classes in
        the image.

        Parameters:
            - img_idx: an integer representing image

        Return:
            - list of classes in the image
        """
        return self.img_class_map[img_idx]

    def sample_episode(self, idx: int, offset: int = 0) -> Tuple[str, List[str], int]:
        """Given an index of an image, sample an episode for few-shot learning.
        
        Sampling an episode consists of:
         - Selecting an image as query image.
         - Selecting a class in the image. If multiple classes are present in the image,
             select one class using a round-robin strategy.
         - Selecting support images containing the selected class. If 1-shot, select 1 image.
             If 5-shot, select 5 images.
         Finally the query image, support images, and the selected class are returned.

        :param idx: index of the episode
        :type idx: int
        :param offset: offset to select the class in the image, defaults to 0.
        :type offset: int, optional
        :return: query image, support images, and the selected class
        :rtype: tuple[str, List[str], int]
        """
        query_name = self.images[idx]
        classes_in_image = self.get_class_in_an_image(idx)
        class_sample = classes_in_image[offset % len(classes_in_image)]
        
        support_names = []
        while True:
            support_name = self.images[np.random.choice(
                self.get_img_containing_class(class_sample), 1, replace=False)[0]]
            if query_name != support_name:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        return query_name, support_names, class_sample

    def load_frame(self, query_name: str, support_names: List[str]) -> tuple:
        """Load the query image and support images and their masks.

        :param query_name: name of the query image
        :type query_name: str
        :param support_names: names of the support images
        :type support_names: List[str]
        :return: query image, query mask, support images, support masks, and the original query image size
        :rtype: tuple
        """
        query_img = Image.open(query_name).convert("RGB")
        query_mask = torch.tensor(self.load_seg_mask(
            self.targets[self.images.index(query_name)]))

        support_imgs = [Image.open(name).convert("RGB")
                        for name in support_names]
        support_masks = [torch.tensor(self.load_seg_mask(
            self.targets[self.images.index(name)])) for name in support_names]

        # saving the original query image size
        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def extract_binary_mask(self, mask: torch.Tensor, class_id: int)-> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the binary mask of the class from the mask.
        
        The original mask could contain multiple classes. This function extracts the mask of the
        class we are interested in by setting to 0 all the pixels that do not belong to the class.

        :param mask: mask containing eventually multiple classes
        :type mask: torch.Tensor
        :param class_id: id of the class we are interested in
        :type class_id: int
        :return: binary mask of the class and the boundary mask
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        boundary = (mask / 255).floor()
        mask[mask != class_id+1] = 0
        mask[mask == class_id+1] = 1

        return mask, boundary

    def __len__(self):
        return len(self.images) if self.train else 1000

    def __getitem__(self, idx):
        # The fold 0 val set has less than 1000 images.
        # Since our evaluation is based on 1000 images,
        # we need to loop over the val set to make it 1000
        idx %= len(self.images)
        offset = idx // len(self.images)

        query_name, support_names, class_sample = self.sample_episode(
            idx, offset)
        class_sample -= 1
        query_img, query_class_mask, support_imgs, support_class_masks, original_query_imgsize = self.load_frame(
            query_name, support_names)

        # Resizing the query image and masks to the desired size
        # e.g. the size accepted by the models (518x518 for example)
        query_img = self.transforms(query_img)
        if not self.use_original_imgsize:
            query_class_mask = F.interpolate(query_class_mask.unsqueeze(
                0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        # From the mask of the images, that may contain the ids of 1 or more
        # classes, we extract the mask of the class we are interested in, by
        # setting to 0 all the pixels that do not belong to the class and to 1
        # all the pixels that belong to the class
        query_mask, query_ignore_idx = self.extract_binary_mask(
            query_class_mask.float(), class_sample)

        # Applying the transformation to the support images
        support_imgs = torch.stack([self.transforms(img)
                                   for img in support_imgs])

        # Applying the transformation to the support masks
        support_masks = []
        support_ignore_idxs = []
        for support_mask in support_class_masks:
            support_mask = F.interpolate(support_mask.unsqueeze(0).unsqueeze(
                0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_binary_mask(
                support_mask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'query_ignore_idx': query_ignore_idx,

            'org_query_imgsize': original_query_imgsize,

            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'support_ignore_idxs': support_ignore_idxs,

            'class_id': torch.tensor(class_sample)
        }

        # target_np = self.load_seg_mask(self.targets[idx])
        # target_np = self.set_bg_pixel(target_np)

        return batch
