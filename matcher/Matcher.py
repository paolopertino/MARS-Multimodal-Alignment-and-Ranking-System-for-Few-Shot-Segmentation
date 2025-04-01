import os
import time
import random

from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ot
import math
import timm

from torchvision import transforms
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap


from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from utils.misc import norm, unnorm, pca
from utils.backbone_loader import BackboneLoader
from matcher.k_means import kmeans_pp


class Matcher:
    def __init__(
            self,
            encoder,
            encoder_transforms,
            use_encoder_registers=False,
            generator=None,
            input_size=518,
            num_centers=8,
            use_box=False,
            use_points_or_centers=True,
            sample_range=(4, 6),
            max_sample_iterations=30,
            alpha=1.,
            beta=0.,
            exp=0.,
            score_filter_cfg=None,
            num_merging_mask=10,
            use_negative_priors_from_discarded=False,
            use_negative_priors_from_cost=False,
            merge_prompt_types=False,
            visualize=False,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        # models
        self.encoder = encoder
        self.generator: SamAutomaticMaskGenerator = generator
        self.rps = None

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # transforms for image encoder
        self.encoder_transform = encoder_transforms
        self.use_encoder_registers = use_encoder_registers

        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks = None
        self.ref_masks_pool = None
        self.nshot = None

        # Used for storing the reference features and the target features.
        # In cascaded post refinement, we don't have to compute them anymore.
        self.unnormalized_ref_feats = None
        self.unnormalized_tar_feat = None
        self.stored_ref_feats = None
        self.stored_tar_feat = None

        self.encoder_img_size = None
        self.encoder_feat_size = None

        self.num_centers = num_centers
        self.use_box = use_box
        self.use_points_or_centers = use_points_or_centers
        self.sample_range = sample_range
        self.max_sample_iterations = max_sample_iterations

        self.alpha, self.beta, self.exp = alpha, beta, exp
        assert score_filter_cfg is not None
        self.score_filter_cfg = score_filter_cfg
        self.num_merging_mask = num_merging_mask

        self.use_negative_priors_from_discarded = use_negative_priors_from_discarded
        self.use_negative_priors_from_cost = use_negative_priors_from_cost
        self.merge_prompt_types = merge_prompt_types

        self.visualize = visualize
        self.visualization_parameters = None
        self.logger = None
        self.device = device

        # Matcher internal state logging
        self.S = None
        self.S_forward = None
        self.S_reverse = None
        self.sim_scores_after_forward_matching = None
        self.sim_scores_after_backward_matching = None
        self.sim_discarded_patches = None
        self.number_support_patches_forward_matching = None
        self.number_query_patches_forward_matching = None
        self.number_support_patches_backward_matching = None
        self.number_query_patches_backward_matching = None
        self.number_of_merged_masks = None
        self.number_of_points_used_for_prediction = None
        self.number_of_points_usable_for_prediction = None
        self.positive_points_inside_mask = None
        self.negative_points_inside_mask = None
        self.masks_to_merge = None
        self.unfiltered_generated_masks = None
        self.metric_filters = {}

    def set_reference(self, imgs, masks):
        """Fetches and store informations about the reference (support) images and masks.

        THe information regard the initial image size, the feature size once they will be 
        passed through the image encoder. Moreover it takes care of avarage pool the masks
        to reduce their size and to make them match the embeddings spatial dimensions.

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

        imgs = imgs.flatten(0, 1)  # bs, 3, h, w
        # since h=w then img_size = h = w = imgs.shape[-1]
        img_size = imgs.shape[-1]
        # checking that the image size is correct
        assert img_size == self.input_size[-1]
        # the feature size will be equal to the image size divided by the patch size
        feat_size = img_size // self.encoder.patch_size
        # since the first layer is using a convolution with kernelsize=stride=patch_size

        self.encoder_img_size = img_size
        self.encoder_feat_size = feat_size

        # process reference masks
        # checking if the masks contain actually 1s. If not, draw a square 14x14 of white pixels
        masks = reference_masks_verification(masks)
        masks = masks.permute(1, 0, 2, 3)  # ns, 1, h, w
        # the mask is resized to the feature dimension using
        ref_masks_pool = F.avg_pool2d(
            masks, (self.encoder.patch_size, self.encoder.patch_size))
        # average pooling.
        nshot = ref_masks_pool.shape[0]
        # mask threshold is set as a class variable to 0.0
        ref_masks_pool = (
            ref_masks_pool > self.generator.predictor.model.mask_threshold).float()
        ref_masks_pool = ref_masks_pool.reshape(-1)  # nshot, N

        self.ref_imgs = imgs
        self.ref_masks = masks
        self.ref_masks_pool = ref_masks_pool
        self.nshot = nshot

    def set_target(self, img):
        """Store the target image to process.

        :param img: target image on to which predict the mask.
        :type img: torch.Tensor
        """
        # getting the target image height and width.
        img_h, img_w = img.shape[-2:]

        # checking consistency with the Matcher accepted dimensions
        assert img_h == self.input_size[0] and img_w == self.input_size[1]

        # transform query to numpy as input of sam
        img_np = img.mul(255).byte()
        img_np = img_np.squeeze(0).permute(
            1, 2, 0).cpu().numpy()               # h x w x 3

        self.tar_img = img
        self.tar_img_np = img_np

    def set_rps(self):
        if self.rps is None:
            assert self.encoder_feat_size is not None
            self.rps = RobustPromptSampler(
                encoder_feat_size=self.encoder_feat_size,
                sample_range=self.sample_range,
                max_iterations=self.max_sample_iterations
            )

    def predict(self, target_mask=None):
        # ref_feats = number_of_shots * number_of_patches x C, tar_feats = number_of_patches x C
        ref_feats, tar_feat = self.extract_img_feats()

        # all_points = list of points in the target image that have been matched with the reference/support image.
        #              They are the candidates to be used as prompts. Passed using x,y coordinates. reduced_points_num x 2
        # box = bounding box of the matched points. Represented as the coordinates of the top-left and bottom-right corners.
        # S = similarity matrix, S = ns*N x N
        # C = cosine distance matrix, C = ns*N x N
        # reduced_points_num = number of matched points.
        all_points, negative_points, box, S, C, reduced_points_num, reduced_points_num_neg = self.patch_level_matching(
            ref_feats=ref_feats, tar_feat=tar_feat)

        # Visualize the internal state of the matcher.
        if self.visualize:
            self.visualize_internal_state(all_points, negative_points)

        # applies kmeans++ to the matched points in order to cluster them. The number of clusters is set as hyperparameter and set to 8. Only the centroids are returned.
        points = self.clustering(
            all_points) if not self.use_points_or_centers else all_points

        # Setup the Robust Prompt Sampler.
        self.set_rps()

        # Generate the masks. The parameters required are:
        # - self.tar_img_np: the target image of shape H x W x 3
        # - points: the cluster centers or the matched points
        # - box: the bounding box of the matched points if any, otherwise None
        # - all_points: the matched points in the patch level matching procedure.
        # - self.ref_masks_pool: the reference masks pool, i.e. the reference masks resized to the feature dimension throught average pooling.
        # - C: the cosine distance matrix, obtained through the cosine similarity matrix S as C = (1-S)/2.
        pred_masks, final_mask_scores = self.mask_generation(
            self.tar_img_np, points, box, all_points, self.ref_masks_pool, C, negative_points, target_mask=target_mask)
        return pred_masks, final_mask_scores

    def extract_img_feats(self):
        """Encodes, using the backbone encoder, both the support/reference images and the target image.

        encoder_transform contains 2 transforms: a MaybeToTensor (i.e. apply a ToTensor transform only if the input
        is not a tensor yet) and an image normalization (preprocessing required by the backbone).   

        `ref_feats` and `tar_feats` which are respectively the reference/support img features and the target img features
        are then normalized in order to be used later on for computing the similarity between the two.

        :return: the features extracted from the reference/support images and the features of the target image.
        :rtype: tuple
        """
        if self.stored_ref_feats is not None and self.stored_tar_feat is not None:
            return self.stored_ref_feats, self.stored_tar_feat

        ref_imgs = torch.cat([self.encoder_transform(rimg)[None, ...]
                             for rimg in self.ref_imgs], dim=0)
        tar_img = torch.cat([self.encoder_transform(timg)[None, ...]
                            for timg in self.tar_img], dim=0)

        if self.encoder.family.startswith('vits'):
            if not self.use_encoder_registers:
                ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))[
                    "x_prenorm"][:, 1:]  # BS X H*W X C, with 1: we are discarding the cls token
                tar_feat = self.encoder.forward_features(tar_img.to(self.device))[
                    "x_prenorm"][:, 1:]   # BS X H*W X C
            else:
                # Note that in the version with 4 registers we have few additional learnable tokens before the image tokens:
                #   * 1 CLS token
                #   * 4 register tokens (encoder.num_register_tokens = 4 by default)
                # Thus for extracting patch level features we start from the 5th token up to the last one.
                ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))[
                    'x_prenorm'][:, 1+self.encoder.num_register_tokens:]
                tar_feat = self.encoder.forward_features(tar_img.to(self.device))[
                    'x_prenorm'][:, 1+self.encoder.num_register_tokens:]
        elif self.encoder.family.startswith('convnets'):
            ref_feats = self.encoder(ref_imgs.to(self.device))
            tar_feat = self.encoder(tar_img.to(self.device))

        # ns, N, c = ref_feats.shape, so N is the number of patches
        ref_feats = ref_feats.reshape(-1, self.encoder.embed_dim)  # ns*N, c
        tar_feat = tar_feat.reshape(-1, self.encoder.embed_dim)  # N, c
        self.unnormalized_ref_feats = ref_feats.clone()
        self.unnormalized_tar_feat = tar_feat.clone()

        # normalize for cosine similarity
        ref_feats = F.normalize(ref_feats, dim=1, p=2)
        tar_feat = F.normalize(tar_feat, dim=1, p=2)
        self.stored_ref_feats = ref_feats
        self.stored_tar_feat = tar_feat

        return ref_feats, tar_feat

    def sample_negative_points_from_discarded(self, idxs_forward, sim_scores_forward, idxs_reverse, idxs_mask):
        # I want to retain the indices that are not matched to use them as negative priors.
        discarded_ind = torch.isin(idxs_reverse[1], idxs_mask, invert=True)

        indices_forward_neg = None
        dissim_scores_f = None
        if not (discarded_ind == False).all().item():
            indices_forward_neg = [
                idxs_forward[0][discarded_ind], idxs_forward[1][discarded_ind]]
            dissim_scores_f = sim_scores_forward[discarded_ind]
        inds_unmatched, sim_unmatched = indices_forward_neg, dissim_scores_f

        if indices_forward_neg is not None:
            reduced_points_num_neg = len(
                sim_unmatched) // 2 if len(sim_unmatched) > 40 else len(sim_unmatched)
            sim_sorted_neg, sim_idx_sorted_neg = torch.sort(
                sim_unmatched, descending=False)
            sim_filter_neg = sim_idx_sorted_neg[:reduced_points_num_neg]
            # These are the candidates negative priors
            points_unmatched_inds = indices_forward_neg[1][sim_filter_neg]
        else:
            reduced_points_num_neg = None
            points_unmatched_inds = None

        # Handling the negative priors
        if points_unmatched_inds is not None:
            points_unmatched_inds_set = torch.tensor(
                list(set(points_unmatched_inds.cpu().tolist())))  # removing duplicates from negative priors
            points_unmatched_inds_set_w = points_unmatched_inds_set % (
                self.encoder_feat_size)  # getting the x coordinate of the unmatched points
            points_unmatched_inds_set_h = points_unmatched_inds_set // (
                self.encoder_feat_size)
            idxs_mask_set_x_neg = (points_unmatched_inds_set_w *
                                   self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
            idxs_mask_set_y_neg = (points_unmatched_inds_set_h *
                                   self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
            points_unmatched = []
            for x, y in zip(idxs_mask_set_x_neg, idxs_mask_set_y_neg):
                if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                    points_unmatched.append([int(x), int(y)])
            negative_priors = np.array(points_unmatched)
        else:
            negative_priors = None

        return negative_priors, reduced_points_num_neg

    def sample_negative_points_from_cost(self, C):
        """
        Samples negative points from the cost matrix C.

        Parameters:
        C (torch.Tensor): The cost matrix of shape N x N, where N is the number of patches in the target image.

        Returns:
        points (numpy.ndarray): The sampled negative points in the target image.
        reduced_points_num (int): The number of reduced negative points after filtering.
        """
        C_forward = C.clone()

        # Perform forward matching to find the most similar patches in the target image for each patch in the reference image.
        indices_forward_neg = linear_sum_assignment(
            C_forward.cpu(), maximize=True)
        indices_forward_neg = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_forward_neg]
        cost_scores_forward = C_forward[indices_forward_neg[0],
                                        indices_forward_neg[1]]

        # Get the indices of the patches within the reference image mask.
        indices_mask = self.ref_masks_pool.flatten().nonzero()[:, 0]

        # Perform reverse matching to find the patches in the reference image that are most similar to each patch in the target image.
        C_reverse = C.t()[indices_forward_neg[1]]
        indices_reverse = linear_sum_assignment(C_reverse.cpu(), maximize=True)
        indices_reverse = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_reverse]
        retain_ind = torch.isin(indices_reverse[1], indices_mask, invert=True)

        # Keep only the negative points that are not within the reference image mask.
        indices_forward_neg_f = indices_forward_neg
        cost_scores_f = cost_scores_forward.clone()
        if not (retain_ind == False).all().item():
            indices_forward_neg = [
                indices_forward_neg_f[0][retain_ind], indices_forward_neg_f[1][retain_ind]]
            cost_scores_f = cost_scores_f[retain_ind]
        inds_neg_matched, cost_matched = indices_forward_neg_f, cost_scores_f

        # If there are more than 40 matched points, keep only half of them.
        reduced_points_num = len(
            cost_matched) // 2 if len(cost_matched) > 40 else len(cost_matched)
        cost_sorted, cost_idx_sorted = torch.sort(
            cost_matched, descending=True)
        cost_filter = cost_idx_sorted[:reduced_points_num]
        points_matched_inds = indices_forward_neg_f[1][cost_filter]

        # Remove duplicate points and convert the indices to the original image coordinates.
        points_matched_inds_set = torch.tensor(
            list(set(points_matched_inds.cpu().tolist())))
        points_matched_inds_set_w = points_matched_inds_set % (
            self.encoder_feat_size)
        points_matched_inds_set_h = points_matched_inds_set // (
            self.encoder_feat_size)
        idxs_mask_set_x = (points_matched_inds_set_w *
                           self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_y = (points_matched_inds_set_h *
                           self.encoder.patch_size + self.encoder.patch_size // 2).tolist()

        # Create the array of negative points in the original image coordinates.
        points_matched = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                points_matched.append([int(x), int(y)])
        points = np.array(points_matched)

        return points, reduced_points_num

    def patch_level_matching(self, ref_feats, tar_feat):
        """
        Performs patch-level matching between the reference and target image features.

        Parameters:
        ref_feats (torch.Tensor): Reference image features of shape ns*N x C.
        tar_feat (torch.Tensor): Target image features of shape N x C.

        Returns:
        ponits (numpy.ndarray): Matched points in the target image.
        negative_priors (list[numpy.ndarray]): Negative points in the target image.
        box (numpy.ndarray): Bounding box of the matched points.
        S (torch.Tensor): Cosine similarity matrix.
        C (torch.Tensor): Cosine distance matrix.
        reduced_points_num (int): Number of reduced points after filtering.
        reduced_points_num_neg (int): Number of reduced negative points after filtering.
        """
        # forward matching
        self.S = ref_feats @ tar_feat.t()  # S = ns*N x N

        # from cosine similarity ----> to cosine distance C = ns*N x N
        C = (1 - self.S) / 2

        # keeping only the points of the reference/support image that are within the support mask.
        self.S_forward = self.S[self.ref_masks_pool.flatten().bool()]
        number_support_patches = self.S_forward.shape[0]
        # S_forward = T x N, where T is the number of points within the mask.

        # Patches in the reference/support image feature(s) and the target image features are seen as nodes in a bipartite graph.
        # The similarity matrix S is used to compute the optimal matching between the two sets of nodes.
        indices_forward = linear_sum_assignment(
            self.S_forward.cpu(), maximize=True)

        # Indices forward will contain 2 tuples: the first tuple will contain the indices of the reference patches, the second tuple will contain the
        # indices of the target patches that have been matched. We first convert them to tensors and then from the similarity matrix S_forward we extract
        # the similarity scores of the matched patches.
        indices_forward = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_forward]
        self.number_support_patches_forward_matching = len(indices_forward[0])
        self.number_query_patches_forward_matching = len(indices_forward[1])
        # sim_scores_f = T, i.e. the similarity scores of the matched patches.
        sim_scores_f = self.S_forward[indices_forward[0], indices_forward[1]]
        self.sim_scores_after_forward_matching = sim_scores_f
        # self.ref_masks_pool.flatten() = ns*N, self.ref_masks_pool.flatten().nonzero() = T x 1,
        indices_mask = self.ref_masks_pool.flatten().nonzero()[:, 0]
        # self.ref_masks_pool.flatten().nonzero()[:, 0] = T, i.e. the indices of the patches within the mask.

        # reverse matching
        # S.t() = N x ns*N, S_reverse = K x ns*N. We are keeping only the points of the target image that have been matched.
        self.S_reverse = self.S.t()[indices_forward[1]]

        # indices_reverse will contain 2 tuples: the first tuple will contain the indices of the target patches,
        indices_reverse = linear_sum_assignment(
            self.S_reverse.cpu(), maximize=True)
        # the second tuple will contain the indices of the reference patches that have been matched.
        indices_reverse = [torch.as_tensor(
            index, dtype=torch.int64, device=self.device) for index in indices_reverse]
        # I want to retain only the indices of the reference/support patches that have been matched in the reverse matching and
        retain_ind = torch.isin(indices_reverse[1], indices_mask)

        # that are within the initial reference/support mask.
        indices_forward_pos = indices_forward
        indices_forward_neg = indices_forward
        sim_scores_f_pos = sim_scores_f.clone()
        sim_scores_f_neg = sim_scores_f.clone()
        self.number_support_patches_backward_matching = len(indices_forward[0])
        self.number_query_patches_backward_matching = len(indices_forward[1])
        if not (retain_ind == False).all().item():
            indices_forward_pos = [indices_forward[0]
                                   [retain_ind], indices_forward[1][retain_ind]]
            indices_forward_neg = [indices_forward[0]
                                   [~retain_ind], indices_forward[1][~retain_ind]]
            sim_scores_f_pos = sim_scores_f[retain_ind]
            sim_scores_f_neg = sim_scores_f[~retain_ind]
            self.number_support_patches_backward_matching = len(
                indices_forward[0][retain_ind])
            self.number_query_patches_backward_matching = len(
                indices_forward[1][retain_ind])
        else:
            print('[WARNING] - All the matched points have been discarded.')

        inds_matched, sim_matched = indices_forward_pos, sim_scores_f_pos
        self.sim_scores_after_backward_matching = sim_matched
        self.sim_discarded_patches = sim_scores_f_neg

        # if there are more than 40 matched points, we keep only half of them.
        reduced_points_num = len(
            sim_matched) // 2 if len(sim_matched) > 40 else len(sim_matched)
        sim_sorted, sim_idx_sorted = torch.sort(sim_matched, descending=True)
        sim_filter = sim_idx_sorted[:reduced_points_num]
        points_matched_inds = indices_forward_pos[1][sim_filter]
        points_unmatched_inds = indices_forward_neg[1]

        # removing duplicates from positive priors (for the few-shot scenario)
        points_matched_inds_set = torch.tensor(
            list(set(points_matched_inds.cpu().tolist())))
        points_unmatched_inds_set = torch.tensor(
            list(set(points_unmatched_inds.cpu().tolist())))

        # getting the x coordinate of the matched points
        points_matched_inds_set_w = points_matched_inds_set % self.encoder_feat_size
        points_unmatched_inds_set_w = points_unmatched_inds_set % self.encoder_feat_size
        # getting the y coordinate of the matched points
        points_matched_inds_set_h = points_matched_inds_set // self.encoder_feat_size
        points_unmatched_inds_set_h = points_unmatched_inds_set // self.encoder_feat_size

        # converting the x coordinate to the original image coordinate
        idxs_mask_set_x = (points_matched_inds_set_w *
                           self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_x_unmatched = (
            points_unmatched_inds_set_w * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        # converting the y coordinate to the original image coordinate
        idxs_mask_set_y = (points_matched_inds_set_h *
                           self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_y_unmatched = (
            points_unmatched_inds_set_h * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        # Note that the original image coordinate is the coordinate of the center of the patch.

        ponits_matched = []
        points_discarded = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                ponits_matched.append([int(x), int(y)])

        for x, y in zip(idxs_mask_set_x_unmatched, idxs_mask_set_y_unmatched):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                points_discarded.append([int(x), int(y)])

        ponits = np.array(ponits_matched)
        points_discarded = np.array(points_discarded)

        # Sampling negative points from the discarded points and from the cost matrix.
        # The negative points are used as negative priors in the mask generation.
        negative_priors = []
        reduced_points_num_neg = []
        if self.use_negative_priors_from_discarded:
            negative_priors_from_discarded, reduced_points_num_neg_from_discarded = self.sample_negative_points_from_discarded(
                indices_forward, sim_scores_f, indices_reverse, indices_mask)
            negative_priors.append(negative_priors_from_discarded)
            reduced_points_num_neg.append(
                reduced_points_num_neg_from_discarded)
        if self.use_negative_priors_from_cost:
            negative_priors_from_cost, reduced_points_num_neg_from_cost = self.sample_negative_points_from_cost(
                C)
            negative_priors.append(negative_priors_from_cost)
            reduced_points_num_neg.append(reduced_points_num_neg_from_cost)

        # In case of bounding box prompts are added, the box is computed as the bounding box of the matched points.
        if self.use_box:
            box = np.array([
                max(ponits[:, 0].min(), 0),
                max(ponits[:, 1].min(), 0),
                min(ponits[:, 0].max(), self.input_size[1] - 1),
                min(ponits[:, 1].max(), self.input_size[0] - 1),
            ])
        else:
            box = None

        return ponits, negative_priors if len(negative_priors) > 0 else points_discarded, box, self.S, C, reduced_points_num, reduced_points_num_neg

    def clustering(self, points):

        num_centers = min(self.num_centers, len(points))
        flag = True
        while (flag):
            centers, cluster_assignment = kmeans_pp(points, num_centers)
            id, fre = torch.unique(cluster_assignment, return_counts=True)
            if id.shape[0] == num_centers:
                flag = False
            else:
                print('Kmeans++ failed, re-run')
        centers = np.array(centers).astype(np.int64)
        return centers

    def get_ref_to_target_similarity(self, ref_feats, tar_feat, ref_masks_pool):
        # Making it work for one-shot. Later on for few-shot
        # ref_feats.shape = ns*N x C, tar_feat.shape = N x C, ref_masks_pool.shape = ns x N
        ref_feats = ref_feats.reshape(
            self.nshot, -1, ref_feats.shape[-1])  # ns x N x C
        tar_feat = tar_feat.unsqueeze(0)  # 1 x N x C

        # Applying the ref mask to the ref image to extract the patch embeddings corresponding to the mask.
        ref_feats_masked = ref_feats[ref_masks_pool.bool()]  # T x C

        # Getting the similarity between the masked ref patches and the target patches.
        # ref_to_target_similarity.shape = N x T
        # 1 x N x C @ C x T = 1 x N x T
        ref_to_target_similarity = tar_feat @ ref_feats_masked.t()

        # Get the mean, i.e., a vector of dimensions 1xN.
        ref_to_target_similarity = ref_to_target_similarity.mean(dim=-1)

        return ref_to_target_similarity

    def get_negative_point_priors(self, similarity):
        pass
    
    def get_purity_filter(self):
        return self.metric_filters['purity']

    def mask_generation(self, tar_img_np, points, box, all_ponits, ref_masks_pool, C, negative_points=None, target_mask: torch.Tensor = None):
        """Generates the masks for the target image using the points as prompts.

        :param tar_img_np: target image of shape H x W x 3
        :type tar_img_np: np.ndarray
        :param points: cluster centers or matched points
        :type points: np.ndarray
        :param box: bounding box of the matched points
        :type box: np.ndarray
        :param all_ponits: matched points in the patch level matching procedure
        :type all_ponits: np.ndarray
        :param ref_masks_pool: reference masks pool, i.e. the reference masks resized to the feature dimension throught average pooling
        :type ref_masks_pool: torch.Tensor
        :param C: cosine distance matrix
        :type C: torch.Tensor
        :param target_mask: already present target mask (supposed to come from past iterations). Defaults to None, i.e., no target mask available.
        :type target_mask: torch.Tensor
        :return: the predicted masks
        :rtype: torch.Tensor
        """
        # points here are the cluster centers or all the points. All points are the ones got from the bipartite matching.
        # num_prompts = 0
        samples_list = []
        label_list = []
        if self.use_negative_priors_from_discarded or self.use_negative_priors_from_cost:
            # If using negative points priors, we add them to the samples_list and label_list.
            for negative_pts in negative_points:
                if negative_pts is not None and len(negative_pts) > 0:
                    sampled_pts, sampled_pts_labels = self.rps.sample_points(
                        points, negative_points=negative_pts)
                else:
                    sampled_pts, sampled_pts_labels = self.rps.sample_points(
                        points)
                samples_list.extend(sampled_pts)
                label_list.extend(sampled_pts_labels)

            if self.merge_prompt_types:
                sampled_pts, sampled_pts_labels = self.rps.sample_points(
                    points)
                samples_list.extend(sampled_pts)
                label_list.extend(sampled_pts_labels)

        else:
            samples_list, label_list = self.rps.sample_points(points)

        # sample list contains the possible combinations of cluster centers that can be used as prompts.
        # - tar_img_np: the target image of shape H x W x 3
        # - samples_list: the possible combinations of cluster centers (or drawn points) that can be used as prompts
        # - label_list: the labels of the cluster centers (or drawn points) (all 1s since we are using only foreground points)
        # - box: the bounding box of the matched points if any, otherwise None
        # The generate method, given some prompts, generates the masks for the target image.
        tar_masks_ori = self.generator.generate(
            tar_img_np,
            select_point_coords=samples_list,
            select_point_labels=label_list,
            select_box=[box] if self.use_box else None,
            select_mask_input=target_mask.cpu().numpy() if target_mask is not None else None,
        )

        # task_mask_ori should be a list of dictionaries (one dictionary per mask proposal).
        # The segmentation key of each dictionary contains the mask of the target image and should be an array of dimension H*W.
        tar_masks = torch.cat(
            [torch.from_numpy(qmask['segmentation']).float()[None, None, ...].to(self.device) for
             qmask in tar_masks_ori], dim=0).cpu().numpy() > 0  # masks are converted to boolean values - nmask x 1 x H x W
        # list[list[list[floats]]] - num_generated_masks x num_points x 2
        tar_masks_point_coords = [qmask['point_coords']
                                  for qmask in tar_masks_ori]

        # append to original results
        purity = torch.zeros(tar_masks.shape[0])
        coverage = torch.zeros(tar_masks.shape[0])
        emd = torch.zeros(tar_masks.shape[0])

        samples = samples_list[-1]
        labels = torch.ones(tar_masks.shape[0], samples.shape[1])  # nmask x 6
        samples = torch.ones(
            tar_masks.shape[0], samples.shape[1], 2)  # nmask x 6 x 2

        # compute scores for each mask
        for i in range(len(tar_masks)):
            purity_, coverage_, emd_, sample_, label_, mask_ = \
                self.rps.get_mask_scores(
                    points=points,
                    masks=tar_masks[i],
                    all_points=all_ponits,
                    emd_cost=C,
                    ref_masks_pool=ref_masks_pool
                )
            assert np.all(mask_ == tar_masks[i])
            purity[i] = purity_
            coverage[i] = coverage_
            emd[i] = emd_

        pred_masks = tar_masks.squeeze(1)  # nmask x H x W
        metric_preds = {
            "purity": purity,
            "coverage": coverage,
            "emd": emd
        }

        scores = self.alpha * emd + self.beta * \
            purity * coverage ** self.exp  # nmask x 1

        def check_pred_mask(pred_masks):
            if len(pred_masks.shape) < 3:  # avoid only one mask
                pred_masks = pred_masks[None, ...]
            return pred_masks

        pred_masks = check_pred_mask(pred_masks)  # nmask x H x W
        self.unfiltered_generated_masks = torch.tensor(
            pred_masks.copy() > 0, dtype=torch.float)

        # filter the false-positive mask fragments by using the proposed metrics
        for metric in ["coverage", "emd", "purity"]:
            if self.score_filter_cfg[metric] > 0:
                thres = min(
                    self.score_filter_cfg[metric], metric_preds[metric].max())
                idx = torch.where(metric_preds[metric] >= thres)[0]
                self.metric_filters[metric] = idx
                scores = scores[idx]
                samples = samples[idx]
                labels = labels[idx]
                pred_masks = check_pred_mask(pred_masks[idx])
                tar_masks_point_coords = [
                    tar_masks_point_coords[int(i.item())] for i in idx]

                for key in metric_preds.keys():
                    metric_preds[key] = metric_preds[key][idx]

        #  score-based masks selection, masks merging
        if self.score_filter_cfg["score_filter"]:
            distances = 1 - scores  # nmask x 1
            distances, rank = torch.sort(distances, descending=False)
            distances_norm = distances - distances.min()
            distances_norm = distances_norm / (distances.max() + 1e-6)
            filer_dis = distances < self.score_filter_cfg["score"]
            filer_dis[..., 0] = True
            filer_dis_norm = distances_norm < self.score_filter_cfg["score_norm"]
            filer_dis = filer_dis * filer_dis_norm

            pred_masks = check_pred_mask(pred_masks)
            self.number_of_masks_before_score_filtering = min(
                len(pred_masks), self.num_merging_mask)
            masks = pred_masks[rank[filer_dis][:self.num_merging_mask]]
            tar_masks_point_coords = [tar_masks_point_coords[int(
                i.item())] for i in rank[filer_dis][:self.num_merging_mask]]
            used_points_for_prediction = set(
                tuple(point) for mask_points in tar_masks_point_coords for point in mask_points)
            self.number_of_points_used_for_prediction = len(
                used_points_for_prediction)
            self.number_of_points_usable_for_prediction = len(all_ponits)

            self.number_of_merged_masks = len(masks)
            masks = check_pred_mask(masks)
            self.masks_to_merge = torch.tensor(
                masks.copy() > 0, dtype=torch.float)

            masks = masks.sum(0) > 0 # HXW

            # Counting the number of both positive and negative points that fall inside the predicted mask.
            self.positive_points_inside_mask = len(
                [point for point in all_ponits if masks[point[1], point[0]]])
            self.negative_points_inside_mask = len(
                [point for point in negative_points if masks[point[1], point[0]]])

            masks = masks[None, ...]

            final_mask_score = scores[rank[filer_dis]
                                      [:self.num_merging_mask]].mean()
        else:
            # we either merge the top num_merging_mask masks or all the masks if they are less than num_merging_mask.
            topk = min(self.num_merging_mask, scores.size(0))
            self.number_of_masks_before_score_filtering = topk
            topk_idx = scores.topk(topk)[1]
            topk_samples = samples[topk_idx].cpu().numpy()
            topk_scores = scores[topk_idx].cpu().numpy()
            topk_pred_masks = pred_masks[topk_idx]
            topk_pred_masks = check_pred_mask(topk_pred_masks)
            tar_masks_point_coords = [
                tar_masks_point_coords[int(i.item())] for i in topk_idx]

            if self.score_filter_cfg["topk_scores_threshold"] > 0:
                # map scores to 0-1
                topk_scores = topk_scores / (topk_scores.max())

            idx = topk_scores > self.score_filter_cfg["topk_scores_threshold"]
            topk_samples = topk_samples[idx]

            topk_pred_masks = check_pred_mask(topk_pred_masks)
            topk_pred_masks = topk_pred_masks[idx]
            tar_masks_point_coords = [tar_masks_point_coords[i]
                                      for i, j in enumerate(idx) if j != False]
            used_points_for_prediction = set(
                tuple(point) for mask_points in tar_masks_point_coords for point in mask_points)
            self.number_of_points_used_for_prediction = len(
                used_points_for_prediction)
            self.number_of_points_usable_for_prediction = len(all_ponits)
            mask_list = []
            for i in range(len(topk_samples)):
                mask = topk_pred_masks[i][None, ...]
                mask_list.append(mask)
            self.masks_to_merge = torch.tensor(
                np.array(mask_list).copy() > 0, dtype=torch.float)
            self.number_of_merged_masks = len(mask_list)
            masks = np.sum(mask_list, axis=0) > 0  # H x W

            # Counting the number of both positive and negative points that fall inside the predicted mask.
            self.positive_points_inside_mask = len(
                [point for point in all_ponits if masks[point[1], point[0]]])
            self.negative_points_inside_mask = len(
                [point for point in negative_points if masks[point[1], point[0]]])

            masks = check_pred_mask(masks)  # 1 x H x W
            final_mask_score = topk_scores[idx].mean()

        return torch.tensor(masks, device=self.device, dtype=torch.float), final_mask_score

    def positive_prompts_experiment(self):
        self.merge_prompt_types = False
        self.use_negative_priors_from_discarded = False
        self.use_negative_priors_from_cost = False

    def negative_prompts_from_discarded_experiment(self):
        self.merge_prompt_types = False
        self.use_negative_priors_from_discarded = True
        self.use_negative_priors_from_cost = False

    def negative_prompts_from_cost_experiment(self):
        self.merge_prompt_types = False
        self.use_negative_priors_from_discarded = False
        self.use_negative_priors_from_cost = True

    def set_visualizer_parameters(
            self, query_mask: torch.tensor,
            class_idx: int,
            batch_idx: int,
            fold_num: int,
            idx2label: dict,
            experiment_name: str,
            save_plots=True):
        self.visualization_parameters = {
            "tar_mask": query_mask,
            "class_idx": class_idx.item(),
            "batch_idx": batch_idx,
            "fold_num": fold_num,
            "idx2label": idx2label,
            "base_path": f'./vis/internal_state/{experiment_name}/fold{fold_num}/{class_idx.item()}-{idx2label[class_idx.item()]}/',
            "save_plots": save_plots
        }

        if save_plots and not os.path.exists(self.visualization_parameters["base_path"]):
            os.makedirs(self.visualization_parameters["base_path"])

    def visualize_internal_state(self, positive_prompts, negative_prompts, debug=False):
        def upsample_similarity(similarity, original_size, patch_size):
            factor = original_size[0] // (similarity.shape[0] * patch_size)
            upsampled = zoom(similarity, factor, order=1)
            upsampled = upsampled[:original_size[0], :original_size[1]]
            return upsampled

        def blend_images(image, heatmap, alpha=0.7):
            # Ensure heatmap has the same shape as the image
            if heatmap.shape[:2] != image.shape[:2]:
                heatmap = zoom(
                    heatmap, (image.shape[0] / heatmap.shape[0], image.shape[1] / heatmap.shape[1]), order=1)

            # Normalize heatmap to [0, 1]
            heatmap_normalized = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min())

            # Create RGB heatmap
            heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]

            # Blend original image with heatmap
            blended = alpha * image + (1 - alpha) * heatmap_colored

            # Ensure values are in [0, 1] range
            blended = np.clip(blended, 0, 1)

            return blended

        def apply_mask(image, mask, color, alpha=0.5):
            r""" Apply mask to the given image. """
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c],
                                          image[:, :, c])
            return image

        @torch.no_grad()
        def prepare_feats(features: torch.Tensor):
            pca_feats, _ = pca([features])

            return pca_feats

        colors = {'red': (1, 0.196, 0.196), 'blue': (0.4, 0.549, 1)}

        if debug:
            print(f"""
                Similarity matrix shape: {self.S_forward.shape}
                Support images shape: {self.ref_imgs.shape}
                Target image shape: {self.tar_img.shape}
                Support masks shape: {self.ref_masks.shape}
                Positive prompts shape: {len(positive_prompts)}
                Negative prompts shape: {len(negative_prompts)}
            """)

        # Converting support image and query image to numpy.
        support_image = self.ref_imgs[0].permute(1, 2, 0).cpu().numpy()
        # BS x C x H x W --> C x H x W --> H x W x C --> H x W
        support_mask = self.ref_masks[0].permute(
            1, 2, 0)[:, :, 0].cpu().numpy()
        query_image = self.tar_img_np / 255

        embedding_shape = int(np.sqrt(self.S_forward.shape[-1]))
        support_features = self.stored_ref_feats.reshape(
            embedding_shape, embedding_shape, -1).permute(2, 0, 1).cpu().unsqueeze(0)
        target_features = self.stored_tar_feat.reshape(
            embedding_shape, embedding_shape, -1).permute(2, 0, 1).cpu().unsqueeze(0)

        similarity_matrix_forward = self.S_forward.mean(dim=0).reshape(
            (embedding_shape, embedding_shape)).cpu().numpy()
        similarity_matrix_reverse = self.S_reverse.mean(dim=0).reshape(
            (embedding_shape, embedding_shape)).cpu().numpy()

        # Upsample similarity matrices to match original image size
        upsampled_similarity_support = upsample_similarity(
            similarity_matrix_reverse, support_image.shape[:2], 14)
        upsampled_similarity_target = upsample_similarity(
            similarity_matrix_forward, query_image.shape[:2], 14)

        # Blend images with heatmaps
        blended_support_img_with_similarity = blend_images(
            support_image, upsampled_similarity_support)
        blended_target_img_with_similarity = blend_images(
            query_image, upsampled_similarity_target)

        # Create the figure and axes
        fig, axs = plt.subplots(2, 5, figsize=(20, 10))

        # Showing the support image with its mask
        axs[0][0].imshow(apply_mask(
            support_image, support_mask, colors['blue']))
        axs[0][0].axis('off')
        axs[0][0].set_title('Support image with mask')

        # Showing the query image
        axs[0][1].imshow(query_image)
        axs[0][1].axis('off')
        axs[0][1].set_title('Query image')

        axs[0][2].imshow(apply_mask(
            query_image, self.visualization_parameters['tar_mask'].cpu().numpy(), colors['red']))
        axs[0][2].axis('off')
        axs[0][2].set_title('Query image with mask')

        # Showing the support features
        [prepared_support_feats] = prepare_feats(support_features)
        axs[0][3].imshow(prepared_support_feats[0].permute(
            1, 2, 0).detach().cpu())
        axs[0][3].axis('off')
        axs[0][3].set_title('Support features')

        # Showing the query features
        [prepared_target_feats] = prepare_feats(target_features)
        axs[0][4].imshow(prepared_target_feats[0].permute(
            1, 2, 0).detach().cpu())
        axs[0][4].axis('off')
        axs[0][4].set_title('Target features')

        # Showing the Support image blended with the similarity
        axs[1][0].imshow(blended_support_img_with_similarity)
        axs[1][0].axis('off')
        axs[1][0].set_title('Support image similarity with target')

        # Showing the target image blended with the similarity
        axs[1][1].imshow(blended_target_img_with_similarity)
        axs[1][1].axis('off')
        axs[1][1].set_title('Query image similarity with support')

        # Showing the forward similarity matrix
        axs[1][2].imshow(upsampled_similarity_target)
        axs[1][2].axis('off')
        axs[1][2].set_title('Forward similarity matrix')

        # Showing the backward similarity matrix
        axs[1][3].imshow(upsampled_similarity_support)
        axs[1][3].axis('off')
        axs[1][3].set_title('Backward similarity matrix')

        # Showing the target image with points on it
        axs[1][4].imshow(query_image)
        axs[1][4].axis('off')
        axs[1][4].set_title('Query image with prompts')

        # For each positive prior draw a red point
        if positive_prompts is not None:
            for point in positive_prompts:
                axs[1][4].scatter(point[0], point[1], color='red', s=1)

        colors_negative_priors = ["blue", "green"]
        if negative_prompts is not None:
            for i, neg_prompts in enumerate(negative_prompts):
                if neg_prompts is not None:
                    for point in neg_prompts:
                        axs[1][4].scatter(point[0], point[1],
                                          color=colors_negative_priors[i], s=1)

        # Saving the whole plot
        if self.visualization_parameters['save_plots']:
            plot_name = f"IS-fold{self.visualization_parameters['fold_num']}_{self.visualization_parameters['batch_idx']}_class{self.visualization_parameters['class_idx']}-{self.visualization_parameters['idx2label'][self.visualization_parameters['class_idx']]}.png"
            fig.savefig(os.path.join(
                self.visualization_parameters['base_path'], plot_name))
            if self.logger is not None:
                self.logger.log_figure(figure=fig, name=plot_name)
            plt.close(fig)
        else:
            plt.show()

    def get_similarities(self):
        return self.S_forward, self.S_reverse, self.sim_scores_after_forward_matching, self.sim_scores_after_backward_matching, self.sim_discarded_patches

    def get_patch_matching_statistics(self) -> dict:
        result = {
            'number_support_patches_forward_matching': self.number_support_patches_forward_matching,
            'number_support_patches_backward_matching': self.number_support_patches_backward_matching,
            'number_discarded_support_patches': self.number_support_patches_forward_matching - self.number_support_patches_backward_matching,
            'number_query_patches_forward_matching': self.number_query_patches_forward_matching,
            'number_query_patches_backward_matching': self.number_query_patches_backward_matching,
            'number_discarded_query_patches': self.number_query_patches_forward_matching - self.number_query_patches_backward_matching
        }
        return result

    def get_mask_generation_statistics(self) -> dict:
        result = {
            'number_of_merged_masks': self.number_of_merged_masks,
            'number_of_masks_before_score_filtering': self.number_of_masks_before_score_filtering,
            'ratio_of_merged_masks': self.number_of_merged_masks / self.number_of_masks_before_score_filtering,
            'number_of_points_usable_for_prediction': self.number_of_points_usable_for_prediction,
            'number_of_points_used_for_prediction': self.number_of_points_used_for_prediction,
            'ratio_points_used_vs_usable': self.number_of_points_used_for_prediction / self.number_of_points_usable_for_prediction,
            'positive_points_inside_mask': self.positive_points_inside_mask,
            'negative_points_inside_mask': self.negative_points_inside_mask,
            'ratio_negative_vs_positive_points_inside_mask': self.negative_points_inside_mask / max(1, self.positive_points_inside_mask),
            'ratio_positive_points_inside_mask_vs_usable_points': self.positive_points_inside_mask / self.number_of_points_usable_for_prediction
        }

        return result

    def get_aposteriori_statistics(self, mask: torch.Tensor):
        target_mask_pooled = F.avg_pool2d(
            mask, (self.encoder.patch_size, self.encoder.patch_size))
        target_mask_pooled = target_mask_pooled > self.generator.predictor.model.mask_threshold
        target_mask_pooled = target_mask_pooled.reshape(-1)
        reference_mask_pooled = self.ref_masks_pool.bool()
        similarity_matrix = self.S[reference_mask_pooled,
                                   :][:, target_mask_pooled]
        ref_feats_prototype = self.unnormalized_ref_feats[reference_mask_pooled].mean(
            dim=0)
        tar_feats_prototype = self.unnormalized_tar_feat[target_mask_pooled].mean(
            dim=0)

        result = {
            'aposteriori_similarity_mean': similarity_matrix.mean().item(),
            'aposteriori_similarity_max': similarity_matrix.max().item() if similarity_matrix.numel() > 0 else 0,
            'aposteriori_similarity_std': similarity_matrix.std().item(),
            'embeddings_euclidean_distance': torch.norm(ref_feats_prototype - tar_feats_prototype, p=2).item() if ref_feats_prototype.numel() > 0 and tar_feats_prototype.numel() > 0 else 0,
        }

        return result

    def get_masks_to_merge(self):
        return self.masks_to_merge

    def get_unfiltered_generated_masks(self):
        return self.unfiltered_generated_masks

    def clear(self):
        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None

        self.encoder_img_size = None
        self.encoder_feat_size = None

        self.unnormalized_ref_feats = None
        self.unnormalized_tar_feat = None
        self.stored_ref_feats = None
        self.stored_tar_feat = None

        # Matcher internal state reset
        self.S = None
        self.S_forward = None
        self.S_reverse = None
        self.sim_scores_after_forward_matching = None
        self.sim_scores_after_backward_matching = None
        self.sim_discarded_patches = None
        self.number_support_patches_forward_matching = None
        self.number_query_patches_forward_matching = None
        self.number_support_patches_backward_matching = None
        self.number_query_patches_backward_matching = None
        self.number_of_merged_masks = None
        self.number_of_masks_before_score_filtering = None
        self.number_of_points_used_for_prediction = None
        self.number_of_points_usable_for_prediction = None
        self.positive_points_inside_mask = None
        self.negative_points_inside_mask = None
        self.masks_to_merge = None
        self.unfiltered_generated_masks = None
        self.metric_filters = {}

        self.generator.reset_stored_features()

    def set_logger(self, logger):
        self.logger = logger


class RobustPromptSampler:

    def __init__(
        self,
        encoder_feat_size,
        sample_range,
        max_iterations
    ):
        self.encoder_feat_size = encoder_feat_size
        self.sample_range = sample_range
        self.max_iterations = max_iterations

    def get_mask_scores(self, points, masks, all_points, emd_cost, ref_masks_pool):
        """Get various mask scores for a given proposed mask.

        :param points: cluster centers
        :type points: np.ndarray
        :param masks: proposed mask by SAM
        :type masks: np.ndarray
        :param all_points: matched points in the patch level matching procedure
        :type all_points: np.ndarray
        :param emd_cost: EMD cost matrix. C = (1-S) / 2
        :type emd_cost: torch.Tensor
        :param ref_masks_pool: reference masks pool, i.e. the reference masks resized to the feature dimension throught average pooling
        :type ref_masks_pool: torch.Tensor
        """

        def is_in_mask(point, mask):
            # input: point: n*2, mask: h*w
            # output: n*1
            h, w = mask.shape
            point = point.astype(np.int64)
            point = point[:, ::-1]  # y,x
            point = np.clip(point, 0, [h - 1, w - 1])
            return mask[point[:, 0], point[:, 1]]

        ori_masks = masks
        masks = cv2.resize(
            masks[0].astype(np.float32),
            (self.encoder_feat_size, self.encoder_feat_size),
            interpolation=cv2.INTER_AREA)
        if masks.max() <= 0:
            thres = masks.max() - 1e-6
        else:
            thres = 0
        masks = masks > thres

        # 1. emd
        emd_cost_pool = emd_cost[ref_masks_pool.flatten(
        ).bool(), :][:, masks.flatten()]
        emd = ot.emd2(a=[1. / emd_cost_pool.shape[0] for i in range(emd_cost_pool.shape[0])],
                      b=[1. / emd_cost_pool.shape[1]
                          for i in range(emd_cost_pool.shape[1])],
                      M=emd_cost_pool.cpu().numpy())
        emd_score = 1 - emd

        labels = np.ones((points.shape[0],))

        # 2. purity and coverage
        assert all_points is not None
        points_in_mask = is_in_mask(all_points, ori_masks[0])
        points_in_mask = all_points[points_in_mask]
        # here we define two metrics for local matching , purity and coverage
        # purity: points_in/mask_area, the higher means the denser points in mask
        # coverage: points_in / all_points, the higher means the mask is more complete
        mask_area = max(float(masks.sum()), 1.0)
        purity = points_in_mask.shape[0] / mask_area
        coverage = points_in_mask.shape[0] / all_points.shape[0]
        purity = torch.tensor([purity]) + 1e-6
        coverage = torch.tensor([coverage]) + 1e-6
        return purity, coverage, emd_score, points, labels, ori_masks

    def combinations(self, n, k):
        # generate a combination of k elements from a set of n integers.
        if k > n:
            return []
        if k == 0:
            return [[]]
        if k == n:
            return [[i for i in range(n)]]
        res = []
        for i in range(n):
            for j in self.combinations(i, k - 1):
                res.append(j + [i])
        return res

    def sample_points(self, points, negative_points=None):
        # return list of arrary

        sample_list = []
        label_list = []

        # By default, sample_range[0] = 4 and sample_range[1] = 6
        # In the example script call by the authors sample_range = (1, 6)
        for i in range(min(self.sample_range[0], len(points)), min(self.sample_range[1], len(points)) + 1):
            # 8 is the number of clusters passed to kmeans, so most likely the number of cluster centers. If we use all the matched points len(points) > 8.
            if len(points) > 8:
                # max_iterations defaults to 30. The example script call by the authors max_iterations = 64
                index = [random.sample(range(len(points)), i)
                         for j in range(self.max_iterations)]
                # (max_iterations * i) * 2
                sample = np.take(points, index, axis=0)

                sample_neg = None
                label_neg = None
                if negative_points is not None:
                    if len(negative_points) > 8:
                        index_neg = [random.sample(
                            range(len(negative_points)), i) for j in range(self.max_iterations)]
                    else:
                        index_neg = [random.choices(
                            range(len(negative_points)), k=i) for j in range(self.max_iterations)]
                    sample_neg = np.take(negative_points, index_neg, axis=0)
                    label_neg = np.zeros((sample_neg.shape[0], i))
            else:
                # what are the combinations of i points (ranging from 1 to 6) from the number of cluster centers.
                index = self.combinations(len(points), i)
                # So, if we have 5 cluster centers (or general points), we will have 5 combinations of 4 points and 1 combination of 5 points.
                # It is the binomial coefficient (len(points)   i)
                sample = np.take(points, index, axis=0)  # i * n * 2

                sample_neg = None
                label_neg = None
                if negative_points is not None:
                    index_neg = [random.choices(
                        range(len(negative_points)), k=i) for j in range(len(index))]
                    sample_neg = np.take(negative_points, index_neg, axis=0)
                    label_neg = np.zeros((sample_neg.shape[0], i))

                # If we are using negative priors, we also need to generate the negative samples. In the same number of positive samples.

            # generate label  max_iterations * i
            # So, we are taking in the sample_list, all the possible combinations of i points from the cluster centers (if len(points) <= 8).
            # As labels, since we are using only foreground points, we are setting the labels to 1.

            label = np.ones((sample.shape[0], i))
            sample_list.append(sample)
            label_list.append(label)

            if sample_neg is not None and label_neg is not None:
                sample_list.append(sample_neg)
                label_list.append(label_neg)
                assert sample.shape[0] == sample_neg.shape[0]

        concatenated_points = []
        concatenated_labels = []
        if negative_points is not None:
            for i in range(0, len(sample_list), 2):
                concatenated_points.append(
                    np.hstack((sample_list[i], sample_list[i+1])))
                concatenated_labels.append(
                    np.hstack((label_list[i], label_list[i+1])))

            return concatenated_points, concatenated_labels

        return sample_list, label_list


def build_matcher_oss(args):

    # DINOv2, Image Encoder
    dino_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        num_register_tokens=args.dinov2_num_register_tokens,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    ) if args.backbone == "dinov2" else None

    encoder, encoder_transforms = BackboneLoader.load_backbone(
        backbone_name=args.backbone,
        backbone_size=args.backbone_size,
        device=args.device,
        backbone_weights_path=args.backbone_weights,
        encoder_kwargs=dino_kwargs
    )

    # SAM
    sam = sam_model_registry[args.sam_size](checkpoint=args.sam_weights)
    sam.to(device=args.device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=64,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=1.0,
        sel_stability_score_thresh=args.sel_stability_score_thresh,
        sel_pred_iou_thresh=args.iou_filter,
        box_nms_thresh=args.box_nms_thresh,
        sel_output_layer=args.output_layer,
        output_layer=args.dense_multimask_output,
        dense_pred=args.use_dense_mask,
        multimask_output=args.dense_multimask_output > 0,
        sel_multimask_output=args.multimask_output > 0,
    )

    score_filter_cfg = {
        "emd": args.emd_filter,
        "purity": args.purity_filter,
        "coverage": args.coverage_filter,
        "score_filter": args.use_score_filter,
        "score": args.deep_score_filter,
        "score_norm": args.deep_score_norm_filter,
        "topk_scores_threshold": args.topk_scores_threshold
    }

    return Matcher(
        encoder=encoder,
        encoder_transforms=encoder_transforms,
        use_encoder_registers=args.dinov2_num_register_tokens > 0,
        input_size=args.img_size,
        generator=generator,
        num_centers=args.num_centers,
        use_box=args.use_box,
        use_points_or_centers=args.use_points_or_centers,
        sample_range=args.sample_range,
        max_sample_iterations=args.max_sample_iterations,
        alpha=args.alpha,
        beta=args.beta,
        exp=args.exp,
        score_filter_cfg=score_filter_cfg,
        num_merging_mask=args.num_merging_mask,
        use_negative_priors_from_discarded=args.use_negative_priors_from_discarded,
        use_negative_priors_from_cost=args.use_negative_priors_from_cost,
        merge_prompt_types=args.merge_prompt_types,
        visualize=args.visualize != 0,
        device=args.device
    )
