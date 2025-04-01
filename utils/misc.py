# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu) | Edited by Paolo Pertino
# --------------------------------------------------------
import math

from collections import defaultdict, deque

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from sklearn.decomposition import PCA
from pytorch_lightning import seed_everything


# HACK for evalution 
def hook_metadata(metadata, name):
    if name == 'cityscapes_fine_sem_seg_val':
        metadata.__setattr__("keep_sem_bgd", False)
    return metadata

def hook_opt(model, name):
    if name in ['cityscapes_fine_panoptic_val', 'ade20k_panoptic_val', 'bdd10k_40_panoptic_val', 'cityscapes_fine_panoptic_val', 'scannet_21_panoptic_val']:
        model.model.object_mask_threshold = 0.4
    else:
        model.model.object_mask_threshold = 0.8

# HACK for evalution 
def hook_switcher(model, name):
    mappings = {}
    if name in ['cityscapes_fine_sem_seg_val', 'scannet_21_val_seg', 'scannet_38_val_seg', 'scannet_41_val_seg', 'sunrgbd_37_val_seg', 'bdd10k_val_sem_seg', 'ade20k_full_sem_seg_val']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False}
    elif name in ['cityscapes_fine_instance_seg_val', 'pascal_part_val_interactive', 'pascal_part_val', 'pascal_part_train'] or 'seginw' in name:
        mappings = {'SEMANTIC_ON': False, 'INSTANCE_ON': True, 'PANOPTIC_ON': False}
    elif name in ['cityscapes_fine_panoptic_val', 'scannet_21_panoptic_val', 'bdd10k_40_panoptic_val']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': True}
    elif 'coco_2017_val_panoptic_with_sem_seg' in name or name in ['ade20k_panoptic_val', 'coco_2017_test-dev', 'sam_val', 'sam_minival']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True}
    else:
        if name not in ["vlp_val", "vlp_captioning_val", "vlp_val2017", "vlp_captioning_val2017", "imagenet_val", "refcocog_val_google", "phrasecut_val", "phrasecut_test", "refcocop_val_unc", "refcoco_val_unc", "refcocog_val_umd"]:
            assert False, "dataset switcher is not defined"
    for key, value in mappings.items():
        if key == 'SEMANTIC_ON':
            model.model.semantic_on = value
        if key == 'INSTANCE_ON':
            model.model.instance_on = value
        if key == 'PANOPTIC_ON':
            model.model.panoptic_on = value

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, decay=0):
        self.val = val
        if decay:
            alpha = math.exp(-n / decay)  # exponential decay over 100 updates
            self.sum = alpha * self.sum + (1 - alpha) * val * n
            self.count = alpha * self.count + (1 - alpha) * n
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

class RollingAvg:

    def __init__(self, length):
        self.length = length
        self.metrics = defaultdict(lambda: deque(maxlen=self.length))

    def add(self, name, metric):
        self.metrics[name].append(metric)

    def get(self, name):
        return torch.tensor(list(self.metrics[name])).mean()

    def logall(self, log_func):
        for k in self.metrics.keys():
            log_func(k, self.get(k))


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

midas_norm = T.Normalize([0.5] * 3, [0.5] * 3)
midas_unnorm = UnNormalize([0.5] * 3, [0.5] * 3)


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def show_heatmap(ax,
                 image,
                 heatmap,
                 cmap="bwr",
                 color=False,
                 center=False,
                 show_negative=False,
                 cax=None,
                 vmax=None):
    frame = []

    if color:
        frame.append(ax.imshow(image))
    else:
        bw = np.dot(np.array(image)[..., :3] / 255, [0.2989, 0.5870, 0.1140])
        bw = np.ones_like(image) * np.expand_dims(bw, -1)
        frame.append(ax.imshow(bw))

    if center:
        heatmap -= heatmap.mean()

    if not show_negative:
        heatmap = heatmap.clamp_min(0)

    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), (image.shape[0], image.shape[1])) \
        .squeeze(0).squeeze(0)

    if vmax is None:
        vmax = np.abs(heatmap).max()

    hm = ax.imshow(heatmap, alpha=.5, cmap=cmap, vmax=vmax, vmin=-vmax)
    if cax is not None:
        plt.colorbar(hm, cax=cax, orientation='vertical')

    frame.extend([hm])
    return frame


def implicit_feats(original_image, input_size, color_feats):
    n_freqs = 20
    grid = torch.linspace(-1, 1, input_size, device=original_image.device)
    feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid, grid])]).unsqueeze(0)

    if color_feats:
        feat_list = [feats, original_image]
        dim_multiplier = 5
    else:
        feat_list = [feats]
        dim_multiplier = 2

    feats = torch.cat(feat_list, dim=1)
    freqs = torch.exp(torch.linspace(-2, 10, n_freqs, device=original_image.device)) \
        .reshape(n_freqs, 1, 1, 1)
    feats = (feats * freqs).reshape(1, n_freqs * dim_multiplier, input_size, input_size)

    if color_feats:
        all_feats = [torch.sin(feats), torch.cos(feats), original_image]
    else:
        all_feats = [torch.sin(feats), torch.cos(feats)]
    return torch.cat(all_feats, dim=1)


def load_hr_emb(original_image, model_path, color_feats=True):
    model = torch.load(model_path, map_location="cpu")
    hr_model = model["model"].cuda().eval()
    unprojector = model["unprojector"].cuda().eval()

    with torch.no_grad():
        h, w = original_image.shape[2:]
        assert h == w
        feats = implicit_feats(original_image, h, color_feats).cuda()
        hr_feats = hr_model(feats)
        hr_feats = unprojector(hr_feats.detach().cpu())

        return hr_feats


def generate_subset(n, batch):
    np.random.seed(0)
    return np.random.permutation(n)[:batch]


class TorchPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, VT = torch.linalg.svd(unbiased, full_matrices=False) # .pca_lowrank(unbiased, center=False, niter=4)
        self.components_ = VT.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_[:, :self.n_components]
        return projected

def plot_explained_variance(pca_model, ax = None) -> None:
    if isinstance(pca_model, TorchPCA):
        # For TorchPCA
        singular_values = pca_model.singular_values_
        total_variance = torch.sum(singular_values**2)
        explained_variance_ratio = (singular_values**2) / total_variance
        cumulative_variance_ratio = torch.cumsum(singular_values**2, dim=0) / torch.sum(singular_values**2)
    else:
        # For sklearn PCA
        explained_variance_ratio = pca_model.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    if ax is None:
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    else:
        ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
        ax.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Cumulative explained variance')
        ax.set_ylabel('Explained variance ratio')
        ax.set_xlabel('Principal components')
        ax.legend(loc='best')
        ax.set_title('Explained variance ratio of the principal components')
        ax.grid()
        
    # Printing the value of explained variance explained by the number of PC specified.
    print(f"Explained variance by {pca_model.n_components} PCs: {cumulative_variance_ratio[pca_model.n_components - 1]:.2f}")

def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca

class PCAUnprojector(nn.Module):

    def __init__(self, feats, dim, device, use_torch_pca=False, **kwargs):
        super().__init__()
        self.dim = dim

        if feats is not None:
            self.original_dim = feats.shape[1]
        else:
            self.original_dim = kwargs["original_dim"]

        if self.dim != self.original_dim:
            if feats is not None:
                sklearn_pca = pca([feats], dim=dim, use_torch_pca=use_torch_pca)[1]

                # Register tensors as buffers
                self.register_buffer('components_',
                                     torch.tensor(sklearn_pca.components_, device=device, dtype=feats.dtype))
                self.register_buffer('singular_values_',
                                     torch.tensor(sklearn_pca.singular_values_, device=device, dtype=feats.dtype))
                self.register_buffer('mean_', torch.tensor(sklearn_pca.mean_, device=device, dtype=feats.dtype))
            else:
                self.register_buffer('components_', kwargs["components_"].t())
                self.register_buffer('singular_values_', kwargs["singular_values_"])
                self.register_buffer('mean_', kwargs["mean_"])

        else:
            print("PCAUnprojector will not transform data")

    def forward(self, red_feats):
        if self.dim == self.original_dim:
            return red_feats
        else:
            b, c, h, w = red_feats.shape
            red_feats_reshaped = red_feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            unprojected = (red_feats_reshaped @ self.components_) + self.mean_.unsqueeze(0)
            return unprojected.reshape(b, h, w, self.original_dim).permute(0, 3, 1, 2)

    def project(self, feats):
        if self.dim == self.original_dim:
            return feats
        else:
            b, c, h, w = feats.shape
            feats_reshaped = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            t0 = feats_reshaped - self.mean_.unsqueeze(0).to(feats.device)
            projected = t0 @ self.components_.t().to(feats.device)
            return projected.reshape(b, h, w, self.dim).permute(0, 3, 1, 2)


def prep_image(t, subtract_min=True):
    if subtract_min:
        t -= t.min()
    t /= t.max()
    t = (t * 255).clamp(0, 255).to(torch.uint8)

    if len(t.shape) == 2:
        t = t.unsqueeze(0)

    return t

@torch.no_grad()
def plot_feats(image, lr, hr, img_name = "feats", save_fig = False):
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    seed_everything(0)
    [lr_feats_pca, hr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0)])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Original Features")
    ax[2].imshow(hr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[2].set_title("Upsampled Features")
    remove_axes(ax)
    
    if save_fig:
        plt.savefig(f"{img_name}.png")
    else:
        plt.show()
    
    plt.close(fig)