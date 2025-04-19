r""" MARS testing code for few-shot segmentation with stored mask proposals."""
import argparse
import os
import random
import time

import nltk
import torch

from mars.MARS import MARS, build_MARS_fss
from mars.utils.evaluation import Evaluator
from mars.utils.logger import Logger, AverageMeter
from mars.utils.utils import fix_randseed, to_cuda
from matcher.data.dataset import FSSDataset

def test_MARS(args):
    # Logger Setup
    log_path = os.path.join(args.log_root_path, args.exp_name)
    os.makedirs(log_path, exist_ok=True)
    Logger.initialize(args, root=log_path)
    
    # Fixing the seed
    random.seed(0)
    fix_randseed(0)
    
    # Setting up the dataset
    FSSDataset.initialize(
        img_size=args.input_size, 
        datapath=args.dataset_path,
        use_original_imgsize=False
    )
    dataloader_test = FSSDataset.build_dataloader(
        benchmark=args.benchmark, 
        bsz=1, # bsz set to 1 for testing 
        nworker=args.nworker, 
        fold=args.fold, 
        split='test', 
        shot=args.nshot
    )
    
    # Setting up the evaluator
    Evaluator.initialize()
    average_meter = AverageMeter(dataloader_test.dataset, device=args.device)
    
    # Building MARS
    mars: MARS = build_MARS_fss(args=args)
    
    test_start_time = time.time()
    time_last_batch = test_start_time
    time_elapsed_per_batch = []
    
    # Testing loop
    for idx, batch in enumerate(dataloader_test):
        batch = to_cuda(batch)
        query_img, query_mask, support_imgs, support_masks, class_idx, query_img_name = \
            batch['query_img'], batch['query_mask'], \
            batch['support_imgs'], batch['support_masks'], \
            batch['class_id'].item(), batch['query_name']
        
        # Loading set of pre-generated mask proposals
        mask_proposals = torch.load(os.path.join(args.mask_proposals_path, f'{args.fold}_{idx}.pt'))
        
        predicted_mask = mars.predict(
            support_images=support_imgs,
            support_masks=support_masks,
            query_image=query_img,
            mask_proposals=mask_proposals
        )
        
        # Updating the metrics
        area_inter, area_union = Evaluator.classify_prediction(predicted_mask.clone(), batch)
        average_meter.update(area_inter, area_union, class_idx, loss=None)
        average_meter.write_process(idx, len(dataloader_test), epoch=-1, write_batch_idx=1)
        
        # Calculate time elapsed per batch
        time_elapsed_batch = time.time() - time_last_batch
        time_elapsed_per_batch.append(time_elapsed_batch)
        time_last_batch = time.time()
        
    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou, _ = average_meter.compute_iou()
    
    test_end_time = time.time()
    
    Logger.info(f'mIoU: {miou:.2f} - FB-IoU: {fb_iou:.2f}')
    Logger.info(f'Average time per image: {sum(time_elapsed_per_batch) / len(time_elapsed_per_batch):.2f}')
    Logger.info(f'Test time: {(test_end_time - test_start_time):.2f}')

if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(
        description='MARS Pytorch Implementation for Few-shot Segmentation'
    )
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='/home/pertino/exp/private_datasets')
    parser.add_argument('--annotations_datapath', type=str, default='/home/pertino/exp/private_datasets/COCO2014/annotations')
    parser.add_argument('--nltk_path', type=str, default='/home/pertino/exp/NLTK_data')
    parser.add_argument('--mask_proposals_path', type=str, default='/home/pertino/exp/MARS-Multimodal-Alignment-and-Ranking-System-for-Few-Shot-Segmentation/vis/coco_test/coco_test_pred_masks_unmerged_unfiltered/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['fss', 'coco', 'pascal5i', 'lvis'])
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1, choices=[1, 5])
    # parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=518)
    
    # General models parameters
    parser.add_argument('--models_path', type=str, default='/home/pertino/exp/MARS-Multimodal-Alignment-and-Ranking-System-for-Few-Shot-Segmentation/models')
    
    # Text Information Retrieval Component parameters
    parser.add_argument('--prompt_type', type=str, default='contour', choices=['mask', 'bb', 'contour', 'ellipse'])
    parser.add_argument('--zoom_percentage', type=int, default=50)
    parser.add_argument('--color', type=str, default="red", choices=["red", "green", "blue"])
    parser.add_argument('--ensamble_prompts', action='store_true', help='Use multiple prompts for the same image')
    parser.add_argument('--ensamble_prompts_list', type=str, nargs="+", default=["bb", "contour", "ellipse"])
    parser.add_argument('--ensamble_zoom', action='store_true', help='Use multiple zoom percentages for the same image')
    parser.add_argument('--ensamble_zoom_list', type=int, nargs="+", default=[0, 30, 50])
    parser.add_argument('--ensamble_colors', action='store_true', help='Use multiple colors for the same image')
    parser.add_argument('--ensamble_colors_list', type=str, nargs="+", default=["red", "green", "blue"])
    parser.add_argument('--alpha_blending', type=float, default=0.5)
    parser.add_argument('--thickness', type=int, default=2)
    parser.add_argument('--vlm4bit', action='store_true', help='Load the 4-bit quantized version of the VLM')
    parser.add_argument('--vlm8bit', action='store_true', help='Load the 8-bit quantized version of the VLM')
    
    # Visual-Textual Alignment Component parameters
    parser.add_argument('--vta_backbone', type=str, default='ViT-B/16', choices=['ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--vta_refinement_box_threshold', type=float, default=0.4)
    parser.add_argument('--last_n_attn_for_vta_refinement', type=int, default=8)
    
    # Visual-Visual Alignment Component parameters
    parser.add_argument('--vva_backbone', type=str, default='dino', choices=['dino', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--dino_backbone', type=str, default='vit_large', choices=['vit_large'])
    parser.add_argument('--num_regs', type=int, default=4)
    parser.add_argument('--vva_refinement_box_threshold', type=float, default=0.8)
    parser.add_argument('--last_n_attn_for_vva_refinement', type=int, default=24)
    parser.add_argument('--use_vva_mix', action='store_true', help='Use both max and mean aggregation functions applied to the similarity matrix. If False, only max aggregation is used.')
    parser.add_argument('--use_negative_prior', action='store_true', help='Use both foreground and background information when computing the VVA. If False, only FG info is used.')
    
    # Filtering and Merging Component parameters
    parser.add_argument('--static_threshold', type=float, default=0.55)
    parser.add_argument('--dynamic_threshold', type=float, default=0.95)
    parser.add_argument('--alpha_coverage', type=float, default=0.85)
    
    # Logging parameters
    parser.add_argument('--log_root_path', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    nltk.data.path.append(args.nltk_path)
    nltk.download('wordnet', download_dir=args.nltk_path)
    nltk.download('stopwords', download_dir=args.nltk_path)
    nltk.download('punkt_tab', download_dir=args.nltk_path)
    
    test_MARS(args)