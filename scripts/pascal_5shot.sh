#!/bin/bash

for fold in 0 1 2 3;
do
  python main_MARS.py  \
    --dataset_path /home/pertino/exp/private_datasets \
    --nltk_path /home/pertino/exp/NLTK_data \
    --mask_proposals_path /home/pertino/exp/MARS-Multimodal-Alignment-and-Ranking-System-for-Few-Shot-Segmentation/vis/pascal_test/5shot/pascal_test_pred_masks_unmerged_unfiltered/ \
    --benchmark pascal5i \
    --nworker 0 \
    --nshot 5 \
    --fold ${fold} \
    --input_size 518 \
    --models_path /home/pertino/exp/MARS-Multimodal-Alignment-and-Ranking-System-for-Few-Shot-Segmentation/models \
    --prompt_type contour \
    --zoom_percentage 50 \
    --color red \
    --alpha_blending 0.5 \
    --thickness 2 \
    --vlm4bit \
    --vta_backbone 'ViT-B/16' \
    --vta_refinement_box_threshold 0.4 \
    --last_n_attn_for_vta_refinement 8 \
    --vva_backbone dino \
    --dino_backbone vit_large \
    --num_regs 4 \
    --vva_refinement_box_threshold 0.8 \
    --last_n_attn_for_vva_refinement 24 \
    --use_vva_mix \
    --use_negative_prior \
    --static_threshold 0.55 \
    --dynamic_threshold 0.95 \
    --alpha_coverage 0.85 \
    --log_root_path "/home/pertino/exp/MARS-Multimodal-Alignment-and-Ranking-System-for-Few-Shot-Segmentation/output/mars/pascal/fold${fold}" \
    --exp_name 5shot
done
wait
