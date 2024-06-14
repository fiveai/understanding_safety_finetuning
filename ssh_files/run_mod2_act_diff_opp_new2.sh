#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_transform_all_weights_covariance_analysis.py --plot_path 'supervised_safety_finetune_1em5' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p8_safe_branch_prob_0p5_clip_0p5_nodpo_spec_100k_0p5/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_transform_all_weights_covariance_analysis.py --plot_path 'supervised_safety_finetune_1em5' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p8_safe_branch_prob_0p5_clip_0p5_nodpo_spec_100k_0p5/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=1 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_transform_all_weights_covariance_analysis.py --plot_path 'unlearn_1em5' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_transform_all_weights_covariance_analysis.py --plot_path 'unlearn_1em5' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=1 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_transform_all_weights_covariance_analysis.py --plot_path 'pretrained_model' --data_test 'std_safe' --model_load_path 'pretrained_models_new/model_mini_spec_100k_pcfg_10C_35L_30alph_pcfg_0p5_0p1_comp1_0p2_0p3_comp2_0p2_0p4_new_new/model_100000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'