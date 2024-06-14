#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 2500 --plot_path 'unlearn_1em5' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/unlearn_1em5/model_2500.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 2500 --plot_path 'unlearn_1em4' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/unlearn_1em4/model_2500.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 2500 --plot_path 'supervised_safety_finetune_1em5' --data_test 'std_unsafe' --model_load_path new_models_transferred/ssft_1em5/model_2500.pkl  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 2500 --plot_path 'supervised_safety_finetune_1em4' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/ssft_1em4/model_2500.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3  python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py --num 2500  --plot_path 'dpo_1em4_0p01' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/dpo_1em4/model_2500.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3  python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 2500 --plot_path 'dpo_1em5_0p002' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/dpo_1em5/model_2500.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 5000 --plot_path 'unlearn_1em5' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/unlearn_1em5/model_5000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 5000 --plot_path 'unlearn_1em4' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/unlearn_1em4/model_5000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 5000 --plot_path 'supervised_safety_finetune_1em5' --data_test 'std_unsafe' --model_load_path new_models_transferred/ssft_1em5/model_5000.pkl  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 5000 --plot_path 'supervised_safety_finetune_1em4' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/ssft_1em4/model_5000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3  python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py --num 5000  --plot_path 'dpo_1em4_0p01' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/dpo_1em4/model_5000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3  python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 5000 --plot_path 'dpo_1em5_0p002' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/dpo_1em5/model_5000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 10000 --plot_path 'unlearn_1em5' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/unlearn_1em5/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 10000 --plot_path 'unlearn_1em4' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 10000 --plot_path 'supervised_safety_finetune_1em5' --data_test 'std_unsafe' --model_load_path new_models_transferred/ssft_1em5/model_10000.pkl  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3 python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 10000 --plot_path 'supervised_safety_finetune_1em4' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3  python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py --num 10000  --plot_path 'dpo_1em4_0p01' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/dpo_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=3  python model_analysis_svd_compare_models_weights_costheta_analysis_act_corr_cos_understand_norm_train_iters.py  --num 10000 --plot_path 'dpo_1em5_0p002' --data_test 'std_unsafe' --model_load_path 'new_models_transferred/dpo_1em5/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
