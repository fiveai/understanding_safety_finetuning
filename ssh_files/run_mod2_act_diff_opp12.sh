#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'std_unsafe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'    --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'std_safe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'    --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'mg_tokens' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'    --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'mg_txt' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'    --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'if_text_safe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'    --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4'  --plot_path2 'dpo_1em4_0p01' --data_test 'if_text_unsafe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'    --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4'  --plot_path2 'dpo_1em4_0p01' --data_test 'if_txt' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'    --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'unlearn_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'std_unsafe' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'unlearn_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'std_safe' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'unlearn_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'mg_tokens' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'unlearn_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'mg_txt' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'   --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'unlearn_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'if_text_safe' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'unlearn_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'if_text_unsafe' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'unlearn_1em4' --plot_path2 'dpo_1em4_0p01'  --data_test 'if_txt' --model_load_path 'new_models_transferred/unlearn_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/dpo_1em4/model_10000.pkl'   --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'unlearn_1em4'  --data_test 'std_unsafe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'unlearn_1em4'  --data_test 'std_safe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'unlearn_1em4'  --data_test 'mg_tokens' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'unlearn_1em4'  --data_test 'mg_txt' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4' --plot_path2 'unlearn_1em4'  --data_test 'if_text_safe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4'  --plot_path2 'unlearn_1em4' --data_test 'if_text_unsafe' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=0 python acc_lmc_analysis_different_protocols.py --plot_path 'supervised_safety_finetune_1em4'  --plot_path2 'unlearn_1em4' --data_test 'if_txt' --model_load_path 'new_models_transferred/ssft_1em4/model_10000.pkl' --model_load_path2 'new_models_transferred/unlearn_1em4/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'