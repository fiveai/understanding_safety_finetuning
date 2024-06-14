#!/bin/bash
# #CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10 --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 0.1  --adv_attack_iters 10  --threat_count_adv 2  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac0p1_count2_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac0p1_count2_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac0p1_count2_attakall_morecomplex.txt


# #CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10  --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 0.5  --adv_attack_iters 10  --threat_count_adv 2  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac0p5_count2_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac0p5_count2_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac0p5_count2_attakall_morecomplex.txt
# #CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10  --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 1.0  --adv_attack_iters 10  --threat_count_adv 2  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac1p0_count2_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac1p0_count2_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac1p0_count2_attakall_morecomplex.txt
# #CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10  --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 0.25  --adv_attack_iters 10  --threat_count_adv 2  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac0p25_count2_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac0p25_count2_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac0p25_count2_attakall_morecomplex.txt
# #CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10  --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 0.5  --adv_attack_iters 10  --threat_count_adv 1  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac0p5_count1_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac0p5_count1_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac0p5_count1_attakall_morecomplex.txt
# #CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10  --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 0.5  --adv_attack_iters 10  --threat_count_adv 4  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac0p5_count4_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac0p5_count4_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac0p5_count4_attakall_morecomplex.txt
# #CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10  --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 0.5  --adv_attack_iters 10  --threat_count_adv 5  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac0p5_count5_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac0p5_count5_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac0p5_count5_attakall_morecomplex.txt
# CUDA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_toy_pcfg_simple.py --sample_pcfg_number 1 --train_data_path './saved_data_toy/basic_data_train_repeat_120_pcfg1_simple.pkl' --start_random 0  --test_data_path './saved_data_toy/basic_data_test_repeat_120_pcfg1_simple.pkl'
# CUDA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_toy_pcfg_simple.py --sample_pcfg_number 2 --train_data_path './saved_data_toy/basic_data_train_repeat_120_pcfg2_simple.pkl' --start_random 150  --test_data_path './saved_data_toy/basic_data_test_repeat_120_pcfg2_simple.pkl'
# CUDA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_toy_pcfg_simple.py --sample_pcfg_number 3 --train_data_path './saved_data_toy/basic_data_train_repeat_120_pcfg3_simple.pkl' --start_random 300  --test_data_path './saved_data_toy/basic_data_test_repeat_120_pcfg3_simple.pkl'
# CUDA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_toy_pcfg_simple.py --sample_pcfg_number 4 --train_data_path './saved_data_toy/basic_data_train_repeat_120_pcfg4_simple.pkl' --start_random 450  --test_data_path './saved_data_toy/basic_data_test_repeat_120_pcfg4_simple.pkl'


# CUDA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy_pcfg_simple.py --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --save_path 'model_10k_pcfg_25' --wandb-run '10k_pcfg_25' --path_load_train_data1 './saved_data_toy/basic_data_train_repeat_120_pcfg1_simple.pkl' --path_load_train_data2 './saved_data_toy/basic_data_train_repeat_120_pcfg2_simple.pkl' --path_load_train_data3 './saved_data_toy/basic_data_train_repeat_120_pcfg3_simple.pkl' --path_load_train_data4 './saved_data_toy/basic_data_train_repeat_120_pcfg4_simple.pkl'   --path_load_val_data1 './saved_data_toy/basic_data_train_repeat_120_pcfg1_simple.pkl' --path_load_train_data2 './saved_data_toy/basic_data_train_repeat_120_pcfg2_simple.pkl' --path_load_train_data3 './saved_data_toy/basic_data_train_repeat_120_pcfg3_simple.pkl' --path_load_train_data4 './saved_data_toy/basic_data_train_repeat_120_pcfg4_simple.pkl'   --path_load_test_data1 './saved_data_toy/basic_data_test_repeat_120_pcfg1_simple.pkl' --path_load_test_data2 './saved_data_toy/basic_data_test_repeat_120_pcfg2_simple.pkl' --path_load_test_data3 './saved_data_toy/basic_data_test_repeat_120_pcfg3_simple.pkl' --path_load_test_data4 './saved_data_toy/basic_data_test_repeat_120_pcfg4_simple.pkl'
# CUDA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy_pcfg_simple.py  --max_iters 20000 --max_train_iters 20000 --warmup_iters 4000 --lr_decay_iters 16000 --save_path 'model_20k_pcfg_25'  --wandb-run '20k_pcfg_25' --path_load_train_data1 './saved_data_toy/basic_data_train_repeat_120_pcfg1_simple.pkl' --path_load_train_data2 './saved_data_toy/basic_data_train_repeat_120_pcfg2_simple.pkl' --path_load_train_data3 './saved_data_toy/basic_data_train_repeat_120_pcfg3_simple.pkl' --path_load_train_data4 './saved_data_toy/basic_data_train_repeat_120_pcfg4_simple.pkl'   --path_load_val_data1 './saved_data_toy/basic_data_train_repeat_120_pcfg1_simple.pkl' --path_load_train_data2 './saved_data_toy/basic_data_train_repeat_120_pcfg2_simple.pkl' --path_load_train_data3 './saved_data_toy/basic_data_train_repeat_120_pcfg3_simple.pkl' --path_load_train_data4 './saved_data_toy/basic_data_train_repeat_120_pcfg4_simple.pkl'   --path_load_test_data1 './saved_data_toy/basic_data_test_repeat_120_pcfg1_simple.pkl' --path_load_test_data2 './saved_data_toy/basic_data_test_repeat_120_pcfg2_simple.pkl' --path_load_test_data3 './saved_data_toy/basic_data_test_repeat_120_pcfg3_simple.pkl' --path_load_test_data4 './saved_data_toy/basic_data_test_repeat_120_pcfg4_simple.pkl'
# CUDA_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python pretrain_toy_pcfg_simple.py  --max_iters 50000 --max_train_iters 50000 --warmup_iters 8000 --lr_decay_iters 42000 --save_path 'model_50k_pcfg_25' --wandb-run '50k_pcfg_25' --path_load_train_data1 './saved_data_toy/basic_data_train_repeat_120_pcfg1_simple.pkl' --path_load_train_data2 './saved_data_toy/basic_data_train_repeat_120_pcfg2_simple.pkl' --path_load_train_data3 './saved_data_toy/basic_data_train_repeat_120_pcfg3_simple.pkl' --path_load_train_data4 './saved_data_toy/basic_data_train_repeat_120_pcfg4_simple.pkl'   --path_load_val_data1 './saved_data_toy/basic_data_train_repeat_120_pcfg1_simple.pkl' --path_load_train_data2 './saved_data_toy/basic_data_train_repeat_120_pcfg2_simple.pkl' --path_load_train_data3 './saved_data_toy/basic_data_train_repeat_120_pcfg3_simple.pkl' --path_load_train_data4 './saved_data_toy/basic_data_train_repeat_120_pcfg4_simple.pkl'   --path_load_test_data1 './saved_data_toy/basic_data_test_repeat_120_pcfg1_simple.pkl' --path_load_test_data2 './saved_data_toy/basic_data_test_repeat_120_pcfg2_simple.pkl' --path_load_test_data3 './saved_data_toy/basic_data_test_repeat_120_pcfg3_simple.pkl' --path_load_test_data4 './saved_data_toy/basic_data_test_repeat_120_pcfg4_simple.pkl'

# export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
# CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_pcfg_toy_7_mod.py --sample_pcfg_number 1 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod/basic_data_train_repeat_7_35length_pcfg1.pkl' --start_random 0  --test_data_path './saved_data_toy_mod/basic_data_test_repeat_7_35length_pcfg1.pkl'
# export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
# CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_pcfg_toy_7_mod.py --sample_pcfg_number 2 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod/basic_data_train_repeat_7_35length_pcfg2.pkl' --start_random 150  --test_data_path './saved_data_toy_mod/basic_data_test_repeat_7_35length_pcfg2.pkl'
# export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
# CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_pcfg_toy_7_mod.py --sample_pcfg_number 3 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod/basic_data_train_repeat_7_35length_pcfg3.pkl' --start_random 300  --test_data_path './saved_data_toy_mod/basic_data_test_repeat_7_35length_pcfg3.pkl'
# export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
# CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python make_dataset_pcfg_toy_7_mod.py --sample_pcfg_number 4 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod/basic_data_train_repeat_7_35length_pcfg4.pkl' --start_random 450  --test_data_path './saved_data_toy_mod/basic_data_test_repeat_7_35length_pcfg4.pkl'


# CUDA_VISIBLE_DEVICES=1 python model_analysis_AM.py --plot_path 'unlearn_1em4_H0' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

# CUDA_VISIBLE_DEVICES=1 python model_analysis_AM.py --plot_path 'unlearn_1em4_H0' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'mg_tokens' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'mg_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'if_text_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'if_text_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'if_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'if_comp_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em4' --data_test 'if_comp_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p1/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'mg_tokens' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'mg_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'if_text_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'if_text_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'if_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'if_comp_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=1 python model_analysis_AM_H1_act.py --plot_path 'unlearn_1em5' --data_test 'if_comp_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_unlearn_1p0_0p05/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

