#!/bin/bash
# #CUDA_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python pretrain_toy.py --num_samples_train 1000000 --num_samples_test 1000000 --max_train_iters 5000 --max_test_iters 10 --path_load_train_data './saved_data_toy/basic_data_val.pkl' --path_load_test_data './saved_data_toy/basic_data_test.pkl' --jail_mg_para_frac 0.1  --adv_attack_iters 10  --threat_count_adv 2  --jail_mg_para_attack_type 'all' --wandb-project 'gpt-cap-pretrain' --wandb-run 'models_iters5k_steps10_frac0p1_count2_attakall_morecomplex' --save_path 'models_iters5k_steps10_frac0p1_count2_attakall_morecomplex' --save_iter 100 | tee -a models_iters5k_steps10_frac0p1_count2_attakall_morecomplex.txt


#CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp.py --plot_path 'dpo_1em4_0p0025' --data_test 'mg_tokens' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
#CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp.py --plot_path 'dpo_1em4_0p0025' --data_test 'mg_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
#CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp.py --plot_path 'dpo_1em4_0p0025' --data_test 'if_text_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
#CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp.py --plot_path 'dpo_1em4_0p0025' --data_test 'if_text_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
#CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp.py --plot_path 'dpo_1em4_0p0025' --data_test 'if_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
#CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp.py --plot_path 'dpo_1em4_0p0025' --data_test 'if_comp_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
#CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp.py --plot_path 'dpo_1em4_0p0025' --data_test 'if_comp_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'





CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em4_0p01' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em4_0p01' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em4_0p01' --data_test 'mg_tokens' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em4_0p01' --data_test 'mg_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em4_0p01' --data_test 'if_text_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em4_0p01' --data_test 'if_text_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em4_0p01' --data_test 'if_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'


CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em5_0p002' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em5_0p002' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em5_0p002' --data_test 'mg_tokens' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em5_0p002' --data_test 'mg_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em5_0p002' --data_test 'if_text_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em5_0p002' --data_test 'if_text_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample.py --plot_path 'dpo_1em5_0p002' --data_test 'if_txt' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'



CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample_safe.py --plot_path 'dpo_1em4_0p01' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample_safe.py --plot_path 'dpo_1em4_0p01' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em4_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p01/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'


CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample_safe.py --plot_path 'dpo_1em5_0p002' --data_test 'std_unsafe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'
CUDA_VISIBLE_DEVICES=2 python model_analysis_avg_act_neuron_mod2_mod3_opp2_activation_patching_unsafe_correct_abs_sample_safe.py --plot_path 'dpo_1em5_0p002' --data_test 'std_safe' --model_load_path 'safety_finetuned_models_new/model_finetune_10k_mini_lr1em5_finetuned_prob_safe_0p5_safe_branch_prob_0p5_clip_1p0_spec_100k_0p5_dpo_0p1_0p002/model_10000.pkl'  --learning_rate 0.00005 --min_lr 0.0000005 --model_type 'wrn2-cfg-mini'   --wandb-project 'attention_maps'  --max_input_length 35  --max_window_possible 159 --max_iters 10000 --max_train_iters 10000 --warmup_iters 2000 --lr_decay_iters 8000 --prob_safe 0.0 --prob_unsafe 1.0 --safe_branch_prob 0.5 --id_mg_prob 0.5 --is_dpo 0 --dpo_weight_safe 0 --dpo_weight_unsafe 0  --save_path 'attention' --wandb-run 'attention'

