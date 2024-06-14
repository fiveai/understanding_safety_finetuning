#!/bin/bash


export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg1_new.py --unsafe_id_mg True --from_unsafe_branch True --is_train 1 --sample_pcfg_number 1 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_train_repeat_10_35length_pcfg1_mod_new_toy.pkl' --start_random 0  --test_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_test_repeat_10_35length_pcfg1_mod_new_toy.pkl'
export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg2_new.py  --unsafe_id_mg True --from_unsafe_branch True --is_train 1 --sample_pcfg_number 2 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_train_repeat_10_35length_pcfg2_mod_new_toy.pkl' --start_random 150  --test_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_test_repeat_10_35length_pcfg2_mod_new_toy.pkl'
export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg3_new.py --unsafe_id_mg True  --from_unsafe_branch True --is_train 1 --sample_pcfg_number 3 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_train_repeat_10_35length_pcfg3_mod_new_toy.pkl' --start_random 300  --test_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_test_repeat_10_35length_pcfg3_mod_new_toy.pkl'
export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg4_new.py --unsafe_id_mg True --from_unsafe_branch True  --is_train 1 --sample_pcfg_number 4 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_train_repeat_10_35length_pcfg4_mod_new_toy.pkl' --start_random 450  --test_data_path './saved_data_toy_mod_finetune/unsafe_id_mg_data_test_repeat_10_35length_pcfg4_mod_new_toy.pkl'


export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg1_new.py --safe True --is_train 1 --sample_pcfg_number 1 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/safe_data_train_repeat_10_35length_pcfg1_mod_new_toy.pkl' --start_random 0  --test_data_path './saved_data_toy_mod_finetune/safe_data_test_repeat_10_35length_pcfg1_mod_new_toy.pkl'
export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg2_new.py --safe True --is_train 1 --sample_pcfg_number 2 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/safe_data_train_repeat_10_35length_pcfg2_mod_new_toy.pkl' --start_random 150  --test_data_path './saved_data_toy_mod_finetune/safe_data_test_repeat_10_35length_pcfg2_mod_new_toy.pkl'
export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg3_new.py --safe True --is_train 1 --sample_pcfg_number 3 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/safe_data_train_repeat_10_35length_pcfg3_mod_new_toy.pkl' --start_random 300  --test_data_path './saved_data_toy_mod_finetune/safe_data_test_repeat_10_35length_pcfg3_mod_new_toy.pkl'
export WANDB_API_KEY=ea5312074bc32593713da794cedc9ca4ac04f048
CUDA_VISIBLE_DEVICES=0 python  make_dataset_pcfg_safetyfinetune_toy_10_pcfg4_new.py --safe True --is_train 1 --sample_pcfg_number 4 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data_toy_mod_finetune/safe_data_train_repeat_10_35length_pcfg4_mod_new_toy.pkl' --start_random 450  --test_data_path './saved_data_toy_mod_finetune/safe_data_test_repeat_10_35length_pcfg4_mod_new_toy.pkl'


