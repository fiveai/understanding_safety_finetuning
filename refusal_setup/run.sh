
CUDA_VISIBLE_DEVICES=3 python refuse_residual_5.py --num_eigenvalues 2 | tee -a reresidual_2.txt
CUDA_VISIBLE_DEVICES=3 python refuse_residual_5_2.py --num_eigenvalues 2 | tee -a reresidual_2_2.txt
