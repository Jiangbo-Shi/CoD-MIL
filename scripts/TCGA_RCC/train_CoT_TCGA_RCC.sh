export CUDA_VISIBLE_DEVICES=1
python main.py \
--seed 0 \
--drop_out \
--early_stopping \
--lr 1e-4 \
--k_start 0 \
--k 5 \
--label_frac 1 \
--bag_loss ce \
--task task_subtyping \
--results_dir results/TCGA_RCC \
--exp_code task_TCGA_RCC_CoD_quiltnet \
--layer_num 2 \
--head_num 8 \
--model_type CoT \
--mode transformer \
--log_data \
--data_root_dir /home5/sjb/TCGA_RCC/PLIP \
--data_folder_s 10x \
--data_folder_l 20x \
--split_dir TCGA_RCC/task_2_tumor_subtyping_100 \
--img_size 5000 \
--select_ratio 0.3 \
--text_prompt_feature_path CoT/prompt/text_prompt_feature_kidney_v2_quiltnet.pt \

