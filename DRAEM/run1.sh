




## hyperparameter tune

# python ft_train.py --gpu_id 1  --pretrained_path ./pre_checkpoints/origin --base_model_name  "DRAEM_test_0.0001_700_bs8" --foreground --perlin_ratio 0.00002 --occ_ratio 1.0 --lr_rec 0.0001 --lr_occ 0.001 --total_step 400 --date draem_mvtec_lr0001_occlr001_400s_components1_3_ --seed 1 --minimum_components 1 --maximum_components 3
# # python ft_train.py --gpu_id 1  --pretrained_path ./pre_checkpoints/origin --base_model_name  "DRAEM_test_0.0001_700_bs8" --foreground --perlin_ratio 0.00002 --occ_ratio 1.0 --lr_rec 0.0001 --lr_occ 0.001 --total_step 400 --date draem_mvtec_lr0001_occlr001_400s_components1_5_ --seed 1 --minimum_components 1 --maximum_components 5
# python ft_train.py --gpu_id 1  --pretrained_path ./pre_checkpoints/origin --base_model_name  "DRAEM_test_0.0001_700_bs8" --foreground --perlin_ratio 0.00002 --occ_ratio 1.0 --lr_rec 0.0001 --lr_occ 0.001 --total_step 400 --date draem_mvtec_lr0001_occlr001_400s_components1_10_ --seed 1 --minimum_components 1 --maximum_components 10
# python ft_train.py --gpu_id 1  --pretrained_path ./pre_checkpoints/origin --base_model_name  "DRAEM_test_0.0001_700_bs8" --foreground --perlin_ratio 0.00002 --occ_ratio 1.0 --lr_rec 0.0001 --lr_occ 0.001 --total_step 400 --date draem_mvtec_lr0001_occlr001_400s_components1_ --seed 1 --minimum_components 1 --maximum_components 1
# python ft_train.py --gpu_id 1  --pretrained_path ./pre_checkpoints/origin --base_model_name  "DRAEM_test_0.0001_700_bs8" --foreground --perlin_ratio 0.00002 --occ_ratio 1.0 --lr_rec 0.0001 --lr_occ 0.001 --total_step 400 --date draem_mvtec_lr0001_occlr001_400s_components3_ --seed 1 --minimum_components 3 --maximum_components 3
# python ft_train.py --gpu_id 1  --pretrained_path ./pre_checkpoints/origin --base_model_name  "DRAEM_test_0.0001_700_bs8" --foreground --perlin_ratio 0.00002 --occ_ratio 1.0 --lr_rec 0.0001 --lr_occ 0.001 --total_step 400 --date draem_mvtec_lr0001_occlr001_400s_components5_ --seed 1 --minimum_components 5 --maximum_components 5


## Ensemble
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed1 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.1
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed1 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.3
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed1 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.7
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed1 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.9

# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed10 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.1
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed10 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.3
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed10 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.7
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed10 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.9

# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed42 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.1
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed42 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.3
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed42 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.7
# python ensemble_score.py --gpu_id 3 --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/draem_mvtec_lr0001_occlr001_400s_seed42 --base_model_name "DRAEM_test_0.0001" --origin_weight 0.9
