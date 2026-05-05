export CUDA_VISIBLE_DEVICES=0
#!/bin/bash
# python ensemble_score.py --gpu_id 0 --num_workers 16  --base_model_name DeSTSeg_MVTec_400_ --date destseg_mvtec_lr04_lrocc001_seed1

for seed in $(seq 2 20); do
  # 학습 실행
  # python ft_train.py \
  #   --gpu_id 0 \
  #   --num_workers 8 \
  #   --steps 400 \
  #   --de_st_steps 400 \
  #   --eval_per_steps 400 \
  #   --lr_res 0.1 \
  #   --lr_seghead 0.01 \
  #   --lr_de_st 0.04 \
  #   --lr_occ 0.001 \
  #   --bs 16 \
  #   --perlin_ratio 0.0002 \
  #   --date destseg_mvtec_lr04_lrocc001_seed${seed} \
  #   --seed ${seed}

  # ensemble 실행
  python ensemble_score.py \
    --gpu_id 0 \
    --num_workers 16 \
    --base_model_name DeSTSeg_MVTec_400_ \
    --date destseg_mvtec_lr04_lrocc001_seed${seed}
done


# python ft_train_visa.py --gpu_id 3 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.04 --lr_occ 0.01 --bs 16 --perlin_ratio 0.0002 --date destseg_visa_lr04_lrocc01_400s_scheX --seed 0
# python ft_train_visa.py --gpu_id 3 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_visa_lr04_lrocc001_400s_scheX --seed 0

# python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 100 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_100s_300s_scheX --seed 0
# python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 100 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.0001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc0001_100s_300s_scheX --seed 0

# python ensemble_score.py --gpu_id 3 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr04_400s
# python ensemble_score.py --gpu_id 3 --num_workers 16  --base_model_name DeSTSeg_MVTec_400_ --date destseg_mvtec_lr04_400s_sd0
# python ensemble_score.py --gpu_id 3 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_100s_300s
# python ensemble_params.py --gpu_id 3 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_100s_300s