# python ft_train_st_seg_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 300 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.1 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_300s_100s_lr1_01_01 --seed 0
# python ft_train_st_seg_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 300 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.01 --lr_de_st 0.01 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_300s_100s_lr01_01_01 --seed 0
# python ft_train_st_seg_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 200 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.1 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_200s_200s_lr1_01_01 --seed 0
# python ft_train_st_seg_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 200 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.01 --lr_de_st 0.01 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_200s_200s_lr01_01_01 --seed 0
# python ft_train_st_seg_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 200 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.04 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_200s_200s_lr1_01_04 --seed 0

# best: python ft_train_st_seg_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 100 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_100s_300s --seed 0

python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 100 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_100s_300s_C2_nonGuas_seed0 --seed 2
python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 100 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_100s_300s_C2_nonGuas_seed1 --seed 3
python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_400s_C2_nonGuas_seed0 --seed 0
python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_00s_C2_nonGuas_seed1 --seed 1
python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_400s_C2_nonGuas_seed2 --seed 2
python ft_train_btad.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.01 --lr_seghead 0.0001 --lr_de_st 0.04 --lr_occ 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_btad_lr01_0001_04_lrocc001_00s_C2_nonGuas_seed3 --seed 3


python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_100s_300s_C2_nonGuas_seed2
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_100s_300s_C2_nonGuas_seed3
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_400s_C2_nonGuas_seed0
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_00s_C2_nonGuas_seed1
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_400s_C2_nonGuas_seed2
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_Btad_400_ --date destseg_btad_lr01_0001_04_lrocc001_00s_C2_nonGuas_seed3
