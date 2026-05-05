# python ft_train_st_seg_visa.py --gpu_id 3 --num_workers 8 --steps 200 --de_st_steps 200 --eval_per_steps 200  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.01 --bs 16 --perlin_ratio 0.0002 --date destseg_visa_lr01_200s --seed 0 
# python ft_train_st_seg_visa.py --gpu_id 3 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.01 --bs 16 --perlin_ratio 0.0002 --date destseg_visa_lr01_400s --seed 0 
# python ft_train_st_seg_visa.py --gpu_id 3 --num_workers 8 --steps 200 --de_st_steps 200 --eval_per_steps 200  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_visa_lr001_200s --seed 0 
# python ft_train_st_seg_visa.py --gpu_id 3 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --date destseg_visa_lr001_400s --seed 0 


# python ensemble_params.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr04_400s
# python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr04_lrocc01_400s_scheX
# python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr04_lrocc001_400s_scheX
python ft_train_visa.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --lr_occ 0.001 --date destseg_visa_lr001_lrocc001_seed1 --seed 1 
python ft_train_visa.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --lr_occ 0.001 --date destseg_visa_lr001_lrocc001_seed2 --seed 2 
python ft_train_visa.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --lr_occ 0.001 --date destseg_visa_lr001_lrocc001_seed3 --seed 3 
python ft_train_visa.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --lr_occ 0.001 --date destseg_visa_lr001_lrocc001_seed4 --seed 4 
python ft_train_visa.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --lr_occ 0.001 --date destseg_visa_lr001_lrocc001_seed5 --seed 5 
python ft_train_visa.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --lr_occ 0.001 --date destseg_visa_lr001_lrocc001_seed10 --seed 10 
python ft_train_visa.py --gpu_id 2 --num_workers 8 --steps 400 --de_st_steps 400 --eval_per_steps 400  --lr_res 0.1 --lr_seghead 0.01 --lr_de_st 0.001 --bs 16 --perlin_ratio 0.0002 --lr_occ 0.001 --date destseg_visa_lr001_lrocc001_seed42 --seed 42
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr001_lrocc001_seed1
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr001_lrocc001_seed2
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr001_lrocc001_seed3
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr001_lrocc001_seed4
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr001_lrocc001_seed5
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr001_lrocc001_seed10
python ensemble_score.py --gpu_id 2 --num_workers 16  --base_model_name DeSTSeg_VisA_400_ --date destseg_visa_lr001_lrocc001_seed42