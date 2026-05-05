datapath=../datasets/visa
augpath=../datasets/dtd/images
classes=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd .
python ft_main_scheX.py \
    --results_path ft_results/glass_visa_lr000001_occlr001_seed0 \
    --gpu 3 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --lr 0.000001 \
    --occ_lr 0.001 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 999 \
    --meta_steps 400 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 2 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 "${flags[@]}" visa $datapath $augpath