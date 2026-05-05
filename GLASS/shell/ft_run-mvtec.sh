datapath=../datasets/mvtec
augpath=../datasets/dtd/images
classes=('carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')

flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd .
python ft_main.py \
    --results_path ft_results/glass_mvtec_lr000001_occlr0001_400s_sigma4_seed2 \
    --gpu 0 \
    --seed 2 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --lr 0.000001 \
    --occ_lr 0.0001 \
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
    --imagesize 288 "${flags[@]}" mvtec $datapath $augpath