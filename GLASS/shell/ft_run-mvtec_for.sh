datapath=../datasets/mvtec
augpath=../datasets/dtd/images
classes=('carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')
# classes=('carpet' 'grid' 'leather' 'tile' 'wood')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

for seed in $(seq 0 10); do
  python ft_main_scheX.py \
      --results_path ft_results/glass_mvtec_lr000001_occlr001_nonGaus_seed${seed} \
      --gpu 1 \
      --seed ${seed} \
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
      --imagesize 288 "${flags[@]}" mvtec $datapath $augpath
done