
# bash shell/ft_run-mvtec1.sh #glass_mvtec_lr000001_occlr0001_400s_scheX
# bash shell/ft_run-mvtec2.sh #glass_mvtec_lr000001_occlr00001_400s_scheX
# bash shell/ft_run-visa_for.sh 

# for seed in $(seq 4 20); do
#   python ensemble_score.py \
#     --data_path ../datasets/visa \
#     --date glass_visa_lr000001_occlr001_seed${seed}
# done



# bash shell/ft_run-btad_for_C2.sh 

# for seed in $(seq 0 10); do
#   python ensemble_score.py \
#     --data_path ../datasets/btad \
#     --date glass_btad_lr00001_C2_nonGaus_${seed}
# done



bash shell/ft_run-mvtec_for.sh 

for seed in $(seq 0 10); do
  python ensemble_score.py \
    --data_path ../datasets/mvtec \
    --date glass_mvtec_lr000001_occlr001_nonGaus_seed${seed}
done


# python ensemble_score.py --data_path ../datasets/visa --date glass_visa_lr000001_occlr001_seed0

# bash shell/ft_run-btad1.sh #glass_btad_lr000001_occlr001_400s_scheX
# bash shell/ft_run-btad2.sh #glass_btad_lr000001_occlr0001_400s_scheX