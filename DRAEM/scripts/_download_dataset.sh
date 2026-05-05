cd ../datasets

## MAD-man (GLASS) tar파일 로컬에서 직접 다운
# tar -xf "MAD-man.tar.gz"
# rm "MAD-man.tar.gz"

#mkdir visa
# cd visa
# wget https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
# tar -xf VisA_20220922.tar
# rm VisA_20220922.tar

# mkdir btad
# cd btad
# wget http://avires.dimi.uniud.it/papers/btad/btad.zip
# unzip btad.zip
# rm btad.zip


# mkdir -p "mvtecloco"
# cd mvtecloco
# wget -O "mvtec_loco_anomaly_detection.tar.xz" "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz" #다운로드
# tar -xf "mvtec_loco_anomaly_detection.tar.xz" #압축 해제
# rm "mvtec_loco_anomaly_detection.tar.xz" #압축 파일 삭제
# ##권한 777 설정해야 보임 ..



#다운로드 실패 -> zip 파일 로컬에서 직접 다운로드#
### https://drive.google.com/file/d/1Rs6XRb6v3WdSidiFsMHMK9tALqaSMY3u/view?usp=sharing

#mkdir mvtec_sdas
# cd mvtec_sdas
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Rs6XRb6v3WdSidiFsMHMK9tALqaSMY3u' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Rs6XRb6v3WdSidiFsMHMK9tALqaSMY3u" -O sia_mvtec_anomaly_images.zip && rm -rf /tmp/cookies.txt

# unzip sia_mvtec_anomaly_images.zip
