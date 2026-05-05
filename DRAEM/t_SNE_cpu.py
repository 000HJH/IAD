"""Train normal이 아니라, Test normal을 이용하는 게 맞는 듯"""

import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset,MVTecDRAEMTrainDataset
# from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import sys #$
import PIL#$
import torchvision.transforms as transforms#$
import cv2
import time
from _image_check import save_img
import pandas as pd
import glob
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# obj_names = ['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
obj_names = ['toothbrush','metal_nut','leather','bottle','carpet','tile']
# obj_names = ['leather']
data_path = '../datasets/mvtec/'


"""원본 모델"""
checkpoint_path = './pre_checkpoints/origin'
base_model_name = 'DRAEM_test_0.0001_700_bs8'

# result_folder = f"_quality_results/localize_fpr0.05_{checkpoint_path.split('/')[-1]}"
result_folder = "_quality_results/t-SNE"
os.makedirs(result_folder, exist_ok=True)


gpu_id = 0

# transform = transforms.Compose([
#     # transforms.Resize((256, 256)),  # 256x256 크기로 조정
#     transforms.ToTensor(),  # 0~1 범위로 변환 (C, H, W)
# ])


img_dim = 256

for seed in range(30):
    for per in [15,30]:
        with torch.no_grad():
            for obj_name in obj_names:
                features = []
                labels = []

                run_name = base_model_name + "_" + obj_name + '_'

                model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
                model.load_state_dict(torch.load(
                    os.path.join(checkpoint_path, run_name + ".pckl"),
                    map_location='cpu'))  # CPU로 변경
                model.to('cpu')
                model.eval()

                model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)#2이미지(6채널), 마스크1개씩 반환
                model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"),  map_location='cpu'))
                model_seg.to('cpu')
                model_seg.eval()






                """Pseudo anomaly"""

                base_path1 = f'_check_image/draem_mvtec_lr00001_01_400s_scheX/{obj_name}'
                base_path2 = f'_check_image/draem_mvtec_lr0001_occlr001_400s_scheX_0425/{obj_name}'
                for base_path in [base_path1,base_path2]:
                    
                    pseudo_anomaly_list = glob.glob(f'{base_path}/_aug_gray_batch_*')

                    image_list = [p_img for p_img in pseudo_anomaly_list if "_mask" not in p_img] #원본 이미지 파일

                    for image_path in image_list:
                        label = torch.tensor([-1])

                        # image_path = "../datasets/mvtec/cable/test/cut_inner_insulation/013.png"
                        image = cv2.imread(image_path, cv2.IMREAD_COLOR) / 255.0
                        image_shape = image.shape #원본 사이즈

                        image = cv2.resize(image, dsize=(256,256))
                        
                        image = np.array(image).reshape((256,256, 3)).astype(np.float32)
                        gray_batch = np.transpose(image, (2, 0, 1))
                        gray_batch = gray_batch[[2,1,0],:,:]
                        

                        
                        if isinstance(gray_batch, np.ndarray):
                            gray_batch = torch.tensor(gray_batch, dtype=torch.float32)

                        # gray_batch = gray_batch.unsqueeze(0)  # batch dim 추가
                        gray_batch = gray_batch.unsqueeze(0).to('cpu')     # 명시적으로 CPU로


                        # 1) Recon encoder Feature 추출
                        # b5  = model.encoder(gray_batch)  # 1,1024,16,16
                        
                        # 2) seg encoder Feature 추출
                        gray_rec = model(gray_batch)
                        
                        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1) #[1,6,256,256]

                        b1,b2,b3,b4,b5,b6 = model_seg.encoder_segment(joined_in)
                        
                        

                        b_flat = torch.nn.functional.adaptive_avg_pool2d(b1, (1, 1))  # shape: (B, C, 1, 1)
                        b_flat = b_flat.view(b_flat.size(0), -1)  # shape: (B, C)

                        features.append(b_flat.cpu().numpy())
                        labels.extend(label.cpu().numpy())
                        
                    
                    
                    
                
                """Test anomaly"""
                test_dataset = MVTecDRAEMTestDataset(data_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
                test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

                
                for i_batch, sample_batched in enumerate(test_dataloader):
                    ############
                    # if i_batch >= 20:
                    #     break
                    ############

                    
                    if sample_batched['has_anomaly'] == 0:
                        continue

                    label = torch.tensor([1]) #anomaly
                    gray_batch = sample_batched["image"]  # GPU로 보내지 않음

                    
                    if isinstance(gray_batch, np.ndarray):
                        gray_batch = torch.tensor(gray_batch, dtype=torch.float32)

                    # gray_batch = gray_batch.unsqueeze(0)  # batch dim 추가
                    gray_batch = gray_batch.to('cpu')     # 명시적으로 CPU로




                    # 1) Recon encoder Feature 추출
                    # b5  = model.encoder(gray_batch)  # 1,1024,16,16
                    
                    # 2) seg encoder Feature 추출
                    gray_rec = model(gray_batch)
                    
                    joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1) #[1,6,256,256]

                    b1,b2,b3,b4,b5,b6 = model_seg.encoder_segment(joined_in)
                    
                    

                    b_flat = torch.nn.functional.adaptive_avg_pool2d(b1, (1, 1))  # shape: (B, C, 1, 1)
                    b_flat = b_flat.view(b_flat.size(0), -1)  # shape: (B, C)

                    features.append(b_flat.cpu().numpy())
                    labels.extend(label.cpu().numpy())

                    
                    
                
        
                """Train normal"""
                dataset = MVTecDRAEMTrainDataset(data_path + obj_name + "/train/good", '../datasets/dtd/images/',resize_shape=[img_dim, img_dim])
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                
                for i_batch, sample_batched in enumerate(dataloader):
                    ############
                    if i_batch >= len(test_dataset):
                        break
                    ############
                    
                    # if sample_batched['has_anomaly'] == np.array([0], dtype=np.float32):
                    #     continue

                    label = torch.tensor([0])
                    gray_batch = sample_batched["image"]  # GPU로 보내지 않음
                    

                    if isinstance(gray_batch, np.ndarray):
                        gray_batch = torch.tensor(gray_batch, dtype=torch.float32)

                    # gray_batch = gray_batch.unsqueeze(0)  # batch dim 추가
                    gray_batch = gray_batch.to('cpu')     # 명시적으로 CPU로


                    # 1) Recon encoder Feature 추출
                    # b5  = model.encoder(gray_batch)  # 1,1024,16,16
                    
                    # 2) seg encoder Feature 추출
                    gray_rec = model(gray_batch)
                    
                    joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1) #[1,6,256,256]

                    b1,b2,b3,b4,b5,b6 = model_seg.encoder_segment(joined_in)
                    
                    

                    b_flat = torch.nn.functional.adaptive_avg_pool2d(b1, (1, 1))  # shape: (B, C, 1, 1)
                    b_flat = b_flat.view(b_flat.size(0), -1)  # shape: (B, C)

                    features.append(b_flat.cpu().numpy())
                    labels.extend(label.cpu().numpy())




            
                    

                X = np.concatenate(features, axis=0)
                y = np.array(labels).ravel() # 또는 np.squeeze(y)

                X_tsne = TSNE(n_components=2, perplexity=per, random_state=seed).fit_transform(X)
                # X_tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=).fit_transform(X)
                
                plt.figure(figsize=(8, 6))
                # plt.scatter(X_tsne[0], X_tsne[1], c='blue', label='Normal', alpha=0.6)
                plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='green', label='Normal', alpha=0.6)
                plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='red', label='Real Anomaly', alpha=0.6)
                plt.scatter(X_tsne[y==-1, 0], X_tsne[y==-1, 1], c='purple', label='Pseudo Anomaly', alpha=0.6)
                plt.legend()
                plt.title("t-SNE of Encoder Feature")
                
                plt.savefig(f"{result_folder}/{obj_name}_{per}_{seed}.png", dpi=300, bbox_inches='tight')
                plt.show()
            
            
        # save_img(np.transpose(overlay/255.0, (2, 0, 1)), out_mask_path)
        