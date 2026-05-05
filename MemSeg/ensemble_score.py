import logging
import os
import torch
import torch.nn as nn
import argparse

from omegaconf import OmegaConf
from timm import create_model
# from data_ft import create_dataset_ft, create_dataloader_ft #ft
from data import create_dataset, create_dataset_visa, create_dataset_btad,create_dataset_loco, create_dataloader # test 원본 

from models import MemSeg, MemoryBank
from focal_loss import FocalLoss
from ft_train import training
from log import setup_default_logging
from utils import torch_seed
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score
from _metrics import compute_pro
import pandas as pd
import sys
args = OmegaConf.from_cli()



if 'visa' in args.configs:
    dataset = 'visa'
    ft_dataset_dir = 'saved_model_visa_ft'
    origin_model_datafolder = 'saved_model_visa'
    testset_def = create_dataset_visa
    obj_list = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
elif 'btad' in args.configs:
    dataset = 'btad'
    ft_dataset_dir = 'saved_model_btad_ft'
    origin_model_datafolder = 'saved_model_btad'
    testset_def = create_dataset_btad
    obj_list = ['01','02','03']
elif 'loco' in args.configs:
    dataset = 'loco'
    ft_dataset_dir = 'saved_model_loco_ft'
    origin_model_datafolder = 'saved_model_loco'
    testset_def = create_dataset_loco
    obj_list = ['splicing_connectors','juice_bottle','breakfast_box','screw_bag','pushpins']
else:
    dataset = 'mvtec'
    ft_dataset_dir = 'saved_model_ft'
    origin_model_datafolder = 'saved_model'
    testset_def = create_dataset
    obj_list=['bottle','cable','capsule','carpet','grid','hazelnut','leather', 'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
    # obj_list=['carpet','grid','leather','tile','wood']

# load default config
cfg = OmegaConf.load(args.configs)
del args['configs']

cfg.gpu_id = args.gpu_id  #$
del args['gpu_id']  


device = f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu'

# merge config with new keys
cfg = OmegaConf.merge(cfg, args)
    
# cfg.RESULT.savedir =  f'./{ft_dataset_dir}/{cfg.date}' #$
cfg.RESULT.savedir = os.path.join(ft_dataset_dir,cfg.date)


'''최종 결과 정리 csv '''#$
# os.makedirs(cfg.RESULT.savedir, exist_ok=True)
file_path = os.path.join(cfg.RESULT.savedir, f'_memseg_{dataset}_ensemble_score_{args.origin_weight}_{round(1-args.origin_weight,1)}.csv')
score_df = pd.DataFrame({'Objects':[], 'AUC Image':[],'AUC Pixel':[], 'AP Image':[], 'AP Pixel':[], 'PRO':[]})

for category in obj_list:
# def evaluate(model, dataloader, device: str = 'cpu'):

    # testset = create_dataset_visa(
    testset = testset_def(
        datadir   = cfg.DATASET.datadir,
        target    = category, 
        is_train  = False,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )
    
    feature_extractor = create_model(
        cfg.MODEL.feature_extractor_name, #resnet18
        pretrained    = True, 
        features_only = True
    ).to(device)
    
    # ## freeze weight of layer1,2,3
    # for l in ['layer1','layer2','layer3']:
    #     for p in feature_extractor[l].parameters():
    #         p.requires_grad = False
            
            
    '''memory bank는 원본?'''
    memory_bank = torch.load(f"{origin_model_datafolder}/MemSeg-{category}/memory_bank.pt", map_location="cpu") #불러옴
    ###
    memory_bank.device = device  # 저장된 모델의 device 변경
    for key in memory_bank.memory_information.keys():
        memory_bank.memory_information[key] = memory_bank.memory_information[key].to(device)
    ###
    
    '''파라미터 앙상블'''
    model_path_list = [
        f"{origin_model_datafolder}/MemSeg-{category}/latest_model.pt", #원본
        f"{cfg.RESULT.savedir}/MemSeg-{category}/latest_model_ft.pt"]
    
    ensemble_pixel_scores = []   # 픽셀 단위 스코어 (N, H, W)
    ensemble_image_scores = []   # 이미지 단위 스코어 (N,)
    
    
    for model_path in model_path_list:

        model = MemSeg(memory_bank = memory_bank, feature_extractor = feature_extractor).to(device)
        
        model.load_state_dict(torch.load(model_path)) #불러옴
        model.eval()


        # targets and outputs
        image_targets = []
        image_masks = []
        anomaly_score = []
        anomaly_map = []

        with torch.no_grad():
            for idx, (inputs, masks, targets) in enumerate(testloader):
                inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
                

                # predict
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)
                anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

                # stack targets and outputs
                image_targets.extend(targets.cpu().tolist())
                image_masks.extend(masks.cpu().numpy())
                
                anomaly_score.extend(anomaly_score_i.cpu().tolist())
                anomaly_map.extend(outputs[:,1,:].cpu().numpy())
                
        anomaly_score = np.array(anomaly_score)
        anomaly_map = np.array(anomaly_map)
        
        '''정규화 (이미지 수준)'''
        # anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)
        ensemble_image_scores.append(anomaly_score)
        
        '''정규화 (픽셀 수준)'''
        anomaly_map = np.array(anomaly_map)
        # anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        ensemble_pixel_scores.append(anomaly_map)

            
    ensemble_pixel_scores = np.array(ensemble_pixel_scores)
    ensemble_image_scores = np.array(ensemble_image_scores)
    
    '''(1) 모델 평균 score'''
    # anomaly_score = np.mean(ensemble_image_scores, axis=0)  # 이미지 단위 평균
    # anomaly_map = np.mean(ensemble_pixel_scores, axis=0)        # 픽셀 단위 평균


    '''(2) Weighted sum'''
    anomaly_score = ensemble_image_scores[0]*args.origin_weight + ensemble_image_scores[1]*(1-args.origin_weight)
    anomaly_map = ensemble_pixel_scores[0]*args.origin_weight + ensemble_pixel_scores[1]*(1-args.origin_weight)
    
    

    image_masks = np.array(image_masks) # gt pixel
    
    
    auroc_image = roc_auc_score(image_targets, anomaly_score)
    auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    
    '''추가 메트릭'''
    ap_image = average_precision_score(image_targets, anomaly_score)
    ap_pixel = average_precision_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    
    
    # print('anomaly_map:',anomaly_map.shape, np.unique(anomaly_map[0]))
    # print('image_masks:',image_masks.shape, np.unique(image_masks[0]))
     
    aupro = compute_pro(
        anomaly_maps      = anomaly_map,
        ground_truth_maps = image_masks
    )


    new_row = {'Objects': category, 'AUC Image': round(auroc_image*100,2), 'AUC Pixel': round(auroc_pixel*100, 2), 'AP Image':round(ap_image*100, 2), 'AP Pixel': round(ap_pixel*100, 2), 'PRO': round(aupro*100, 2)}
    score_df.loc[len(score_df)] = new_row

    
 


#전체
# mean_list = score_df.mean()
    
# if dataset == 'btad': #임시
#     mean_list = mean_list[1:]
# else:
#     pass

# new_row = {'Objects': 'mean', score_df.columns[1]: round(mean_list[0],2), score_df.columns[2]: round(mean_list[1],2), score_df.columns[3]:round(mean_list[2],2), score_df.columns[4]: round(mean_list[3],2), score_df.columns[5]: round(mean_list[4],2)}


mean_list = score_df.iloc[:, 1:].mean().round(2)

# 새 행 생성 (소수점 둘째자리까지 문자열로 표현)
new_row = {'Objects': 'mean',
    score_df.columns[1]: f"{mean_list.iloc[0]:.2f}", score_df.columns[2]: f"{mean_list.iloc[1]:.2f}", score_df.columns[3]: f"{mean_list.iloc[2]:.2f}", score_df.columns[4]: f"{mean_list.iloc[3]:.2f}", score_df.columns[5]: f"{mean_list.iloc[4]:.2f}"}




score_df.loc[len(score_df)] = new_row
score_df.to_csv(file_path, index=False)
print(score_df)