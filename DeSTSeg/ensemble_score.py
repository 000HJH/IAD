import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from data.visa_dataset import VisADataset
from data.btad_dataset import BTadDataset
from model.destseg import DeSTSeg
# from model.metrics import AUPRO, IAPS

import sys
from sklearn.metrics import roc_auc_score, average_precision_score
from _utils.metrics import compute_pro
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


def evaluate(args, category, TestDataset, dataset_name,  global_step=0):

    assert os.path.exists(
        os.path.join(args.checkpoint_path,args.date, args.base_model_name + category + ".pckl")
    )
    
    model_path_list = [
        os.path.join(args.checkpoint_path[:args.checkpoint_path.find('_ft')],  f"DeSTSeg_{dataset_name}_5000_{category}.pckl"), #원본
        # os.path.join(args.date, args.base_model_name + category + ".pckl") #ft400
        os.path.join(args.checkpoint_path,args.date, args.base_model_name + category + ".pckl")
    ] 

    

    
    # dataset = MVTecDataset(
    dataset = TestDataset(
        False,
        args.mvtec_path + category + "/test/",
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
    )
    
            
    
    '''앙상블'''
    ensemble_pixel_scores = []   # 픽셀 단위 스코어 (N, H, W)
    ensemble_image_scores = []   # 이미지 단위 스코어 (N,)
        
    for model_path in model_path_list:
        
        total_gt_pixel_scores = []
        de_st_total_pixel_scores = []
        seg_total_pixel_scores = []

        total_gt_sample_scores = []
        de_st_total_sample_scores= []
        seg_total_sample_scores  = []

        

        model = DeSTSeg(dest=True, ed=True).cuda()
        
        # model.load_state_dict(
        #     torch.load(model_path)
        # )
        
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        # 원하는 GPU로 이동
        device = torch.device(f"cuda:{args.gpu_id}")
        model.to(device)
    
    

        model.eval()
        with torch.no_grad():


            for _, sample_batched in enumerate(dataloader):
                img = sample_batched["img"].cuda()
                mask = sample_batched["mask"].to(torch.int64).cuda()

                output_segmentation, output_de_st, output_de_st_list = model(img)

                output_segmentation = F.interpolate(
                    output_segmentation,
                    size=mask.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # output_de_st = F.interpolate(
                #     output_de_st, size=mask.size()[2:], mode="bilinear", align_corners=False
                # )

                mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0] #이미지 GT
                output_segmentation_sample, _ = torch.sort(
                    output_segmentation.view(output_segmentation.size(0), -1),
                    dim=1,
                    descending=True,
                )
                output_segmentation_sample = torch.mean(
                    output_segmentation_sample[:, : args.T], dim=1
                )
                # output_de_st_sample, _ = torch.sort(
                #     output_de_st.view(output_de_st.size(0), -1), dim=1, descending=True
                # )
                # output_de_st_sample = torch.mean(output_de_st_sample[:, : args.T], dim=1)


                #픽셀 수준
                total_gt_pixel_scores.extend(mask.cpu().numpy())
                seg_total_pixel_scores.extend(output_segmentation.cpu().numpy())
                # de_st_total_pixel_scores.extend(output_de_st.cpu().numpy())
                
                #이미지 수준
                total_gt_sample_scores.extend(mask_sample.cpu().numpy())
                seg_total_sample_scores.extend(output_segmentation_sample.cpu().numpy())
                # de_st_total_sample_scores.extend(output_de_st_sample.cpu().numpy())
                
                
            
            '''정규화 (이미지 수준)'''
            # seg_total_sample_scores = seg_total_sample_scores.cpu().numpy()
            
            seg_total_sample_scores = np.array(seg_total_sample_scores)
            image_score_norm = (seg_total_sample_scores - seg_total_sample_scores.min()) / (seg_total_sample_scores.max() - seg_total_sample_scores.min() + 1e-8)
            ensemble_image_scores.append(image_score_norm)
            
            '''정규화 (픽셀 수준)'''
            
            seg_total_pixel_scores = np.array(seg_total_pixel_scores)
            seg_total_pixel_scores = (seg_total_pixel_scores - seg_total_pixel_scores.min()) / (seg_total_pixel_scores.max() - seg_total_pixel_scores.min() + 1e-8)
            ensemble_pixel_scores.append(seg_total_pixel_scores)


    

    # #픽셀
    # total_gt_pixel_scores = np.squeeze(np.array(total_gt_pixel_scores),1)
    # ensemble_pixel_scores = np.squeeze(np.array(ensemble_pixel_scores),1)
    # # de_st_total_pixel_scores = np.squeeze(np.array(de_st_total_pixel_scores),1)
    
    # #이미지
    # total_gt_sample_scores = np.array(total_gt_sample_scores)
    # ensemble_image_scores = np.array(ensemble_image_scores)
    # # de_st_total_sample_scores = np.array(de_st_total_sample_scores)
    
    total_gt_pixel_scores = np.array(total_gt_pixel_scores)
    total_gt_sample_scores = np.array(total_gt_sample_scores)
    
    
    

    '''(1) 모델 평균 score'''
    anomaly_score_prediction = np.mean(ensemble_image_scores, axis=0)  # 이미지 단위 평균
    total_pixel_scores = np.mean(ensemble_pixel_scores, axis=0)        # 픽셀 단위 평균

    '''(2) 모델 MAX score'''        
    # anomaly_score_prediction = np.maximum(ensemble_image_scores[0], ensemble_image_scores[1])  # 이미지 단위 
    # total_pixel_scores = np.maximum(ensemble_pixel_scores[0], ensemble_pixel_scores[1])        # 픽셀 단위 
    

    
    
    """de st"""
    # #픽셀
    # de_st_aupro = compute_pro(de_st_total_pixel_scores, total_gt_pixel_scores)
    # de_st_auroc_pixel = roc_auc_score(total_gt_pixel_scores.reshape(-1).astype(int), de_st_total_pixel_scores.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
    # de_st_ap_pixel = average_precision_score(total_gt_pixel_scores.reshape(-1).astype(int), de_st_total_pixel_scores.reshape(-1))
    # #이미지
    # de_st_auroc_sample = roc_auc_score(total_gt_sample_scores.reshape(-1).astype(int), de_st_total_sample_scores.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
    # de_st_ap_sample = average_precision_score(total_gt_sample_scores.reshape(-1).astype(int), de_st_total_sample_scores.reshape(-1))

    total_pixel_scores = total_pixel_scores.squeeze(1)
    total_gt_pixel_scores = total_gt_pixel_scores.squeeze(1)
    
    """seg"""
    #픽셀
    seg_aupro = compute_pro(total_pixel_scores, total_gt_pixel_scores)
    seg_auroc_pixel = roc_auc_score(total_gt_pixel_scores.reshape(-1).astype(int), total_pixel_scores.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
    seg_ap_pixel = average_precision_score(total_gt_pixel_scores.reshape(-1).astype(int), total_pixel_scores.reshape(-1))

    #이미지
    seg_auroc_sample = roc_auc_score(total_gt_sample_scores.reshape(-1).astype(int), anomaly_score_prediction.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
    seg_ap_sample = average_precision_score(total_gt_sample_scores.reshape(-1).astype(int), anomaly_score_prediction.reshape(-1))
    
    
    
    print('AUROC_Image: ',round(seg_auroc_sample,4))
    print('AUROC_pixel: ',round(seg_auroc_pixel,4))
    print('AP_Image: ',round(seg_ap_sample,4))
    print('AP_pixel: ',round(seg_ap_pixel,4))
    print('AUPRO: ',round(seg_aupro,4))
    print('==============')

    score_df = pd.read_csv(f'{args.checkpoint_path}/{args.date}/{args.date}_{csv_name}')
    new_row = {'Objects': category, 'AUC Image': round(seg_auroc_sample*100,2), 'AUC Pixel': round(seg_auroc_pixel*100, 2), 'AP Image':round(seg_ap_sample*100, 2), 'AP Pixel': round(seg_ap_pixel*100, 2), 'PRO': round(seg_aupro*100, 2)}
    score_df.loc[len(score_df)] = new_row


    score_df.to_csv(f'{args.checkpoint_path}/{args.date}/{args.date}_{csv_name}', index=False)




if __name__ == "__main__":
    csv_name = "ensemble_score.csv"

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="../datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="../datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model_ft/") #
    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_MVTec_5000_")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    parser.add_argument("--date", type=str, default="date/")
    parser.add_argument("--dataset", type=str, default="mvtec") #$

    args = parser.parse_args()



    if ('mvtec' in args.base_model_name.lower()) & ('loce' not in args.base_model_name.lower()):
        TestDataset = MVTecDataset
        dataset_name = 'MVTec'
        obj_list=['bottle','cable','capsule','carpet','grid','hazelnut','leather', 'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
        # obj_list=['carpet','grid','leather','tile','wood']
    elif 'visa' in args.base_model_name.lower():
        TestDataset = VisADataset
        dataset_name = 'VisA'
        args.mvtec_path = '../datasets/visa/'
        args.checkpoint_path = './visa_saved_model_ft/'
        obj_list = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
    # elif data_path.split('/')[-2] == 'mvtecloco':
    #     TestDataset = MVTecLOCODRAEMTestDataset
    elif 'btad' in args.base_model_name.lower():
        TestDataset = BTadDataset
        dataset_name = 'BTad'
        args.mvtec_path = '../datasets/btad/'
        args.checkpoint_path = './btad_saved_model_ft/'
        obj_list = ['01','02','03']
        # obj_list = ['02']
    # elif 'loco' in args.base_model_name:
    #     TestDataset = MVTeclocoDRAEMTestDataset
    #     pre_path = '_loco'
    #     obj_list = ['splicing_connectors','juice_bottle','breakfast_box','screw_bag','pushpins']
    else:
        print('configs 파일명 확인 필요')


    #결과 csv
    score_df = pd.DataFrame({'Objects':[], 'AUC Image':[],'AUC Pixel':[], 'AP Image':[], 'AP Pixel':[], 'PRO':[]})
    score_df.to_csv(f'{args.checkpoint_path}/{args.date}/{args.date}_{csv_name}', index=False)



    # obj_list = args.category
    
    # for obj in obj_list:
    #     assert obj in ALL_CATEGORY

    with torch.cuda.device(args.gpu_id):
        for obj in obj_list:
            print(obj)
            # test(args, obj)
            evaluate(args, obj, TestDataset, dataset_name)

    
        #전체 평균
        score_df = pd.read_csv(f'{args.checkpoint_path}/{args.date}/{args.date}_{csv_name}')
        mean_list = score_df.iloc[:,1:].mean()
        
        
        new_row = {'Objects': 'mean', score_df.columns[1]: round(mean_list[0],2), score_df.columns[2]: round(mean_list[1],2), score_df.columns[3]:round(mean_list[2],2), score_df.columns[4]: round(mean_list[3],2), score_df.columns[5]: round(mean_list[4],2)}

        score_df.loc[len(score_df)] = new_row
        score_df.to_csv(f'{args.checkpoint_path}/{args.date}/{args.date}_{csv_name}', index=False)   
