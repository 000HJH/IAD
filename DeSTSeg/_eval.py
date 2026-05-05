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
from model.destseg import DeSTSeg
# from model.metrics import AUPRO, IAPS

import sys
from sklearn.metrics import roc_auc_score, average_precision_score
from _utils.metrics import compute_pro
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


def evaluate(args, category, model, visualizer, global_step=0):

        
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    obj_pro_list = []
    
    
    total_gt_pixel_scores = []
    de_st_total_pixel_scores = []
    seg_total_pixel_scores = []

    total_gt_sample_scores = []
    de_st_total_sample_scores= []
    seg_total_sample_scores  = []

    model.eval()
    with torch.no_grad():
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=args.mvtec_path + category + "/test/",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
        )
        # de_st_IAPS = IAPS().cuda()
        # de_st_AUPRO = AUPRO().cuda()
        # de_st_AUROC = AUROC().cuda()
        # de_st_AP = AveragePrecision().cuda()
        # de_st_detect_AUROC = AUROC().cuda()
        # seg_IAPS = IAPS().cuda()
        # seg_AUPRO = AUPRO().cuda()
        # seg_AUROC = AUROC().cuda()
        # seg_AP = AveragePrecision().cuda()
        # seg_detect_AUROC = AUROC().cuda()

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
            output_de_st = F.interpolate(
                output_de_st, size=mask.size()[2:], mode="bilinear", align_corners=False
            )

            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0] #이미지 GT
            output_segmentation_sample, _ = torch.sort(
                output_segmentation.view(output_segmentation.size(0), -1),
                dim=1,
                descending=True,
            )
            output_segmentation_sample = torch.mean(
                output_segmentation_sample[:, : args.T], dim=1
            )
            output_de_st_sample, _ = torch.sort(
                output_de_st.view(output_de_st.size(0), -1), dim=1, descending=True
            )
            output_de_st_sample = torch.mean(output_de_st_sample[:, : args.T], dim=1)


            ''''''
            # de_st_IAPS.update(output_de_st, mask)
            # de_st_AUPRO.update(output_de_st, mask)
            # de_st_AP.update(output_de_st.flatten(), mask.flatten())
            # de_st_AUROC.update(output_de_st.flatten(), mask.flatten())
            # de_st_detect_AUROC.update(output_de_st_sample, mask_sample)

            # seg_IAPS.update(output_segmentation, mask)
            # seg_AUPRO.update(output_segmentation, mask)
            # seg_AP.update(output_segmentation.flatten(), mask.flatten())
            # seg_AUROC.update(output_segmentation.flatten(), mask.flatten())
            # seg_detect_AUROC.update(output_segmentation_sample, mask_sample)
            ''''''

            #픽셀 수준
            total_gt_pixel_scores.extend(mask.cpu().numpy())
            de_st_total_pixel_scores.extend(output_de_st.cpu().numpy())
            seg_total_pixel_scores.extend(output_segmentation.cpu().numpy())
            
            #이미지 수준
            total_gt_sample_scores.extend(mask_sample.cpu().numpy())
            de_st_total_sample_scores.extend(output_de_st_sample.cpu().numpy())
            seg_total_sample_scores.extend(output_segmentation_sample.cpu().numpy())



        #픽셀
        total_gt_pixel_scores = np.squeeze(np.array(total_gt_pixel_scores),1)
        de_st_total_pixel_scores = np.squeeze(np.array(de_st_total_pixel_scores),1)
        seg_total_pixel_scores = np.squeeze(np.array(seg_total_pixel_scores),1)
        #이미지
        total_gt_sample_scores = np.array(total_gt_sample_scores)
        de_st_total_sample_scores = np.array(de_st_total_sample_scores)
        seg_total_sample_scores = np.array(seg_total_sample_scores)
        

        """de st"""
        #픽셀
        de_st_aupro = compute_pro(de_st_total_pixel_scores, total_gt_pixel_scores)
        de_st_auroc_pixel = roc_auc_score(total_gt_pixel_scores.reshape(-1).astype(int), de_st_total_pixel_scores.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
        de_st_ap_pixel = average_precision_score(total_gt_pixel_scores.reshape(-1).astype(int), de_st_total_pixel_scores.reshape(-1))
        #이미지
        de_st_auroc_sample = roc_auc_score(total_gt_sample_scores.reshape(-1).astype(int), de_st_total_sample_scores.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
        de_st_ap_sample = average_precision_score(total_gt_sample_scores.reshape(-1).astype(int), de_st_total_sample_scores.reshape(-1))

        """seg"""
        #픽셀
        seg_aupro = compute_pro(seg_total_pixel_scores, total_gt_pixel_scores)
        seg_auroc_pixel = roc_auc_score(total_gt_pixel_scores.reshape(-1).astype(int), seg_total_pixel_scores.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
        seg_ap_pixel = average_precision_score(total_gt_pixel_scores.reshape(-1).astype(int), seg_total_pixel_scores.reshape(-1))

        #이미지
        seg_auroc_sample = roc_auc_score(total_gt_sample_scores.reshape(-1).astype(int), seg_total_sample_scores.reshape(-1)) #pixel roc 두 개 계산이 4초 걸림
        seg_ap_sample = average_precision_score(total_gt_sample_scores.reshape(-1).astype(int), seg_total_sample_scores.reshape(-1))
        
        
        
        obj_ap_pixel_list.append(de_st_ap_pixel)
        obj_auroc_pixel_list.append(de_st_auroc_pixel)
        obj_auroc_image_list.append(de_st_auroc_sample)
        obj_ap_image_list.append(de_st_ap_sample)
        obj_pro_list.append(de_st_aupro)

        print('AUROC_Image: ',round(seg_auroc_sample,4),'\t',round(de_st_auroc_sample,4))
        print('AUROC_pixel: ',round(seg_auroc_pixel,4),'\t',round(de_st_auroc_pixel,4))
        print('AP_Image: ',round(seg_ap_sample,4),'\t',round(de_st_ap_sample,4))
        print('AP_pixel: ',round(seg_ap_pixel,4),'\t',round(de_st_ap_pixel,4))
        print('AUPRO: ',round(seg_aupro,4),'\t',round(de_st_aupro,4))
        print('==============')

        # score_df = pd.read_csv(f'{args.checkpoint_path}/{args.date}/{args.date}.csv')
        score_df = pd.read_csv(f'{args.checkpoint_path}/{args.date}.csv')
        new_row = {'Objects': category, 'AUC Image': round(seg_auroc_sample*100,2), 'AUC Pixel': round(seg_auroc_pixel*100, 2), 'AP Image':round(seg_ap_sample*100, 2), 'AP Pixel': round(seg_ap_pixel*100, 2), 'PRO': round(seg_aupro*100, 2)}
        score_df.loc[len(score_df)] = new_row


        # score_df.to_csv(f'{args.checkpoint_path}/{args.date}/{args.date}.csv', index=False)
        score_df.to_csv(f'{args.checkpoint_path}/{args.date}.csv', index=False)


def test(args, category):    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"DeSTSeg_MVTec_test_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    assert os.path.exists(
        os.path.join(args.checkpoint_path, args.base_model_name + category + ".pckl")
    )
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.checkpoint_path, args.base_model_name + category + ".pckl"
            )
        )
    )

    evaluate(args, category, model, visualizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="../datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="../datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_MVTec_5000_")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    parser.add_argument("--date", type=str, default="./")

    args = parser.parse_args()

    #결과 csv
    score_df = pd.DataFrame({'Objects':[], 'AUC Image':[],'AUC Pixel':[], 'AP Image':[], 'AP Pixel':[], 'PRO':[]})
    score_df.to_csv(f'{args.checkpoint_path}/{args.date}.csv', index=False)
    
    
    obj_list = args.category
    
    for obj in obj_list:
        assert obj in ALL_CATEGORY

    with torch.cuda.device(args.gpu_id):
        for obj in obj_list:
            print(obj)
            test(args, obj)
