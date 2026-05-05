import os
import sys
import glob
import torch
import numpy as np
import argparse
import logging
import tqdm
import warnings

from typing import Tuple, List
import backbones
import glass  # GLASS class
import utils  # set_torch_device, fix_seeds, etc.
import metrics

LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    # model load

    parser.add_argument('--date', type=str, required=True, default='./results/models/backbone_0/mvtec_bottle',
                        help="F.T 모델 폴더 .")
    # dataset
    parser.add_argument('--data_path', type=str, required=False,
                        help="../datasets/mvtec")
    parser.add_argument('--aug_path', type=str, default='../datasets/dtd/images',
                        help="If needed, path to external augmentation data (dtd, etc.)")
    # parser.add_argument('--subdataset', type=str, default="bottle",
    #                     help="Class name or subdataset name.")
    # test batch
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=288)
    parser.add_argument('--imagesize', type=int, default=288)
    # embed dims
    parser.add_argument('--pretrain_embed_dimension', type=int, default=1536)
    parser.add_argument('--target_embed_dimension', type=int, default=1536)
    # backbone
    parser.add_argument('--backbone_name', type=str, default='wideresnet50')
    parser.add_argument('--layers', nargs='+', default=['layer2', 'layer3'])
    # device
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    # etc
    parser.add_argument('--distribution', type=int, default=0)
    parser.add_argument('--mean', type=float, default=0.5)
    parser.add_argument('--std', type=float, default=0.1)

    return parser.parse_args()


# def load_dataset_for_test(args, dataset):
#     """
#     Simplified version: Only test set from e.g. MVTec or your custom dataset
#     This re-implements part of 'dataset' logic from main.py
#     """
#     import datasets.mvtec as mvtec_ds  # example
    
    

#     # For instance, if name="mvtec", we load MVTecDataset
#     # Here we assume your dataset class is something like MVTecDataset
#     # and we want the TEST split.
#     test_dataset = mvtec_ds.MVTecDataset(
#         data_path=args.data_path,
#         aug_path=args.aug_path,
#         classname=args.subdataset,
#         resize=args.resize,
#         imagesize=args.imagesize,
#         split=mvtec_ds.DatasetSplit.TEST,
#         seed=args.seed,
#     )
#     test_dataloader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         prefetch_factor=2,
#         pin_memory=True,
#     )
#     test_dataloader.name = args.subdataset
#     LOGGER.info(f"Test dataset {args.subdataset}: #samples={len(test_dataset)}")
#     return test_dataloader


def build_glass(args, device):
    """
    Build the GLASS model, load ckpt
    Only need 'discriminator', 'pre_projection' etc. 
    No trainer logic, just load & test.
    """
    # load backbone
    backbone = backbones.load(args.backbone_name)
    backbone.name = args.backbone_name

    glass_inst = glass.GLASS(device)
    # call glass_inst.load(...) with minimal needed params
    glass_inst.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        # some defaults
        patchsize=3,
        meta_epochs=1, 
        meta_steps=1,
        eval_epochs=1,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        train_backbone=False,
        pre_proj=1,
        mining=1,
        noise=0.015,
        radius=0.75,
        p=0.5,
        lr=1e-6,
        occ_lr=1e-4,
        svd=0,
        step=20,
        limit=392,
    )

    # set ckpt_dir
    # glass_inst.set_model_dir(args.ckpt_dir, args.subdataset)
    # glass_inst.ckpt_dir = args.ckpt_dir #os.path.join(self.model_dir, dataset_name)

    # skip trainer, directly load ckpt
    return glass_inst


def test_only(glass_inst, test_dataloader, dataset_name, category, date):
    # if not os.path.exists(f'ft_results/{date}/models/backbone_0/{dataset_name}_{category}/ckpt.pth'):
    #     print(f"No ckpt file found in {f'ft_results/{date}/models/backbone_0/{dataset_name}_{category}/ckpt.pth'}")
    #     sys.exit()
    #     return 0., 0., 0., 0., 0., -1.

    
    model_path_list = [
        f'./results/models/backbone_0/{dataset_name}_{category}/ckpt.pth', #원본
        # f'./results_ft400_2/models/backbone_0/{dataset_name}_{category}/ckpt.pth', #원본ft400
        f'ft_results/{date}/models/backbone_0/{dataset_name}_{category}/ckpt.pth'
    ]


    ensemble_pixel_scores = []   # 픽셀 단위 스코어 (N, H, W)
    ensemble_image_scores = []   # 이미지 단위 스코어 (N,)

    for model_path in model_path_list:



        # load
        state_dict = torch.load(model_path, map_location=glass_inst.device)
        if 'discriminator' in state_dict:
            glass_inst.discriminator.load_state_dict(state_dict['discriminator'])
            if "pre_projection" in state_dict:
                glass_inst.pre_projection.load_state_dict(state_dict["pre_projection"])
        else:
            glass_inst.load_state_dict(state_dict, strict=False)

        # run predict
        # images, scores, segmentations, labels_gt, masks_gt = glass_inst.predict(test_dataloader)
        images, anomaly_score, anomaly_map, image_targets, image_masks = glass_inst.predict(test_dataloader)
        
        
        """        
        images:  (83, 3, 288, 288)                                                                                               
        scores:  (83,)
        segmentations:  (83, 288, 288)
        labels_gt:  (83,)
        masks_gt:  (83, 1, 288, 288)"""
        
        anomaly_score = np.array(anomaly_score)
        anomaly_map = np.array(anomaly_map)
        
        '''정규화 (이미지 수준)'''
        image_score_norm = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)
        ensemble_image_scores.append(image_score_norm)
        
        '''정규화 (픽셀 수준)'''
        anomaly_map = np.array(anomaly_map)
        # anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        ensemble_pixel_scores.append(anomaly_map)
        
        
        
    #두 모델 통합
    ensemble_pixel_scores = np.array(ensemble_pixel_scores)
    ensemble_image_scores = np.array(ensemble_image_scores)
    
    '''(1) 모델 평균 score'''
    anomaly_score = np.mean(ensemble_image_scores, axis=0)  # 이미지 단위 평균
    anomaly_map = np.mean(ensemble_pixel_scores, axis=0)        # 픽셀 단위 평균

    
    # compute metrics
    image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = glass_inst._evaluate(
        images, anomaly_score, anomaly_map, image_targets, image_masks, dataset_name, path='eval'
    )

    return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, -1




def main():
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(f"Command line: {' '.join(sys.argv)}")
    args = parse_args()

    device = utils.set_torch_device([args.gpu])
    utils.fix_seeds(args.seed, device)


        
    if 'visa' in args.data_path:
        dataset = 'visa'
        obj_list = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
        import datasets.visa as visa  # example
        testdataset=visa.VisADataset
        testdataset_split=visa.DatasetSplit.TEST
    elif 'btad' in args.data_path:
        dataset = 'btad'
        obj_list = ['01','02','03']
        # obj_list = ['02']
        import datasets.btad as btad  # example
        testdataset=btad.BTadDataset
        testdataset_split=btad.DatasetSplit.TEST
    else:
        dataset = 'mvtec'
        obj_list=['bottle','cable','capsule','carpet','grid','hazelnut','leather', 'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
        # obj_list=['carpet','grid','leather','tile','wood']
        import datasets.mvtec as mvtec  # example
        testdataset=mvtec.MVTecDataset
        testdataset_split=mvtec.DatasetSplit.TEST

    # test_loader = load_dataset_for_test(args, dataset)


    glass_inst = build_glass(args, device)


    import pandas as pd
    csv_name = 'ensemble_score.csv' 
    score_df = pd.DataFrame({'Objects':[], 'AUC Image':[],'AUC Pixel':[], 'AP Image':[], 'AP Pixel':[], 'PRO':[]})
    score_df.to_csv(f'ft_results/{args.date}/{csv_name}', index=False)
    # score_df.to_csv(f'results_ft400/{csv_name}', index=False)


    for category in obj_list:
        test_dataset = testdataset(
            data_path=args.data_path,
            aug_path=args.aug_path,
            classname=category, #args.subdataset
            resize=args.resize,
            imagesize=args.imagesize,
            split=testdataset_split,
            seed=args.seed,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )
        test_dataloader.name = category #args.subdataset
        LOGGER.info(f"Test dataset {category}: #samples={len(test_dataset)}")
        
        
    
    
        # run test
        i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = test_only(glass_inst, test_dataloader, dataset ,category, args.date)
        # print(f"Test Results on {args.subdataset}: image_auroc={i_auroc}, pixel_auroc={p_auroc}, pixel_pro={p_pro}")
        print(
            "image_auroc: ", round(i_auroc*100, 2),
            "pixel_auroc: ", round(p_auroc*100, 2),
            "image_ap: ", round(i_ap*100, 2),
            "pixel_ap: ", round(p_ap*100, 2),
            "pixel_pro: ", round(p_pro*100, 2),
            'epoch: ',epoch
            )
        
        
        new_row = {'Objects': category, 'AUC Image': round(i_auroc*100,2), 'AUC Pixel': round(p_auroc*100, 2), 'AP Image':round(i_ap*100, 2), 'AP Pixel': round(p_ap*100, 2), 'PRO': round(p_pro*100, 2)}
        score_df.loc[len(score_df)] = new_row
        
    #전체 평균
    mean_list = score_df.iloc[:,1:].mean()

    new_row = {'Objects': 'mean', score_df.columns[1]: round(mean_list[0],2), score_df.columns[2]: round(mean_list[1],2), score_df.columns[3]:round(mean_list[2],2), score_df.columns[4]: round(mean_list[3],2), score_df.columns[5]: round(mean_list[4],2)}


    score_df.loc[len(score_df)] = new_row
    score_df.to_csv(f'ft_results/{args.date}/{csv_name}', index=False)
    # score_df.to_csv(f'results_ft400_2/{csv_name}', index=False)


if __name__=="__main__":
    main()
#