import warnings
import argparse
import torch
from datasets.data_builder import build_dataloader
from easydict import EasyDict
import yaml
import os
from utils.misc_helper import set_seed
from models.model_helper import ModelHelper
from utils.eval_helper import performances
from sklearn.metrics import precision_recall_curve
import numpy as np
# from utils.visualize import export_segment_images
from utils.eval_helper import Report
from train_realnet import update_config
from utils.categories import Categories
import pandas as pd


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="evaluation RealNet")
# parser.add_argument("--config", default="experiments/{}/realnet.yaml")
parser.add_argument("--config", default="experiments_ft/{}/ft_realnet.yaml")
parser.add_argument("--experiments", default="experiments_ft") #$
parser.add_argument("--dataset", default="MVTec-AD",choices=['MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--date", default=".")#$
parser.add_argument("--checkpoints_dir", default="realnet_checkpoints/")#$
parser.add_argument("--class_name", default="bottle",choices=[
        # mvtec-ad
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
        # visa
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
        #mpdd
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
        # btad
         "01",
         "02",
         "03",
        ] )


def main(args):

    class_name_list=Categories[args.dataset]

    assert args.class_name in class_name_list

    args.config=args.config.format(args.dataset)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    
    config.exp_path = os.path.dirname(args.config)

    # F.T 파일 경로
    import glob
    config.saver.checkpoints_dir = args.checkpoints_dir
    args.checkpoints_folder= os.path.join(config.exp_path, config.saver.checkpoints_dir,  args.class_name)#$
    ft_model_path_list = glob.glob(os.path.join(args.checkpoints_folder,"ckpt_*.pth.tar")) #500epoch
    

    assert len(ft_model_path_list) == 1 #파라미터 파일은 1개 존재
    # args.model_path= model_path_list[0]
    
    paths = [
        f'experiments/{args.dataset}/realnet_checkpoints/{args.class_name}/ckpt_500.pth.tar', #원본
        ft_model_path_list[0] #F.T
    ]


    config=update_config(config,args)
    set_seed(config.random_seed)
    
    ensemble_pixel_scores = []   # 픽셀 단위 스코어 (N, H, W)
    ensemble_image_scores = []   # 이미지 단위 스코어 (N,)
    
    for model_path in paths:

        '''파라미터 앙상블'''
        model = ModelHelper(config.net)
        model.cuda()
        state_dict=torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'],strict=False)
        

        
        config.evaluator.metrics['auc'].append({'name':'pro'})

        config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir)
        os.makedirs(config.vis_path, exist_ok=True)

        _, val_loader = build_dataloader(config.dataset,distributed=False)


        model.eval()

        fileinfos = []
        preds = []
        masks = []

        with torch.no_grad():
            for i, input in enumerate(val_loader):
                # forward
                outputs = model(input,train=False)

                for j in range(len(outputs['filename'])):
                    fileinfos.append(
                        {
                            "filename": str(outputs["filename"][j]),
                            "height": int(outputs["height"][j]),
                            "width": int(outputs["width"][j]),
                            "clsname": str(outputs["clsname"][j]),
                        }
                    )
                preds.append(outputs["anomaly_score"].cpu().numpy())
                masks.append(outputs["mask"].cpu().numpy())

        preds = np.squeeze(np.concatenate(preds, axis=0), axis=1)
        masks = np.squeeze(np.concatenate(masks, axis=0), axis=1)
                
                
        '''정규화 (픽셀 수준) -> 이미지 수준은 performance 함수 내에서 처리'''
        # anomaly_map = np.array(anomaly_map)
        preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        ensemble_pixel_scores.append(preds)

    # 두 모델 score 평균 처리
    preds = np.mean(ensemble_pixel_scores, axis=0) 
    
    
    ret_metrics = performances(args.class_name, preds, masks, config.evaluator.metrics)



    print(ret_metrics)
    
    ## CSV 저장
    auroc =ret_metrics[f'{args.class_name}_image_auroc']
    auroc_pixel =ret_metrics[f'{args.class_name}_pixel_auroc']
    aupr =ret_metrics[f'{args.class_name}_image_aupr']
    aupr_pixel =ret_metrics[f'{args.class_name}_pixel_aupr']
    aupro =ret_metrics[f'{args.class_name}_pro_auroc']
    
    # new_row = {'Objects': obj_name, 'AUC Image': round(auroc*100,2), 'AUC Pixel': round(auroc_pixel*100, 2), 'AP Image':round(ap*100, 2), 'AP Pixel': round(ap_pixel*100, 2), 'PRO': round(aupro*100, 2)}
    new_row = {'Objects': args.class_name, 'AUC Image': round(auroc*100,2), 'AUC Pixel': round(auroc_pixel*100, 2), 'AP Image': round(aupr*100, 2), 'AP Pixel': round(aupr_pixel*100, 2), 'PRO': round(aupro*100, 2)}
    score_df = pd.read_csv(f"./{args.experiments}/{args.dataset}/{args.checkpoints_dir}/{args.csv_name}")
    score_df.loc[len(score_df)] = new_row
    score_df.to_csv(f"./{args.experiments}/{args.dataset}/{args.checkpoints_dir}/{args.csv_name}", index=False)
    
    

if __name__ == "__main__": 
    args = parser.parse_args() #$
    
    args.csv_name = 'realnet_ensemble_score.csv' #$
    csv_dir = f"./{args.experiments}/{args.dataset}/{args.checkpoints_dir}"
    score_df = pd.DataFrame({'Objects':[], 'AUC Image':[],'AUC Pixel':[],'AP Image':[],'AP Pixel':[], 'PRO':[]})
    score_df.to_csv(os.path.join(csv_dir,args.csv_name), index=False)

    if args.dataset == 'MVTec-AD':
        class_list = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
        # class_list = ['carpet','grid','leather','tile','wood']
    elif args.dataset == 'VisA':
        class_list = ["candle","capsules","cashew","chewinggum","fryum","macaroni1","macaroni2","pcb1","pcb2","pcb3","pcb4","pipe_fryum"]
    elif args.dataset == 'BTAD':
        class_list = ["01","02","03"]
    else:
        print('Dataset 잘못 입력')
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    for class_name in class_list:
        args.class_name = class_name
        main(args)

    #평균 산정
    score_df = pd.read_csv(os.path.join(csv_dir,args.csv_name))
    mean_list = score_df.iloc[:,1:].mean()

    # new_row = {'Objects': 'mean', score_df.columns[1]: round(mean_list[0],2), score_df.columns[2]: round(mean_list[1],2), score_df.columns[3]:round(mean_list[2],2), score_df.columns[4]: round(mean_list[3],2), score_df.columns[5]: round(mean_list[4],2)}
    new_row = {'Objects': 'mean', score_df.columns[1]: round(mean_list[0],2), score_df.columns[2]: round(mean_list[1],2), score_df.columns[3]:round(mean_list[2],2), score_df.columns[4]:round(mean_list[3],2), score_df.columns[5]:round(mean_list[4],2)}
    score_df.loc[len(score_df)] = new_row
    score_df.to_csv(os.path.join(csv_dir,args.csv_name), index=False)