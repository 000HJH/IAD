"""결과 csv 생성 버전"""
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
parser.add_argument("--config", default="experiments/{}/realnet.yaml") #원본 모델
parser.add_argument("--experiments", default="experiments_ft") #$
# parser.add_argument("--config", default="experiments_ft/{}/ft_realnet.yaml") #F.T 모델
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
    # args.checkpoints_folder = os.path.join(config.exp_path, config.saver.checkpoints_dir,args.class_name)
    # args.model_path=os.path.join(args.checkpoints_folder,"ckpt_best.pth.tar") ##best 사용
    ####
    import glob
    args.checkpoints_folder= os.path.join(config.exp_path, config.saver.checkpoints_dir, args.date, args.class_name)#$
    # model_path_list = glob.glob(os.path.join(args.checkpoints_folder,"ckpt_*.pth.tar"))
    # assert len(model_path_list) == 1 #파라미터 파일은 1개만 존재

    if os.path.isfile(os.path.join(args.checkpoints_folder,"ckpt_500.pth.tar")):
        args.model_path = os.path.join(args.checkpoints_folder,"ckpt_500.pth.tar") #원본
    else:
        args.model_path = glob.glob(os.path.join(args.checkpoints_folder,"ckpt*.pth.tar"))[0] #F.T
        
    # args.model_path = os.path.join(args.checkpoints_folder,"ckpt_500.pth.tar") #원본
        

    ####

    config=update_config(config,args)
    set_seed(config.random_seed)

    config.evaluator.metrics['auc'].append({'name':'pro'})

    config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir)
    os.makedirs(config.vis_path, exist_ok=True)

    _, val_loader = build_dataloader(config.dataset,distributed=False)

    model = ModelHelper(config.net)
    model.cuda()

    state_dict=torch.load(args.model_path)
    model.load_state_dict(state_dict['state_dict'],strict=False)

    ret_metrics = validate(config,val_loader, model,args.class_name)
    print(ret_metrics)
    # print_metrics(ret_metrics, config.evaluator.metrics, args.class_name)


    ## CSV 저장
    auroc =ret_metrics[f'{args.class_name}_image_auroc']
    auroc_pixel =ret_metrics[f'{args.class_name}_pixel_auroc']
    aupr =ret_metrics[f'{args.class_name}_image_aupr']
    aupr_pixel =ret_metrics[f'{args.class_name}_pixel_aupr']
    aupro =ret_metrics[f'{args.class_name}_pro_auroc']
    
    # new_row = {'Objects': obj_name, 'AUC Image': round(auroc*100,2), 'AUC Pixel': round(auroc_pixel*100, 2), 'AP Image':round(ap*100, 2), 'AP Pixel': round(ap_pixel*100, 2), 'PRO': round(aupro*100, 2)}
    new_row = {'Objects': args.class_name, 'AUC Image': round(auroc*100,2), 'AUC Pixel': round(auroc_pixel*100, 2), 'AP Image': round(aupr*100, 2), 'AP Pixel': round(aupr_pixel*100, 2), 'PRO': round(aupro*100, 2)}
    score_df = pd.read_csv(f"./{args.experiments}/{args.dataset}/{args.checkpoints_dir}/{args.date}/{args.csv_name}")
    score_df.loc[len(score_df)] = new_row
    score_df.to_csv(f"./{args.experiments}/{args.dataset}/{args.checkpoints_dir}/{args.date}/{args.csv_name}", index=False)
    
    
def print_metrics(ret_metrics, config, class_name):
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"]))
    clsnames.sort()

    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        evalnames.sort()

        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        print(f"\n{record}")



def validate(config,val_loader, model,class_name):

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

    # preds = np.squeeze(np.concatenate(np.asarray(preds), axis=0),axis=1)  # N x H x W
    # masks = np.squeeze(np.concatenate(np.asarray(masks), axis=0),axis=1)  # N x H x W
    preds = np.squeeze(np.concatenate(preds, axis=0), axis=1)
    masks = np.squeeze(np.concatenate(masks, axis=0), axis=1)


    ret_metrics = performances(class_name, preds, masks, config.evaluator.metrics)

    preds_cls = []
    masks_cls = []
    image_paths = []

    for fileinfo, pred, mask in zip(fileinfos, preds, masks):
        preds_cls.append(pred[None, ...])
        masks_cls.append(mask[None, ...])
        image_paths.append(fileinfo['filename'])

    preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
    masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
    masks_cls[masks_cls != 0.0] = 1.0

    precision, recall, thresholds = precision_recall_curve(masks_cls.flatten(), preds_cls.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]
    # export_segment_images(config, image_paths, masks_cls, preds_cls, seg_threshold, class_name)
    return ret_metrics


if __name__ == "__main__":
    args = parser.parse_args() #$
    
    args.csv_name = args.date + '.csv' #$
    csv_dir = f"./{args.experiments}/{args.dataset}/{args.checkpoints_dir}/{args.date}"
    score_df = pd.DataFrame({'Objects':[], 'AUC Image':[],'AUC Pixel':[],'AP Image':[],'AP Pixel':[], 'PRO':[]})
    score_df.to_csv(os.path.join(csv_dir,args.csv_name), index=False)

    if args.dataset == 'MVTec-AD':
        class_list = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
    elif args.dataset == 'VisA':
        class_list = ["candle","capsules","cashew","chewinggum","fryum","macaroni1","macaroni2","pcb1","pcb2","pcb3","pcb4","pipe_fryum"]
    elif args.dataset == 'BTAD':
        class_list = ["01","02","03"]
    else:
        print('Dataset 잘못 입력')
    
    print('args.csv_name: ',args.csv_name)
    for class_name in class_list:
        args.class_name = class_name
        main(args)

    #평균 산정
    score_df = pd.read_csv(os.path.join(csv_dir, args.csv_name))
    mean_list = score_df.iloc[:,1:].mean()

    # new_row = {'Objects': 'mean', score_df.columns[1]: round(mean_list[0],2), score_df.columns[2]: round(mean_list[1],2), score_df.columns[3]:round(mean_list[2],2), score_df.columns[4]: round(mean_list[3],2), score_df.columns[5]: round(mean_list[4],2)}
    new_row = {'Objects': 'mean', score_df.columns[1]: round(mean_list[0],2), score_df.columns[2]: round(mean_list[1],2), score_df.columns[3]:round(mean_list[2],2), score_df.columns[4]:round(mean_list[3],2), score_df.columns[5]:round(mean_list[4],2)}
    score_df.loc[len(score_df)] = new_row
    score_df.to_csv(os.path.join(csv_dir,args.csv_name), index=False)
    
    #CUDA_VISIBLE_DEVICES=2 python _evaluation_realnet.py --dataset MVTec-AD --date realnet_mvtec_lr00001_lrocc001_500s