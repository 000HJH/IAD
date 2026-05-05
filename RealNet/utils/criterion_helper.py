import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feats_recon = input["feats_recon"]
        if "gt_block_feats" in input:
            gt_block_feats = input["gt_block_feats"]
            losses=[]
            for key in feats_recon:
                losses.append(self.criterion_mse(feats_recon[key], gt_block_feats[key]))
            return torch.sum(torch.stack(losses))
        else:
            return torch.tensor(np.array(0.)).to(input['image'].device)


class SegmentCrossEntropyLoss(nn.Module):
    def __init__(self, weight): #원본
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
    ###여기서 mask 역전파가 끊김!
    def forward(self, input):
        gt_mask = input["mask"]
        logit = input["logit"]
        bsz,_,h,w=logit.size()
        logit = logit.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long() # 1미만이면 0

        return self.criterion(logit,gt_mask)
    
    """GT mask에 역전파가 가능하도록 CE 재정의 -- GT mask는 역전파하면 안 됨!"""
    
    # def __init__(self, weight=1.0):
    #     super().__init__()
    #     self.weight = weight

    # def forward(self, input):
    #     # input dict에서 가져오기
    #     logit = input["logit"]  # (B, 2, H, W)
    #     mask  = input["mask"]   # (B, 1, H, W), 값은 0 또는 1 (Float)

    #     # (B,1,H,W) -> (B,H,W)
    #     mask = mask.squeeze(1)

    #     log_prob = F.log_softmax(logit, dim=1)  # (B,2,H,W)

    #     p0 = log_prob[:, 0, :, :]  # 배경 쪽 log(p)
    #     p1 = log_prob[:, 1, :, :]  # 이상 쪽 log(p)

    #     # 소프트 크로스 엔트로피(2클래스 버전)
    #     #   CE = - [ y*log(p1) + (1-y)*log(p0) ]
    #     ce = - (mask * p1 + (1.0 - mask) * p0)

    #     return self.weight * ce.mean()
    

class SegmentFocalLoss(nn.Module):
    def __init__(self, weight,
                 apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(SegmentFocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.weight=weight

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input):

        target= input['mask']
        logit = torch.softmax(input['logit'], dim=1)

        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)

        num_class = logit.shape[1]
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["ori"]
        image_rec = input["recon"]
        return self.criterion_mse(image, image_rec)


class ClassifierCrossEntropyLoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, input):
        pred = input["pred"]
        label= input["label"].long()
        
        print(pred)
        print(label)
        return self.criterion(pred,label)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
