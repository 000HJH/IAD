import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math #$

'''ADIOS--unet.py'''  
class ConvReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.ReLU(inplace=True)
        )

class ConvINReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True), #instace마다 Batch단위로 Normalization
            nn.ReLU(inplace=True)
        )

class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True)
        )
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

        
"""  
class UNet(nn.Module):#$ ADIOS unet.py

    def __init__(self, num_blocks=3, img_size=256, #$ num_blockm, img_size=64
                 filter_start=32, in_chnls=4, out_chnls=1,
                 norm='in'):
        super(UNet, self).__init__()
        # TODO(martin): make more general
        c = filter_start
        if norm == 'in':
            conv_block = ConvINReLU #$ B.ConvINReLU
        elif norm == 'gn':
            conv_block = ConvGNReLU
        else:
            conv_block = ConvReLU
        
        
        if num_blocks == 3:
            enc_in = [in_chnls, c, 2*c]
            enc_out = [c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 2*c] #dec_in 전에 skip add해서 channel 2배됨
            dec_out = [2*c, c, c]
        elif num_blocks == 4:
            enc_in = [in_chnls, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]


        self.down = []
        self.up = []
        # 3x3 kernels, stride 1, padding 1
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2**(num_blocks) #16
        
        #mlp는 전역적 맥락 정보를 위해 축소 후 확장
        self.mlp = nn.Sequential(
            Flatten(), # [8, -1]로 flatten
            nn.Linear(2*c*self.featuremap_size**2, 32), nn.ReLU(), #64 * 16^2
            nn.Dropout(0.3),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2*c*self.featuremap_size**2), nn.ReLU()
        )
        if out_chnls > 0:
            self.final_conv = nn.Conv2d(c, out_chnls, 1)
        else:
            self.final_conv = nn.Identity()
        self.out_chnls = out_chnls

    def forward(self, x): #이미지마다 1번씩
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        # Down
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down):
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True) #feature size 1/2
            x_down.append(act)

        
        #x_up = x_down[-1]
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size, self.featuremap_size)

        # Up
        for i, block in enumerate(self.up):
            #features = torch.cat([x_up, skip[-1 - i]], dim=1) #down 블럭의 역순으로 skip connection 
            features = torch.cat([x_up, x_down[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up):
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest', recompute_scale_factor=True)

        return self.final_conv(x_up)

#masking 모델
class OcclusionNetwork(nn.Module):#$
    def __init__(self, filter_start=8, in_chnls=3, N = 1, norm='gn'): #N= 4(6) -> 1 : 기존에는 각 마스크의 레이어에서 각기 다른 이미지 내 물체를 탐지했음 i)
        super(OcclusionNetwork, self).__init__()
        self.mask_encoder = UNet(
            num_blocks=4,  #$ 변경 필요
            img_size=256,  #$ 변경 필요
            filter_start=filter_start,  # mask_fbase
            in_chnls=in_chnls,
            out_chnls=-1,
            norm=norm
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(filter_start, N, 1, 1, 0),  # input 채널과 output 채널(mask 수) 변경 필요
            nn.Sigmoid()
            #nn.Softmax(dim=1) #$
        )

    def forward(self, gray_batch):
        batch_feature = self.mask_encoder(gray_batch)
        out = self.mask_head(batch_feature)
        return out
    
    
    
"""     
class UNet(nn.Module):#$ ADIOS unet.py

    def __init__(self, num_blocks=6, img_size=256, #$ num_blockm, img_size=64
                 filter_start=32, in_chnls=4, out_chnls=1,
                 norm='in'):
        super(UNet, self).__init__()
        # TODO(martin): make more general
        c = filter_start
        if norm == 'in':
            conv_block = ConvINReLU #$ B.ConvINReLU
        elif norm == 'gn':
            conv_block = ConvGNReLU
        else:
            conv_block = ConvReLU
        

        if num_blocks == 4:
            enc_in = [in_chnls, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]

        self.down = []
        self.up = []
        
        #self.dropout = nn.Dropout(0.1)  ### 
        # 3x3 kernels, stride 1, padding 1
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2**(num_blocks-1)
        
        self.mlp = nn.Sequential(
            Flatten(), # [8, -1]로 flatten
            #nn.Dropout(0.3),
            nn.Linear(2*c*self.featuremap_size**2, 128), nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(128, 128), nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(128, 2*c*self.featuremap_size**2), nn.ReLU()
        )
        if out_chnls > 0:
            self.final_conv = nn.Conv2d(c, out_chnls, 1)
        else:
            self.final_conv = nn.Identity()
        self.out_chnls = out_chnls

    def forward(self, x): #이미지마다 1번씩
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        # Down
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            #act = self.dropout(act)  ##
            skip.append(act)
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True) #feature size 1/2
            x_down.append(act)

        
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size, self.featuremap_size)
        # Up
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1) #down 블럭의 역순으로 skip connection 
            x_up = block(features)
            #x_up = self.dropout(x_up)
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest', recompute_scale_factor=True)

        return self.final_conv(x_up)


#masking 모델
class OcclusionNetwork(nn.Module):#$
    def __init__(self, filter_start=32, in_chnls=3, N = 1, norm='gn'): #N= 4(6) -> 1 : 기존에는 각 마스크의 레이어에서 각기 다른 이미지 내 물체를 탐지했음 i)
        super(OcclusionNetwork, self).__init__()
        self.mask_encoder = UNet(
            num_blocks=6,  
            img_size=256,  
            filter_start=filter_start,  # mask_fbase
            in_chnls=in_chnls,
            out_chnls=-1,
            norm=norm
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, N, 1, 1, 0),  # input 채널과 output 채널(mask 수) 변경 필요
            nn.Sigmoid()
            #nn.Softmax(dim=1) #$
        )

    def forward(self, gray_batch):
        batch_feature = self.mask_encoder(gray_batch)
        out = self.mask_head(batch_feature)

        return out
