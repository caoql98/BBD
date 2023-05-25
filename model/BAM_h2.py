import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2
import os
import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from util.util import get_train_val_set


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq/2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        
        assert self.layers in [50, 101, 152]
    
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet'+str(args.layers)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)               
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try: 
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512           
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))

        # FUSE Ensemble
        self.prior_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.prior_merge.weight = nn.Parameter(torch.tensor([[0.5],[0.5]]).reshape_as(self.prior_merge.weight))
        self.p1 = nn.Parameter(torch.tensor([0.5]))
        self.p2 = nn.Parameter(torch.tensor([0.5]))
        self.a1 = nn.Parameter(torch.tensor([0.5]))
        self.a2 = nn.Parameter(torch.tensor([0.5]))

        # H Ensemble
        self.h_merge1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.h_merge1.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.h_merge1.weight))
        self.h_merge2 = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.h_merge2.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.h_merge2.weight))

        # self.h_merge = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        # self.h_merge.weight = nn.Parameter(torch.tensor([[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]]).reshape_as(self.h_merge.weight))
        
        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},        
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},
                {'params': model.kshot_rw.parameters()},
                {'params': model.h_merge1.parameters()},          
                {'params': model.h_merge2.parameters()},        
                {'params': model.prior_merge.parameters()},      
                {'params': model.p1},      
                {'params': model.p2},      
                {'params': model.a1},      
                {'params': model.a2},                          
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},        
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},      
                {'params': model.h_merge1.parameters()},          
                {'params': model.h_merge2.parameters()},      
                {'params': model.prior_merge.parameters()},      
                {'params': model.p1},      
                {'params': model.p2},      
                {'params': model.a1},      
                {'params': model.a2},              
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def prior_mask_g(self,final_supp_list,mask_list,query_feat_4,query_feat_3,weight_soft):
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1,True)
        return  corr_query_mask

    def K_Shot_Reweighting(self,que_gram,supp_feat_list,bs):
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1,2))/norm_max).reshape(bs,1,1,1)) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1,True) # [bs, 1, 1, 1]  

        return weight_soft, est_val

    def unnormalize(self,img):
        img = img.clone()
        mean_img = [0.485*255, 0.456*255, 0.406*255]
        std_img = [0.229*255, 0.224*255, 0.225*255]
        for im_channel, mean, std in zip(img, mean_img, std_img):
            im_channel.mul_(std).add_(mean)
        return img
        
    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    # def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None):
    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None,num = None):
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x_h = torch.flip(x, dims=[-1])	
        y_m_h  = torch.flip(y_m, dims=[-1])
        y_b_h  = torch.flip(y_m, dims=[-1])

        s_x_h = torch.flip(s_x, dims=[-1])	
        s_y_h = torch.flip(s_y, dims=[-1])
        
        # name = str(num)+'.jpg'

        # vis_path1 = '/output/query_img'
        # if not os.path.exists(vis_path1): os.makedirs(vis_path1)        
        # query_img = self.unnormalize(x[0,:].detach())
        # query_img = np.uint8(np.transpose(query_img.cpu().numpy(), (1, 2, 0)))
        # query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
        # query_img_name = os.path.join(vis_path1,name)
        # cv2.imwrite(query_img_name,query_img)

        # vis_path11 = '/output/query_mask'
        # if not os.path.exists(vis_path11): os.makedirs(vis_path11)    
        # query_mask_name = os.path.join(vis_path11,name)
        # y_m1 = y_m.clone()
        # y_m1[y_m1==255]=0
        # y_m1[y_m1==1]=255
        # cv2.imwrite(query_mask_name,y_m1[0,:].detach().cpu().numpy())
        

        # vis_path2 = '/output/support_img'
        # if not os.path.exists(vis_path2): os.makedirs(vis_path2)        
        # support_img = self.unnormalize(s_x[0,0,:].detach())
        # support_img = np.uint8(np.transpose(support_img.cpu().numpy(), (1, 2, 0)))
        # support_img = cv2.cvtColor(support_img, cv2.COLOR_RGB2BGR)
        # support_img_name = os.path.join(vis_path2,name)
        # cv2.imwrite(support_img_name,support_img)

        # vis_path21 = '/output/support_mask'
        # if not os.path.exists(vis_path21): os.makedirs(vis_path21)    
        # support_mask_name = os.path.join(vis_path21,name)
        # s_y1 = s_y.clone()
        # s_y1[s_y1==255]=0
        # s_y1[s_y1==1]=255
        # cv2.imwrite(support_mask_name,s_y1[0,0,:].detach().cpu().numpy())


        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        # Query Feature_h
        with torch.no_grad():
            query_feath_0 = self.layer0(x_h)
            query_feath_1 = self.layer1(query_feath_0)
            query_feath_2 = self.layer2(query_feath_1)
            query_feath_3 = self.layer3(query_feath_2)
            query_feath_4 = self.layer4(query_feath_3)
            if self.vgg:
                query_feath_2 = F.interpolate(query_feath_2, size=(query_feath_3.size(2),query_feath_3.size(3)), mode='bilinear', align_corners=True)

        query_feath = torch.cat([query_feath_3, query_feath_2], 1)
        query_feath = self.down_query(query_feath)

        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = [] 
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_pro)
            supp_feat_list.append(eval('supp_feat_' + self.low_fea_id))

        # Support h_Feature
        supp_pro_hlist = []
        final_supp_hlist = []
        mask_hlist = []
        supp_feat_hlist = [] 
        for i in range(self.shot):
            maskh = (s_y_h[:,i,:,:] == 1).float().unsqueeze(1)
            mask_hlist.append(maskh)
            with torch.no_grad():
                supp_feath_0 = self.layer0(s_x_h[:,i,:,:,:])
                supp_feath_1 = self.layer1(supp_feath_0)
                supp_feath_2 = self.layer2(supp_feath_1)
                supp_feath_3 = self.layer3(supp_feath_2)
                maskh = F.interpolate(maskh, size=(supp_feath_3.size(2), supp_feath_3.size(3)), mode='bilinear', align_corners=True)
                supp_feath_4 = self.layer4(supp_feath_3*maskh)
                final_supp_hlist.append(supp_feath_4)
                if self.vgg:
                    supp_feath_2 = F.interpolate(supp_feath_2, size=(supp_feath_3.size(2),supp_feath_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feath = torch.cat([supp_feath_3, supp_feath_2], 1)
            supp_feath = self.down_supp(supp_feath)
            supp_proh = Weighted_GAP(supp_feath, maskh)
            supp_pro_hlist.append(supp_proh)
            supp_feat_hlist.append(eval('supp_feath_' + self.low_fea_id))

        # K-Shot Reweighting
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        weight_soft ,est_val = self.K_Shot_Reweighting(que_gram,supp_feat_list,bs)         

        # K-Shot Reweighting_h
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        weight_hsoft ,est_hval = self.K_Shot_Reweighting(que_gram,supp_feat_hlist,bs)     

        # K-Shot Reweighting queryh
        que_gram = get_gram_matrix(eval('query_feath_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        weight_soft1 ,est_val1 = self.K_Shot_Reweighting(que_gram,supp_feat_list,bs)         
  
        # K-Shot Reweighting_h queryh
        que_gram = get_gram_matrix(eval('query_feath_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        weight_hsoft1 ,est_hval1 = self.K_Shot_Reweighting(que_gram,supp_feat_hlist,bs)   


        # Prior Similarity Mask
        corr_query_mask = self.prior_mask_g(final_supp_list,mask_list,query_feat_4,query_feat_3,weight_soft)

        # Prior Similarity Mask_h
        corr_query_hmask = self.prior_mask_g(final_supp_hlist,mask_hlist,query_feat_4,query_feat_3,weight_hsoft)

        # Prior Similarity Mask queryh
        corr_query_mask1 = self.prior_mask_g(final_supp_list,mask_list,query_feath_4,query_feath_3,weight_soft1)

        # Prior Similarity Mask_h queryh
        corr_query_hmask1 = self.prior_mask_g(final_supp_hlist,mask_hlist,query_feath_4,query_feath_3,weight_hsoft1)

        # vis_path31 = '/output/corr_query_mask'
        # if not os.path.exists(vis_path31): os.makedirs(vis_path31)    
        # priormaskname1 = os.path.join(vis_path31,name)
        # weightmap = corr_query_mask[0,0,:].detach().cpu().numpy()
        # weightmap = (weightmap - np.min(weightmap)) / (np.max(weightmap) - np.min(weightmap))
        # weightmap = np.uint8(255 * weightmap)
        # cv2.imwrite(priormaskname1,weightmap)

        # vis_path32 = '/output/corr_query_hmask'
        # if not os.path.exists(vis_path32): os.makedirs(vis_path32)    
        # priormaskname2 = os.path.join(vis_path32,name)
        # weightmap = corr_query_hmask[0,0,:].detach().cpu().numpy()
        # weightmap = (weightmap - np.min(weightmap)) / (np.max(weightmap) - np.min(weightmap))
        # weightmap = np.uint8(255 * weightmap)
        # cv2.imwrite(priormaskname2,weightmap)

        # vis_path33 = '/output/corr_query_mask1'
        # if not os.path.exists(vis_path33): os.makedirs(vis_path33)    
        # priormaskname3 = os.path.join(vis_path33,name)
        # weightmap = corr_query_mask1[0,0,:].detach().cpu().numpy()
        # weightmap = (weightmap - np.min(weightmap)) / (np.max(weightmap) - np.min(weightmap))
        # weightmap = np.uint8(255 * weightmap)
        # cv2.imwrite(priormaskname3,weightmap)

        # vis_path34 = '/output/corr_query_hmask1'
        # if not os.path.exists(vis_path34): os.makedirs(vis_path34)    
        # priormaskname34 = os.path.join(vis_path34,name)
        # weightmap = corr_query_hmask1[0,0,:].detach().cpu().numpy()
        # weightmap = (weightmap - np.min(weightmap)) / (np.max(weightmap) - np.min(weightmap))
        # weightmap = np.uint8(255 * weightmap)
        # cv2.imwrite(priormaskname34,weightmap)

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro = (weight_soft.permute(0,2,1,3) * supp_pro).sum(2,True)
        supp_pro1 = (weight_soft1.permute(0,2,1,3) * supp_pro).sum(2,True)

        # Support Prototype_h 
        supp_proh = torch.cat(supp_pro_hlist, 2)  # [bs, 256, shot, 1]
        supp_proh = (weight_hsoft.permute(0,2,1,3) * supp_proh).sum(2,True)
        supp_proh1 = (weight_hsoft1.permute(0,2,1,3) * supp_proh).sum(2,True)

        supp_pro = self.p1*supp_pro+self.p2*supp_proh
        supp_pro1 = self.p1*supp_pro1+self.p2*supp_proh1
        corr_query_mask = self.prior_merge(torch.cat([corr_query_mask,corr_query_hmask], dim=1))
        corr_query_mask1 = self.prior_merge(torch.cat([corr_query_mask1,corr_query_hmask1], dim=1))
        est_val = self.a1*est_val+self.a2*est_hval
        est_val1 = self.a1*est_val1+self.a2*est_hval1

        # supp_pro = 0.5*supp_pro+0.5*supp_proh
        # supp_pro1 = 0.5*supp_pro1+0.5*supp_proh1
        # corr_query_mask = 0.5*corr_query_mask+0.5*corr_query_hmask
        # corr_query_mask1 = 0.5*corr_query_mask1+0.5*corr_query_hmask1
        # est_val = 0.5*est_val+0.5*est_hval
        # est_val1 = 0.5*est_val1+0.5*est_hval1

        # vis_path41 = '/output/corr_query_maskf'
        # if not os.path.exists(vis_path41): os.makedirs(vis_path41)    
        # priormasknamef = os.path.join(vis_path41,name)
        # weightmap = corr_query_mask[0,0,:].detach().cpu().numpy()
        # weightmap = (weightmap - np.min(weightmap)) / (np.max(weightmap) - np.min(weightmap))
        # weightmap = np.uint8(255 * weightmap)
        # cv2.imwrite(priormasknamef,weightmap)

        # vis_path42 = '/output/corr_query_maskf1'
        # if not os.path.exists(vis_path42): os.makedirs(vis_path42)    
        # priormasknamef1 = os.path.join(vis_path42,name)
        # weightmap = corr_query_mask1[0,0,:].detach().cpu().numpy()
        # weightmap = (weightmap - np.min(weightmap)) / (np.max(weightmap) - np.min(weightmap))
        # weightmap = np.uint8(255 * weightmap)
        # cv2.imwrite(priormasknamef1,weightmap)
        
        # Tile & Cat
        concat_feat = supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask], 1)   # 256+256+1
        merge_feat = self.init_merge(merge_feat)

        # Tile & Cat_h
        concat_feath = supp_pro1.expand_as(query_feath)
        merge_feath = torch.cat([query_feath, concat_feath, corr_query_mask1], 1)   # 256+256+1
        merge_feath = self.init_merge(merge_feath)

        # Base and Meta
        base_out = self.learner_base(query_feat_4)

        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)   # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta 
        meta_out = self.cls_meta(query_meta)
        
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Base and Meta_h
        base_hout = self.learner_base(query_feath_4)

        query_hmeta = self.ASPP_meta(merge_feath)
        query_hmeta = self.res1_meta(query_hmeta)   # 1080->256
        query_hmeta = self.res2_meta(query_hmeta) + query_hmeta 
        meta_hout = self.cls_meta(query_hmeta)
        
        meta_out_hsoft = meta_hout.softmax(1)
        base_out_hsoft = base_hout.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:,0:1,:,:]                           # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:,1:,:,:]                            # [bs, 1, 60, 60]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes+1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)
                base_map_list.append(base_out_soft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
            base_map = torch.cat(base_map_list,0)
            # <alternative implementation>
            # gather_id = (cat_idx[0]+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
            # fg_map = base_out_soft.gather(1,gather_id)
            # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map            
        else:
            base_map = base_out_soft[:,1:,:,:].sum(1,True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)



        # Classifier Ensemble_h
        meta_map_bgh = meta_out_hsoft[:,0:1,:,:]                           # [bs, 1, 60, 60]
        meta_map_fgh = meta_out_hsoft[:,1:,:,:]                            # [bs, 1, 60, 60]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes+1, device='cuda')
            base_map_hlist = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)
                base_map_hlist.append(base_out_hsoft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
            base_hmap = torch.cat(base_map_hlist,0)
            # <alternative implementation>
            # gather_id = (cat_idx[0]+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
            # fg_map = base_out_soft.gather(1,gather_id)
            # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map            
        else:
            base_hmap = base_out_hsoft[:,1:,:,:].sum(1,True)

        est_hmap = est_val1.expand_as(meta_map_fgh)

        meta_map_bgh = self.gram_merge(torch.cat([meta_map_bgh,est_hmap], dim=1))
        meta_map_fgh = self.gram_merge(torch.cat([meta_map_fgh,est_hmap], dim=1))

        merge_hmap = torch.cat([meta_map_bgh, base_hmap], 1)
        merge_bgh = self.cls_merge(merge_hmap)                     # [bs, 1, 60, 60]

        final_hout = torch.cat([merge_bgh, meta_map_fgh], dim=1)

        # Output merge
        final_houth = torch.flip(final_hout, dims=[-1])	

        # query_merge_hout = 0.5*final_out +0.5*final_houth

        query_merge_houtb = self.h_merge1(torch.cat([final_out[:,0:1,:,:] , final_houth[:,0:1,:,:] ], 1))
        query_merge_houtf = self.h_merge2(torch.cat([final_out[:,1:,:,:] , final_houth[:,1:,:,:] ], 1))
        query_merge_hout = torch.cat([query_merge_houtb, query_merge_houtf], 1)
 

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

            meta_hout = F.interpolate(meta_hout, size=(h, w), mode='bilinear', align_corners=True)
            base_hout = F.interpolate(base_hout, size=(h, w), mode='bilinear', align_corners=True)
            final_hout = F.interpolate(final_hout, size=(h, w), mode='bilinear', align_corners=True)

            query_merge_hout = F.interpolate(query_merge_hout, size=(h, w), mode='bilinear', align_corners=True)

        # vis_path51 = '/output/final_out'
        # if not os.path.exists(vis_path51): os.makedirs(vis_path51)    
        # final_outname = os.path.join(vis_path51,name)
        # cv2.imwrite(final_outname,final_out.max(1)[1][0,:].detach().cpu().numpy()*255)

        # vis_path52 = '/output/final_hout'
        # if not os.path.exists(vis_path52): os.makedirs(vis_path52)    
        # final_houtname = os.path.join(vis_path52,name)
        # cv2.imwrite(final_houtname,final_hout.max(1)[1][0,:].detach().cpu().numpy()*255)

        # vis_path53 = '/output/query_merge_hout'
        # if not os.path.exists(vis_path53): os.makedirs(vis_path53)    
        # final_houtnamef = os.path.join(vis_path53,name)
        # cv2.imwrite(final_houtnamef,query_merge_hout.max(1)[1][0,:].detach().cpu().numpy()*255)

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            main_hloss = self.criterion(final_hout, y_m_h.long())

            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_lossh1 = self.criterion(meta_hout, y_m_h.long())

            aux_loss2 = self.criterion(base_out, y_b.long())
            aux_lossh2 = self.criterion(base_hout, y_b_h.long())

            main_all_loss = self.criterion(query_merge_hout, y_m.long())
            
            main_loss =0.5*main_loss+0.5*main_hloss+main_all_loss
            aux_loss1 = 0.5*aux_loss1 + 0.5*aux_lossh1
            aux_loss2 = 0.5*aux_loss2 + 0.5*aux_lossh2

            # main_loss = main_loss+main_hloss+main_all_loss
            # aux_loss1 = aux_loss1 + aux_lossh1
            # aux_loss2 = aux_loss2 + aux_lossh2

            # return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2
            return query_merge_hout.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            # return final_out, meta_out, base_out
            return query_merge_hout, 0.5*meta_out+0.5*meta_hout, 0.5*base_out+0.5*base_hout

