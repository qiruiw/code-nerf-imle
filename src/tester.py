

import numpy as np
import torch
import torch.nn as nn
import json
from data import SRN
from utils import get_rays, sample_from_rays, volume_rendering, image_float_to_uint8
from skimage.metrics import structural_similarity as compute_ssim
from model import CodeNeRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import math
import time
from tqdm import tqdm
import pdb


class Tester():
    def __init__(self, save_dir, gpu, jsonfile='srncar.json', batch_size=2048):
        super().__init__()
        # Read Hyperparameters
        hpampath = os.path.join('jsonfiles', jsonfile)
        with open(hpampath, 'r') as f:
            self.hpams = json.load(f)
        # self.save_dir = save_dir
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.make_dataloader(num_instances_per_obj=50, crop_img=False)
        self.load_model_codes(save_dir)
        # self.make_codes()
        self.B = batch_size
        # self.make_savedir(save_dir)
        # self.niter, self.nepoch = 0, 0
        # self.check_iter = check_iter
        self.psnr_eval = {}
        self.ssim_eval = {}

    def testing_epoch(self, num_instances_per_obj, crop_img=True):
        # self.make_dataloader(num_instances_per_obj, crop_img = crop_img)
        # self.set_optimizers()
        embdim = self.hpams['net_hyperparams']['code_dim']
        # per object
        for idx, d in enumerate(tqdm(self.dataloader)): # batch size 1
            if idx >= 300: break
            focal, H, W, imgs, poses, instances, obj_idx = d
            # if obj_idx > 4: break
            obj_idx = obj_idx.to(self.device)
            # per image
            shape_code, texture_code = self.shape_codes(obj_idx), self.texture_codes(obj_idx)
            
            ray_samples = []
            with torch.no_grad():
                instances = np.random.choice(50, num_instances_per_obj, replace=False)
                # instances = range(50)
                for k in instances:
                    rays_o, viewdir = get_rays(H.item(), W.item(), focal, poses[0, k])
                    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, self.hpams['near'], self.hpams['far'], self.hpams['N_samples'])
                    ray_samples.append((xyz, viewdir, z_vals))
            
                # for k in instances:
                    # print(k, num_instances_per_obj, poses[0, k].shape, imgs.shape, 'k')
                    # t1 = time.time()
                    # xyz, viewdir, z_vals = ray_samples[k]
                    loss_per_img, generated_img = [], []
                    
                    z_shape = torch.randn(1, embdim).cuda()
                    z_txt = torch.randn(1, embdim).cuda()
                    # pdb.set_trace()
                    sigmas, rgbs = self.model(xyz.to(self.device),
                                                viewdir.to(self.device),
                                                shape_code, 
                                                texture_code,
                                                z_shape,
                                                z_txt)
                    rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                    
                    loss_l2 = torch.mean((rgb_rays - imgs[0, k].type_as(rgb_rays))**2)
                    loss_per_img.append(loss_l2.item())
                    generated_img.append(rgb_rays)
                    
                    self.log_eval_psnr(np.mean(loss_per_img), k, obj_idx.item())
                    self.log_compute_ssim(torch.cat(generated_img).reshape(H, W, 3), imgs[0, k].reshape(H, W, 3), k, obj_idx.item())
                    
        # pdb.set_trace()
        self.save_opts()

    
    def log_eval_psnr(self, loss_per_img, niters, obj_idx):
        psnr = -10 * np.log(loss_per_img) / np.log(10)
        if obj_idx not in self.psnr_eval:
            self.psnr_eval[obj_idx] = [psnr]
        else:
            self.psnr_eval[obj_idx].append(psnr)
            
    def log_compute_ssim(self, generated_img, gt_img, niters, obj_idx):
        generated_img_np = generated_img.detach().cpu().numpy()
        gt_img_np = gt_img.detach().cpu().numpy()
        ssim = compute_ssim(generated_img_np, gt_img_np, multichannel=True)
        if obj_idx not in self.ssim_eval:
            self.ssim_eval[obj_idx] = [ssim]
        else:
            self.ssim_eval[obj_idx].append(ssim)
    
    def save_opts(self):
        saved_dict = {
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval
        }
        with open(os.path.join(self.save_dir, 'eval.json'), 'w') as f:
            json.dump(saved_dict, f)
        # torch.save(saved_dict, os.path.join(self.save_dir, 'codes.pth'))
        # print('We finished the optimization of ' + str(num_obj))
        mean_psnr = [np.mean(psnr) for psnr in self.psnr_eval.values()]
        mean_ssim = [np.mean(ssim) for ssim in self.ssim_eval.values()]
        print(f'avg psnr: {np.mean(mean_psnr)}\navg ssim: {np.mean(mean_ssim)}')

    # def set_optimizers(self):
    #     lr1, lr2 = self.get_learning_rate()
    #     self.opts = torch.optim.AdamW([
    #         {'params':self.model.parameters(), 'lr': lr1},
    #         {'params':self.shape_codes.parameters(), 'lr': lr2},
    #         {'params':self.texture_codes.parameters(), 'lr':lr2}
    #     ])

    # def get_learning_rate(self):
    #     model_lr, latent_lr = self.hpams['lr_schedule'][0], self.hpams['lr_schedule'][1]
    #     num_model = self.niter // model_lr['interval']
    #     num_latent = self.niter // latent_lr['interval']
    #     lr1 = model_lr['lr'] * 2**(-num_model)
    #     lr2 = latent_lr['lr'] * 2**(-num_latent)
    #     return lr1, lr2

    def make_model(self):
        self.model = CodeNeRF(**self.hpams['net_hyperparams']).to(self.device)

    # def make_codes(self):
    #     embdim = self.hpams['net_hyperparams']['code_dim']
    #     d = len(self.dataloader)
    #     self.shape_codes = nn.Embedding(d, embdim)
    #     self.texture_codes = nn.Embedding(d, embdim)
    #     self.shape_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
    #     self.texture_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
    #     self.shape_codes = self.shape_codes.to(self.device)
    #     self.texture_codes = self.texture_codes.to(self.device)
        
    def make_dataloader(self, num_instances_per_obj, crop_img):
        # cat : whether it is 'srn_cars' or 'srn_chairs'
        # split: whether it is 'car_train', 'car_test' or 'car_val'
        # data_dir : the root directory of ShapeNet_SRN
        # num_instances_per_obj : how many images we chosose from objects
        cat = self.hpams['data']['cat']
        data_dir = self.hpams['data']['data_dir']
        splits = self.hpams['data']['splits']
        srn = SRN(cat=cat, splits=splits, data_dir = data_dir,
                  num_instances_per_obj = num_instances_per_obj, crop_img = crop_img)
        self.dataloader = DataLoader(srn, batch_size=1, num_workers=4)

    # def make_savedir(self, save_dir):
    #     self.save_dir = os.path.join('exps', save_dir)
    #     if not os.path.isdir(self.save_dir):
    #         os.makedirs(os.path.join(self.save_dir, 'runs'))
    #     self.writer = SummaryWriter(os.path.join(self.save_dir, 'runs'))
    #     hpampath = os.path.join(self.save_dir, 'hpam.json')
    #     with open(hpampath, 'w') as f:
    #         json.dump(self.hpams, f, indent=2)

    def load_model_codes(self, saved_dir):
        saved_path = os.path.join('exps', saved_dir, 'models.pth')
        saved_data = torch.load(saved_path, map_location = torch.device('cpu'))
        self.make_save_img_dir(os.path.join('exps', saved_dir, 'test'))
        self.make_writer(saved_dir)
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)
        embdim = self.hpams['net_hyperparams']['code_dim']
        d = len(self.dataloader)
        self.shape_codes = nn.Embedding(d, embdim).to(self.device)
        self.texture_codes = nn.Embedding(d, embdim).to(self.device)
        self.shape_codes.load_state_dict(saved_data['shape_code_params'])#.to(self.device)
        self.texture_codes.load_state_dict(saved_data['texture_code_params'])#.to(self.device)
        # self.shape_codes = saved_data['shape_code_params']['weight'].to(self.device)
        # self.texture_codes = saved_data['texture_code_params']['weight'].to(self.device)

    def make_writer(self, saved_dir):
        self.writer = SummaryWriter(os.path.join('exps', saved_dir, 'test', 'runs'))

    def make_save_img_dir(self, save_dir):
        save_dir_tmp = save_dir
        num = 2
        while os.path.isdir(save_dir_tmp):
            save_dir_tmp = save_dir + '_' + str(num)
            num += 1
        os.makedirs(save_dir_tmp)
        self.save_dir = save_dir_tmp