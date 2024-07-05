# -*- coding: utf-8 -*-
"""Main trainer for RoCoSDF.
"""

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.datasetRoCo import DatasetRoCo
from models.sdfSampler import SDFSampler

from models.model import RoCoSDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
import mcubes
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio

import skimage.measure
import plyfile

from models.discriminator import Discriminator

warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_name'] = self.conf['dataset.data_name']
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        self.base_exp_dir2 = self.conf['general.base_exp_dir'] + args.dir2
        os.makedirs(self.base_exp_dir2, exist_ok=True)
        
        if args.dir3 is not None:
            self.base_exp_dir3 = self.conf['general.base_exp_dir'] + args.dir3
            os.makedirs(self.base_exp_dir3, exist_ok=True)
            
        # Dataset
        self.dataset_roco = DatasetRoCo(self.conf['dataset'], args.dataname,args.gpu) # Current
        self.dataset_roco_2 = DatasetRoCo(self.conf['dataset2'], args.dataname2,args.gpu) # Another one

       
        self.dataname = args.dataname
        self.dataname2 = args.dataname2



        # for UNSR-ADL
        self.betas = (0.9, 0.999)

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        self.labmda_scc = self.conf.get_float('train.labmda_scc')
        self.labmda_adl = self.conf.get_float('train.labmda_adl')
        self.labmda_non_mfd = self.conf.get_float('train.labmda_non_mfd')
        self.labmda_mfd = self.conf.get_float('train.labmda_mfd')

        self.iter_step = 0
        self.iter_step2 = 0
        self.iter_step3 = 0
        self.mode = mode

        # Networks
        self.sdf_network = RoCoSDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)
        self.discriminator = Discriminator(**self.conf['model.discriminator']).to(self.device)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,betas=self.betas)

        self.sdf_network2 = RoCoSDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.optimizer2 = torch.optim.Adam(self.sdf_network2.parameters(), lr=self.learning_rate)
        self.discriminator2 = Discriminator(**self.conf['model.discriminator']).to(self.device)
        self.dis_optimizer2 = torch.optim.Adam(self.discriminator2.parameters(), lr=self.learning_rate,betas=self.betas)

        self.sdf_network3 = RoCoSDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.optimizer3 = torch.optim.Adam(self.sdf_network3.parameters(), lr=self.learning_rate)


    def train3(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file3 = os.path.join(os.path.join(self.base_exp_dir3), f'{timestamp}.log')
        logger3 = get_root_logger(log_file=log_file3, name='outs')
        print(log_file3)
        batch_size = self.batch_size


        loss_l1 = torch.nn.L1Loss(reduction="sum")

        res_step = self.maxiter - self.iter_step3
        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i,3)

            samples, samples_sdf = self.dataset_roco3.train_data(batch_size)
                
            samples.requires_grad = True
            pred_sdf = self.sdf_network3.sdf(samples)                      # 5000x1

            ########################## loss define ##############################

            loss_sdf = loss_l1(pred_sdf,samples_sdf)/samples.shape[0]
            loss_mfd = pred_sdf.abs().mean() 

            loss =  loss_sdf + loss_mfd*self.labmda_mfd


            self.optimizer3.zero_grad()
            loss.backward()
            self.optimizer3.step()
            

            self.iter_step3 += 1
            if self.iter_step3 % self.report_freq == 0:
                print_log('iter:{:8>d} loss = {} lr={}'.format(self.iter_step3, loss, self.optimizer3.param_groups[0]['lr']), logger=logger3)

            if self.iter_step3 % self.val_freq == 0 and self.iter_step3!=0: 
                self.reconstruct_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=None, iter_step=self.iter_step3, logger=logger3,netNumber=3)
            if self.iter_step3 % self.save_freq == 0 and self.iter_step3!=0: 
                self.save_checkpoint(netNumber=3)



    def train2(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file2 = os.path.join(os.path.join(self.base_exp_dir2), f'{timestamp}.log')
        print(log_file2)
        logger2 = get_root_logger(log_file=log_file2, name='outs2')
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step2

        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i,2)
            self.optimizer2.zero_grad()

            points, samples, point_gt = self.dataset_roco_2.train_data(batch_size)
                
            samples.requires_grad = True
            gradients_sample = self.sdf_network2.gradient(samples).squeeze() # 5000x3
            sdf_sample = self.sdf_network2.sdf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * sdf_sample                 # 5000x3


            ################## Loss Define ####################
            loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()

            consis_constraint = (1.0 - F.cosine_similarity(grad_norm, sample_moved-points, dim=1))
            
            R_non_mfd = torch.exp(-self.labmda_non_mfd * torch.abs(sdf_sample)).reshape(-1,consis_constraint.shape[-1]) 


            loss_SCC = consis_constraint * R_non_mfd


            G_loss = loss_sdf + loss_SCC.mean()*self.labmda_scc

            
            ############# Train ADL Discriminator #################
            self.dis_optimizer2.zero_grad()
            d_fake_output = self.discriminator2.sdf(sdf_sample.detach())
            d_fake_loss=self.get_discriminator_loss_single(d_fake_output,label=False)
            
            real_sdf = torch.zeros(points.size(0), 1).to(self.device)
            d_real_output = self.discriminator2.sdf(real_sdf)
            d_real_loss=self.get_discriminator_loss_single(d_real_output,label=True)
            dis_loss = d_real_loss + d_fake_loss
            dis_loss.backward()
            self.dis_optimizer2.step()


            ################ Total Loss ################
            d_fake_output = self.discriminator2.sdf(sdf_sample)
            gan_loss=self.get_generator_loss(d_fake_output)
            total_loss = gan_loss* self.labmda_adl + G_loss
            total_loss.backward()
            self.optimizer2.step()

            self.iter_step2 += 1
            if self.iter_step2 % self.report_freq == 0:
                print_log('iter:{:8>d} loss = {} lr={}'.format(self.iter_step2, loss_sdf, self.optimizer2.param_groups[0]['lr']), logger=logger2)

            if self.iter_step2 % self.val_freq == 0 and self.iter_step2!=0: 
                self.reconstruct_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger2,netNumber=2)

            if self.iter_step2 % self.save_freq == 0 and self.iter_step2!=0: 
                self.save_checkpoint(netNumber=2)

    def train1(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file1 = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger1 = get_root_logger(log_file=log_file1, name='outs1')
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step

        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i,1)

            points, samples, point_gt = self.dataset_roco.train_data(batch_size)
            self.optimizer.zero_grad()
  
            samples.requires_grad = True
            gradients_sample = self.sdf_network.gradient(samples).squeeze() # 5000x3
            sdf_sample = self.sdf_network.sdf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * sdf_sample                 # 5000x3

            ################## Loss Define ####################
            loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()

            consis_constraint = (1.0 - F.cosine_similarity(grad_norm, sample_moved-points, dim=1))
            
            R_non_mfd = torch.exp(-self.labmda_non_mfd * torch.abs(sdf_sample)).reshape(-1,consis_constraint.shape[-1]) 


            loss_SCC = consis_constraint * R_non_mfd


            G_loss = loss_sdf + loss_SCC.mean()*self.labmda_scc   

            ############# Train ADL Discriminator #################
            self.dis_optimizer.zero_grad()
            d_fake_output = self.discriminator.sdf(sdf_sample.detach())
            d_fake_loss=self.get_discriminator_loss_single(d_fake_output,label=False)
            
            real_sdf = torch.zeros(points.size(0), 1).to(self.device)
            d_real_output = self.discriminator.sdf(real_sdf)
            d_real_loss=self.get_discriminator_loss_single(d_real_output,label=True)
            dis_loss = d_real_loss + d_fake_loss
            dis_loss.backward()
            self.dis_optimizer.step()


            ################ Total Loss ################
            d_fake_output = self.discriminator.sdf(sdf_sample)
            gan_loss=self.get_generator_loss(d_fake_output)
            total_loss = gan_loss* self.labmda_adl + G_loss
            total_loss.backward()
            self.optimizer.step()
            
            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, G_loss, self.optimizer.param_groups[0]['lr']), logger=logger1)

            if self.iter_step % self.val_freq == 0 and self.iter_step!=0: 
                self.reconstruct_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger1, netNumber=1)

            if self.iter_step % self.save_freq == 0 and self.iter_step!=0: 
                self.save_checkpoint(netNumber=1)

    ############# ADL Loss ##############################
    def get_generator_loss(self,pred_fake):
        fake_loss=torch.mean((pred_fake-1)**2)
        return fake_loss
    def get_discriminator_loss_single(self,pred,label=True):
        if label==True:
            loss=torch.mean((pred-1)**2)
            return loss
        else:
            loss=torch.mean((pred)**2)
            return loss

    # create cube from (-1,-1,-1) to (1,1,1) and uniformly sample points for marching cube
    def create_cube(self,N):

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)
        
        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long().float() / N) % N
        samples[:, 0] = ((overall_index.long().float() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        samples.requires_grad = False

        return samples




    def reconstruct_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None,netNumber = None):

        if netNumber == 1:
            os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
            mesh = self.extract_geometry(resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))

            mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))
        elif netNumber == 2: 
            os.makedirs(os.path.join(self.base_exp_dir2, 'outputs'), exist_ok=True)
            mesh = self.extract_geometry(resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network2.sdf(pts))
            mesh.export(os.path.join(self.base_exp_dir2, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step2,str(threshold))))

        else:
            os.makedirs(os.path.join(self.base_exp_dir3, 'outputs'), exist_ok=True)
            mesh = self.extract_geometry(resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network3.sdf(pts))

            mesh.export(os.path.join(self.base_exp_dir3, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step3,str(threshold))))


    def query_function(self,netNumber=None):
        if netNumber == 1:
            query_func=lambda pts: self.sdf_network.sdf(pts)
        else:
            query_func=lambda pts: self.sdf_network2.sdf(pts)

        return query_func






    def update_learning_rate_np(self, iter_step,netNumber = 1):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        if netNumber == 1:
            for g in self.optimizer.param_groups:
                g['lr'] = lr
        elif netNumber == 2:
            for g in self.optimizer2.param_groups:
                g['lr'] = lr
        else:
            for g in self.optimizer3.param_groups:
                g['lr'] = lr


    def extract_fields_grad(self, bound_min, bound_max, resolution, query_func,grad_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)

        # with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    grad = grad_func(pts).reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                                        
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                    g[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = grad
                        
        return u,g

    def extract_fields(self, resolution, query_func):
        N = resolution
        max_batch = 1000000
        # the voxel_origin is the (bottom, left, down) corner, not the middle
        cube = self.create_cube(resolution).cuda()
        cube_points = cube.shape[0]

        with torch.no_grad():
            head = 0
            while head < cube_points:
                
                query = cube[head : min(head + max_batch, cube_points), 0:3]
                
                # inference defined in forward function per pytorch lightning convention
                pred_sdf = query_func(query)

                cube[head : min(head + max_batch, cube_points), 3] = pred_sdf.squeeze()
                    
                head += max_batch

        sdf_values = cube[:, 3]
        sdf_values = sdf_values.reshape(N, N, N).detach().cpu()


        return sdf_values

    def extract_geometry(self, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields( resolution, query_func).numpy()
        vertices, triangles = mcubes.marching_cubes(u, threshold)

        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (resolution - 1)

        vertices[:,0] = vertices[:,0]*voxel_size + voxel_origin[0]
        vertices[:,1] = vertices[:,1]*voxel_size + voxel_origin[0]
        vertices[:,2] = vertices[:,2]*voxel_size + voxel_origin[0]

        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh


    def reconstruct_mesh_CSG(self, resolution=64, threshold=0.0, intersactionSDF = None):

        bound_min = torch.tensor(self.dataset_roco.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_roco.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh = self.extract_geometry_CSG(bound_min, bound_max, resolution=resolution, threshold=threshold, u = intersactionSDF)

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}_intersactSDF.ply'.format(self.iter_step,str(threshold))))

    def extract_geometry_CSG(self, bound_min, bound_max, resolution, threshold, u):
        print('Creating mesh with threshold: {}'.format(threshold))
        vertices, triangles = mcubes.marching_cubes(u, threshold)

        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (resolution - 1)

        vertices[:,0] = vertices[:,0]*voxel_size + voxel_origin[0]
        vertices[:,1] = vertices[:,1]*voxel_size + voxel_origin[0]
        vertices[:,2] = vertices[:,2]*voxel_size + voxel_origin[0]

        mesh = trimesh.Trimesh(vertices, triangles)
     
    
        return mesh



    def load_checkpoint(self, checkpoint_name, dirNumber = 1):
        if dirNumber == 1:
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
            print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
            self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
            self.iter_step = checkpoint['iter_step']
        elif dirNumber == 2:
            checkpoint = torch.load(os.path.join(self.base_exp_dir2, 'checkpoints', checkpoint_name), map_location=self.device)
            print(os.path.join(self.base_exp_dir2, 'checkpoints', checkpoint_name))
            self.sdf_network2.load_state_dict(checkpoint['sdf_network_fine'])
            self.iter_step2 = checkpoint['iter_step']
        else:
            checkpoint = torch.load(os.path.join(self.base_exp_dir3, 'checkpoints', checkpoint_name), map_location=self.device)
            print(os.path.join(self.base_exp_dir3, 'checkpoints', checkpoint_name))
            self.sdf_network3.load_state_dict(checkpoint['sdf_network_fine'])
            self.iter_step3 = checkpoint['iter_step']
            
    def save_checkpoint(self,netNumber = 1):
        if netNumber == 1:
            checkpoint = {
                'sdf_network_fine': self.sdf_network.state_dict(),
                'iter_step': self.iter_step,
            }
            os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
            torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
        elif netNumber == 2:
            checkpoint = {
                'sdf_network_fine': self.sdf_network2.state_dict(),
                'iter_step': self.iter_step2,
            }
            os.makedirs(os.path.join(self.base_exp_dir2, 'checkpoints'), exist_ok=True)
            torch.save(checkpoint, os.path.join(self.base_exp_dir2, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step2)))
        else:
            checkpoint = {
                'sdf_network_fine': self.sdf_network3.state_dict(),
                'iter_step': self.iter_step3,
            }
            os.makedirs(os.path.join(self.base_exp_dir3, 'checkpoints'), exist_ok=True)
            torch.save(checkpoint, os.path.join(self.base_exp_dir3, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step3)))




if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/conf.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='T4')
    parser.add_argument('--dir2', type=str, default=None)
    parser.add_argument('--dir3', type=str, default=None)

    parser.add_argument('--dataname', type=str, default='T4')
    parser.add_argument('--dataname2', type=str, default=None)


    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    
    if args.mode == 'train':
        # Train Row-Column Scan

        print('Train fco')
        runner.train1() # Column
        print('Train fro')
        runner.train2() # Row

        
        # Load fco and fro
        runner.load_checkpoint('ckpt_010000.pth',dirNumber=1)
        qury_function1 = runner.query_function(netNumber=1)    # f_co

        runner.load_checkpoint('ckpt_010000.pth',dirNumber=2)
        qury_function2 = runner.query_function(netNumber=2)    # f_ro

        # CSG fusion and SDF Sampler
        runner.dataset_roco3 = SDFSampler(runner.conf['dataset'], args.dataname+'_sampler',qury_function1,qury_function2,args.gpu)
        print('Train roco')     
        # Optimize SDF  
        runner.train3() # Row

    elif args.mode == 'train_refine':
        # Load fco and fro
        runner.load_checkpoint('ckpt_010000.pth',dirNumber=1)
        qury_function1 = runner.query_function(netNumber=1)    # f_co

        runner.load_checkpoint('ckpt_010000.pth',dirNumber=2)
        qury_function2 = runner.query_function(netNumber=2)     # f_ro

        # CSG fusion and SDF Sampler
        runner.dataset_roco3 = SDFSampler(runner.conf['dataset'], args.dataname+'_sampler',qury_function1,qury_function2,args.gpu) 
        # Optimize SDF 
        runner.train3()
    