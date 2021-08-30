from model_boundary import model_fn_decorator
from model_boundary import SemanticPrediction as Network
from dataset import ABCDataset

import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np

from util.config import cfg
from util.log import logger
import util.utils as utils
import glob

from lib.pointgroup_ops.functions import pointgroup_ops
from ctypes import *
from scipy.spatial import cKDTree
import glob
import numpy as np
import yaml
from multiprocessing import Pool
import os
from ctypes import *

Regiongrow = cdll.LoadLibrary('./cpp/build/libregiongrow.so')

files = sorted([cfg.root_dir + '/val/' + p for p in os.listdir(cfg.root_dir + '/val/') if p[-3:] == 'npz'])

def init():
    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

def crop(xyz):
    '''
    :param xyz: (n, 3) >= 0
    '''
    xyz_offset = xyz.copy()

    offset = (xyz_offset.max(0) - np.array([512] * 3)) * np.array([0.5,0.5,0.5])
    xyz_offset = xyz_offset - offset

    valid_idxs = (xyz_offset.min(1) >= 0) * (xyz_offset.max(1) < 512 - 1)

    return xyz_offset, valid_idxs

def Parse(iii, model, model_fn, start_epoch):
    global files
    locs = []
    locs_indices = []
    locs_float = []
    locs_float_gt = []
    normals = []
    normals_gt = []
    boundaries = []
    edge_indices = []
    semantics_gt = []

    fn = files[iii]
    #if len(fn.split('taihedianguangchang')) <= 1:
    #    return
    data = np.load(files[iii])

    #xyz_origin, normal, boundary = data['V'], data['N'], data['B']
    xyz_origin, normal, boundary, F, SF = data['V'], data['N'], data['B'], data['F'], data['S']
    semantics = np.zeros((xyz_origin.shape[0]), dtype='int32')
    semantics[F[:,0]] = SF
    semantics[F[:,1]] = SF
    semantics[F[:,2]] = SF

    original_indices = np.arange(xyz_origin.shape[0])
    xyz_middle, normal_middle = xyz_origin, normal

    xyz_middle_noise = xyz_middle# + (np.random.rand(xyz_middle.shape[0], xyz_middle.shape[1]) * 2 - 1) * cfg.gen_noise
    normal_middle_noise = normal_middle
    xyz_middle -= xyz_middle_noise.min(0)
    xyz_middle_noise -= xyz_middle_noise.min(0)

    ### scale
    xyz = xyz_middle_noise * cfg.scale

    xyz, valid_idxs = crop(xyz)

    sampling_map = np.zeros((xyz_origin.shape[0]),dtype='int32')
    final_indices = np.arange(original_indices.shape[0])
    sampling_map[original_indices[:]] = final_indices

    tree = cKDTree(xyz_middle)
    d, ii = tree.query(xyz_middle, k=16, n_jobs=16)
    locs_indices.append(torch.from_numpy(ii))

    locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), torch.from_numpy(xyz).long()], 1))
    locs_float.append(torch.from_numpy(xyz_middle_noise))
    locs_float_gt.append(torch.from_numpy(xyz_middle))
    normals.append(torch.from_numpy(normal_middle_noise))
    normals_gt.append(torch.from_numpy(normal_middle))
    semantics_gt.append(torch.from_numpy(semantics))

    # get valid edges
    mask = np.zeros(xyz_origin.shape[0], dtype='int32')
    mask[original_indices] = 1
    v1 = np.concatenate([F[:,0:1], F[:,1:2], F[:,2:3]])
    v2 = np.concatenate([F[:,1:2], F[:,2:3], F[:,0:1]])
    vmask = ((mask[v1] + mask[v2]) == 2)
    v1 = v1[vmask]
    v2 = v2[vmask]
    edge_idx = np.concatenate([v1.reshape(-1,1), v2.reshape(-1,1)], axis=1)
    edge_idx = sampling_map[edge_idx]
    edge_boundary = (np.sum(boundary[edge_idx], axis=1) > 0).astype('int64')

    edge_indices.append(torch.from_numpy(edge_idx))
    boundaries.append(torch.from_numpy(edge_boundary))


    locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
    locs_float_gt = torch.cat(locs_float_gt, 0).to(torch.float32)  # float (N, 3)
    locs_indices = torch.cat(locs_indices, 0).long()
    normals = torch.cat(normals, 0).to(torch.float32)                              # float (N, C)
    normals_gt = torch.cat(normals_gt, 0).to(torch.float32)
    semantics_gt = torch.cat(semantics_gt, 0).long()
    boundaries = torch.cat(boundaries, 0).long()                     # long (N)
    edge_indices = torch.cat(edge_indices, 0).long()

    max_dim = locs.max(0)[0][1:].numpy()
    min_dim = locs.min(0)[0][1:].numpy()
    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), 512, None)     # long (3)

    ### voxelize
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, 1, cfg.mode)
    if np.min(min_dim) < 0 or np.max(max_dim) > 511:
        return
    batch = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        'locs_float': locs_float, 'locs_indices': locs_indices, 'locs_float_gt': locs_float_gt,
        'normals': normals, 'normals_gt': normals_gt, 'semantics_gt':semantics_gt, 'boundaries': boundaries, 'edge_indices': edge_indices,
        'id': id, 'spatial_shape': spatial_shape}

    prediction = model_fn(batch, model, start_epoch)

    V = xyz_middle_noise

    pb = (prediction['b'][:,1] > 0.5).data.cpu().numpy().astype('int32')
    pp = np.argmax(prediction['p'].data.cpu().numpy(), axis=1)

    edges = edge_indices.data.cpu().numpy().astype('int32')
    intensity = prediction['b'][:,1].data.cpu().numpy() * 0.99
  
    face_labels = np.zeros((F.shape[0]), dtype='int32')
    masks = np.zeros((V.shape[0]), dtype='int32')
    pb = (prediction['b'][:,1]>prediction['b'][:,0]).data.cpu().numpy().astype('int32')
    Regiongrow.RegionGrowing(c_void_p(pb.ctypes.data), c_void_p(F.ctypes.data),
        V.shape[0], F.shape[0], c_void_p(face_labels.ctypes.data), c_void_p(masks.ctypes.data),
        c_float(0.99))

    pb = boundaries.data.cpu().numpy().astype('int32')
    gt_face_labels = np.zeros((F.shape[0]), dtype='int32')
    gt_masks = np.zeros((V.shape[0]), dtype='int32')
                
    Regiongrow.RegionGrowing(c_void_p(pb.ctypes.data), c_void_p(F.ctypes.data),
        V.shape[0], F.shape[0], c_void_p(gt_face_labels.ctypes.data), c_void_p(gt_masks.ctypes.data),
        c_float(0.99))

    semantic_faces = semantics[F[:,0]]
    semantic_faces_gt = semantics_gt.data.cpu().numpy()[F[:,0]]
    np.savez_compressed('results/predictions/%s'%(fn.split('/')[-1]), V=V,F=F,L=face_labels,L_gt=gt_face_labels, S=semantic_faces, S_gt=semantic_faces_gt)
    
    colors = np.random.rand(10000, 3)
    VC = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0

    fp = open('results/visualize/%s.obj'%(fn.split('/')[-1][:-4]), 'w')
    for i in range(VC.shape[0]):
        v = VC[i]
        if face_labels[i] < 0:
            p = np.array([0,0,0])
        else:
            p = colors[face_labels[i]]
        fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
    fp.close()

if __name__ == '__main__':
    ##### init
    init()

    #if not os.path.exists('eval-%f'%(cfg.gen_noise)):
    #    os.mkdir('eval-%f'%(cfg.gen_noise))
    ##### model
    logger.info('=> creating model ...')

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda

    model = model.cuda()
    ##### model_fn (criterion)
    model_fn = model_fn_decorator(True)

    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, 0, False, cfg.pretrain)

    model.eval()
    for i in range(0,len(files)):
        Parse(i, model, model_fn, start_epoch)