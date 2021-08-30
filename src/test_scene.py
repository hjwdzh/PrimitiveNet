import glob
import numpy as np

from model_boundary import model_fn_decorator
from model_boundary import SemanticPrediction as Network

import torch
import torch.optim as optim
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
import os

Regiongrow = cdll.LoadLibrary('./cpp/build/libregiongrow.so')

dataset = 'cyberverse'

token = ['']
verts = 0

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

def SaveOBJ(fn, V, F, C):
	fp = open(fn, 'w')
	for i in range(V.shape[0]):
		v = V[i]
		c = C[i]
		fp.write("v %f %f %f %d %d %d\n"%(v[0],v[1],v[2],c[0],c[1],c[2]))
	for i in range(F.shape[0]):
		f = F[i]
		fp.write('f %d %d %d\n'%(f[0]+1, f[1]+1, f[2]+1))
	fp.close()

def AverageSplit(V, indices, groups, axis):
	if indices.shape[0] <= 1024 * 128:
		groups.append(indices)
		return
	vtemp = V[indices]
	m = np.median(vtemp[:,axis])
	idx1 = np.where(vtemp[:,axis] < m)[0]
	idx2 = np.where(vtemp[:,axis] >= m)[0]
	AverageSplit(V, indices[idx1], groups, 1 - axis)
	AverageSplit(V, indices[idx2], groups, 1 - axis)

def Predict(xyz_origin, normal, F):
    locs = []
    locs_indices = []
    locs_float = []
    locs_float_gt = []
    normals = []
    normals_gt = []
    edge_indices = []

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

    edge_indices.append(torch.from_numpy(edge_idx))


    locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
    locs_float_gt = torch.cat(locs_float_gt, 0).to(torch.float32)  # float (N, 3)
    locs_indices = torch.cat(locs_indices, 0).long()
    normals = torch.cat(normals, 0).to(torch.float32)                              # float (N, C)
    normals_gt = torch.cat(normals_gt, 0).to(torch.float32)
    edge_indices = torch.cat(edge_indices, 0).long()

    max_dim = locs.max(0)[0][1:].numpy()
    min_dim = locs.min(0)[0][1:].numpy()
    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), 512, None)     # long (3)

    ### voxelize
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, 1, cfg.mode)

    batch = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        'locs_float': locs_float, 'locs_indices': locs_indices, 'locs_float_gt': locs_float_gt,
        'normals': normals, 'normals_gt': normals_gt, 'edge_indices': edge_indices,
        'id': id, 'spatial_shape': spatial_shape}

    prediction = model_fn(batch, model, start_epoch)
    return prediction['b'][:,1].data.cpu().numpy().astype('float32'),\
    	edge_indices.numpy()


def Parse(ii):
	global b_files, fn_files, f_files, i_files, n_files, vc_files, v_files
	print(v_files[ii])
	V = np.fromfile(v_files[ii], dtype='float64').reshape((-1, 3))
	VC = np.fromfile(vc_files[ii], dtype='float64').reshape((-1, 3))
	N = np.fromfile(n_files[ii], dtype='float64').reshape((-1, 3))
	I = np.fromfile(i_files[ii], dtype='int32')
	F = np.fromfile(f_files[ii], dtype='int32').reshape((-1, 3))
	FN = np.fromfile(fn_files[ii], dtype='float64').reshape((-1, 3))
	B = np.fromfile(b_files[ii], dtype='int32')
	S = np.zeros((F.shape[0]), dtype='int32')

	groups = []
	indices = np.arange(V.shape[0])
	AverageSplit(V, indices, groups, 0)

	pred_boundary = np.zeros((B.shape[0]), dtype='int32')
	pred_face_labels = np.zeros((F.shape[0]), dtype='int32')

	for j in range(len(groups)):
		print(j, len(groups))
		g = groups[j]

		mask = np.zeros((V.shape[0]), dtype='bool')
		map_mask = np.zeros((V.shape[0]), dtype='int32')
		V_split = V[g]
		N_split = N[g]
		B_split = B[g]
		mask[g] = 1
		map_mask[g] = np.arange(g.shape[0])
		valid_fidx = mask[F[:,0]] & mask[F[:,1]] & mask[F[:,2]]
		
		F_split = map_mask[F[valid_fidx]]
		VC_split = VC[valid_fidx]
		FN_split = FN[valid_fidx]
		S_split = S[valid_fidx]
		I_split = I[valid_fidx]

		edge_score, edge_indices = Predict(V_split, N_split, F_split)
		edge_score = (edge_score > 0.5).astype('int32')

		BB = (np.sum(B_split[edge_indices],axis=1) > 0).astype('int32')
		#print(B.shape, F_split.shape, edge_score.shape, edge_indices.shape)
		face_labels = np.zeros((F_split.shape[0]), dtype='int32')
		masks = np.zeros((V_split.shape[0]), dtype='int32')
		Regiongrow.RegionGrowing(c_void_p(edge_score.ctypes.data), c_void_p(F_split.ctypes.data),
			V_split.shape[0], F_split.shape[0], c_void_p(face_labels.ctypes.data), c_void_p(masks.ctypes.data),
			c_float(0.5))

		pred_boundary[g] = masks

	v1 = np.concatenate([F[:,0:1], F[:,1:2], F[:,2:3]])
	v2 = np.concatenate([F[:,1:2], F[:,2:3], F[:,0:1]])
	edge_idx = np.concatenate([v1.reshape(-1,1), v2.reshape(-1,1)], axis=1)
	pred_boundary = (np.sum(pred_boundary[edge_idx], axis=1) > 0).astype('int32')
	masks = np.zeros((V.shape[0]), dtype='int32')
	Regiongrow.RegionGrowing(c_void_p(pred_boundary.ctypes.data), c_void_p(F.ctypes.data),
		V.shape[0], F.shape[0], c_void_p(pred_face_labels.ctypes.data), c_void_p(masks.ctypes.data),
		c_float(0.99))
	print('done')

	colors = np.random.rand(np.max(pred_face_labels) + 1,3)
	fp = open('results/visualize/final.obj', 'w')
	for k in range(VC.shape[0]):
		v = VC[k]# + offset
		if pred_face_labels[k] < 0:
			c = np.array([0,0,0])
		else:
			c = colors[pred_face_labels[k]]
		fp.write('v %f %f %f %f %f %f\n'%(
			v[0], v[1], v[2], c[0], c[1], c[2]))
	fp.close()

model = Network(cfg)

use_cuda = torch.cuda.is_available()
logger.info('cuda available: {}'.format(use_cuda))
assert use_cuda

model = model.cuda()
##### model_fn (criterion)
model_fn = model_fn_decorator(True)

start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, 0, False, cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

model.eval()

t = ''
b_files = sorted(glob.glob('../data/Scene/*%s*_b.txt'%(t)))
fn_files = sorted(glob.glob('../data/Scene/*%s*_fn.txt'%(t)))
f_files = sorted(glob.glob('../data/Scene/*%s*_f.txt'%(t)))
i_files = sorted(glob.glob('../data/Scene/*%s*_i.txt'%(t)))
n_files = sorted(glob.glob('../data/Scene/*%s*_n.txt'%(t)))
vc_files = sorted(glob.glob('../data/Scene/*%s*_vc.txt'%(t)))
v_files = sorted(glob.glob('../data/Scene/*%s*_v.txt'%(t)))

Parse(0)
