import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader
import psutil
sys.path.append('../')

from util.config import cfg
from lib.pointgroup_ops.functions import pointgroup_ops

from util.config import cfg

from scipy.spatial import cKDTree

class ABCDataset:
	def __init__(self):
		self.root_dir = cfg.root_dir
		self.batch_size = cfg.batch_size
		self.scale = cfg.scale
		train_files = sorted([self.root_dir + '/train/' + p for p in os.listdir(self.root_dir + '/train/') if p[-3:] == 'npz'])
		self.files = train_files
		train_set = list(range(len(self.files)))
		self.full_scale = cfg.full_scale
		self.max_npoint = cfg.max_npoint
		self.mode = cfg.mode
		self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.Merge, num_workers=0,
			shuffle=False, sampler=None, drop_last=True, pin_memory=True)

	def __len__(self):
		return len(self.files)

	def dataAugment(self, xyz, normal, jitter=False, flip=False, rot=False):
		m = np.eye(3)
		if jitter:
			m += np.random.randn(3, 3) * 0.1
		if flip:
			m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
		if rot:
			theta = np.random.rand() * 2 * math.pi
			m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
		return np.matmul(xyz, m), np.matmul(normal, m)

	#Elastic distortion
	def elastic(self, x, gran, mag):
		blur0 = np.ones((3, 1, 1)).astype('float32') / 3
		blur1 = np.ones((1, 3, 1)).astype('float32') / 3
		blur2 = np.ones((1, 1, 3)).astype('float32') / 3

		bb = np.abs(x).max(0).astype(np.int32)//gran + 3
		noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
		noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
		noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
		noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
		noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
		noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
		noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
		ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
		interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
		def g(x_):
			return np.hstack([i(x_)[:,None] for i in interp])
		return x + g(x) * mag

	def crop(self, xyz):
		'''
		:param xyz: (n, 3) >= 0
		'''
		xyz_offset = xyz.copy()

		offset = (xyz_offset.max(0) - np.array([self.full_scale[1]] * 3)) * np.random.rand(3)
		xyz_offset = xyz_offset - offset

		valid_idxs = (xyz_offset.min(1) >= 0) * (xyz_offset.max(1) < self.full_scale[1] - 1)

		full_scale = np.array([self.full_scale[1]] * 3)
		room_range = xyz.max(0) - xyz.min(0)
		while (valid_idxs.sum() > self.max_npoint):
			offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
			xyz_offset = xyz + offset
			valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale - 1).sum(1) == 3)
			full_scale[:2] -= 32

		return xyz_offset, valid_idxs

	def Merge(self, id):
		locs = []
		locs_indices = []
		locs_float = []
		locs_float_gt = []
		normals = []
		normals_gt = []
		boundaries = []
		edge_indices = []
		file_names = []
		semantics_gt = []
		voffset = 0
		for i, idx in enumerate(id):

			file_names.append(self.files[idx])
			data = np.load(self.files[idx])

			#xyz_origin, normal, boundary = data['V'], data['N'], data['B']
			xyz_origin, normal, boundary, F, SF = data['V'], data['N'], data['B'], data['F'], data['S']
			semantics = np.zeros((xyz_origin.shape[0]), dtype='int32')
			semantics[F[:,0]] = SF
			semantics[F[:,1]] = SF
			semantics[F[:,2]] = SF
			original_indices = np.arange(xyz_origin.shape[0])
			xyz_middle, normal_middle = self.dataAugment(xyz_origin, normal, False, True, True)

			xyz_middle_noise = xyz_middle
			normal_middle_noise = normal_middle
			xyz_middle -= xyz_middle_noise.min(0)
			xyz_middle_noise -= xyz_middle_noise.min(0)

			### scale
			xyz = xyz_middle_noise * self.scale

			### elastic
			#xyz = self.elastic(xyz, 6, 40)
			#xyz = self.elastic(xyz, 20, 160)

			### offset0
			xyz, valid_idxs = self.crop(xyz)
			xyz_middle = xyz_middle[valid_idxs]
			xyz_middle_noise = xyz_middle_noise[valid_idxs]
			normal_middle_noise = normal_middle_noise[valid_idxs]
			xyz = xyz[valid_idxs]
			normal_middle = normal_middle[valid_idxs]

			boundary = boundary[valid_idxs]
			semantics = semantics[valid_idxs]
			original_indices = original_indices[valid_idxs]
			sampling_map = np.zeros((xyz_origin.shape[0]),dtype='int32')
			final_indices = np.arange(original_indices.shape[0])
			sampling_map[original_indices[:]] = final_indices

			tree = cKDTree(xyz_middle_noise)
			d, ii = tree.query(xyz_middle_noise, k=16, n_jobs=16)
			locs_indices.append(torch.from_numpy(ii + voffset))

			locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
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

			#edge_boundary = (np.sum(boundary[edge_idx], axis=1) > 0).astype('int64')

			edge_indices.append(torch.from_numpy(edge_idx + voffset))
			boundaries.append(torch.from_numpy(boundary))

			voffset += xyz_middle.shape[0]

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
		voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

		#if cfg.thick == 1:
		pt_idx = torch.where(boundaries > 0)[0]
		p2v_map_long = p2v_map.long()

		boundary_mask = torch.zeros((voxel_locs.shape[0])).long()
		boundary_mask[p2v_map_long[pt_idx].long()] = 1
		boundaries = boundary_mask[p2v_map_long]

		boundaries = (boundaries[edge_indices].sum(dim=1) > 0).long()

		return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
			'locs_float': locs_float, 'locs_indices': locs_indices, 'locs_float_gt': locs_float_gt,
			'normals': normals, 'normals_gt': normals_gt, 'semantics_gt':semantics_gt, 'boundaries': boundaries, 'edge_indices': edge_indices,
			'id': id, 'spatial_shape': spatial_shape, 'file_names': file_names}

def Visualize(locs, locs_float, locs_float_gt, labels, prefix):
	max_label = np.max(labels) + 1
	colors = np.random.rand(max_label, 3)
	fp = open('Visualize/points.obj', 'w')
	for i in range(locs_float.shape[0]):
		if locs[i] == 0:
			p = locs_float[i]
			if labels[i] < 0:
				c = np.array([0,0,0])
			else:
				c = colors[labels[i]]
			fp.write('v %f %f %f %f %f %f\n'%(p[0],p[1],p[2],c[0],c[1],c[2]))
	fp.close()

	fp = open('Visualize/points_gt.obj', 'w')
	for i in range(locs_float_gt.shape[0]):
		if locs[i] == 0:
			p = locs_float_gt[i]
			if labels[i] < 0:
				c = np.array([0,0,0])
			else:
				c = colors[labels[i]]
			fp.write('v %f %f %f %f %f %f\n'%(p[0],p[1],p[2],c[0],c[1],c[2]))
	fp.close()

if __name__ == '__main__':
	dataset = ABCDataset()
	for i, batch in enumerate(dataset.train_data_loader):
		locs_float = batch['locs_float'].data.cpu().numpy()
		locs_float_gt = batch['locs_float_gt'].data.cpu().numpy()
		locs = batch['locs'].data.cpu().numpy()
		#labels = batch['instances'].data.cpu().numpy()
		#return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
		#	'locs_float': locs_float, 'locs_indices': locs_indices, 'locs_float_gt': locs_float_gt,
		#	'normals': normals, 'normals_gt': normals_gt, 'boundaries': boundaries, 'edge_indices': edge_indices,
		#	'id': id, 'spatial_shape': spatial_shape, 'file_names': file_names}

		normals = batch['normals'].data.cpu().numpy()
		normals_gt = batch['normals_gt'].data.cpu().numpy()
		semantics = batch['semantics_gt'].data.cpu().numpy()
		edge_indices = batch['edge_indices'].data.cpu().numpy()
		boundaries = batch['boundaries'].data.cpu().numpy()

		for j in range(3,4):
			vindices = locs[:,0] == j

			V = locs_float[vindices]
			N = normals[vindices]
			S = semantics[vindices]
			colors = np.random.rand(10000,3)

			print(V.shape[0], np.max(edge_indices))

			fp = open('Visualizes/model-%d-n.obj'%(j), 'w')
			for k in range(V.shape[0]):
				v = V[k]
				n = colors[S[k]]
				fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],n[0],n[1],n[2]))
			fp.close()

			V = locs_float
			N = normals
			fp = open('Visualizes/model-%d-b0.obj'%(j), 'w')
			voffset = 1
			for k in range(boundaries.shape[0]):
				if boundaries[k] == 1:
					continue
				if vindices[edge_indices[k][0]] == 0:
					continue
				v0 = V[edge_indices[k][0]]
				v1 = V[edge_indices[k][1]]
				fp.write('v %f %f %f\n'%(v0[0],v0[1],v0[2]))
				fp.write('v %f %f %f\n'%(v1[0],v1[1],v1[2]))
				fp.write('l %d %d\n'%(voffset, voffset + 1))
				voffset += 2
			fp.close()

			fp = open('Visualizes/model-%d-b1.obj'%(j), 'w')
			voffset = 1
			for k in range(boundaries.shape[0]):
				if boundaries[k] == 0:
					continue
				if vindices[edge_indices[k][0]] == 0:
					continue
				v0 = V[edge_indices[k][0]]
				v1 = V[edge_indices[k][1]]
				fp.write('v %f %f %f\n'%(v0[0],v0[1],v0[2]))
				fp.write('v %f %f %f\n'%(v1[0],v1[1],v1[2]))
				fp.write('l %d %d\n'%(voffset, voffset + 1))
				voffset += 2
			fp.close()
		print(normals.shape, normals_gt.shape, locs_float.shape)
		print(edge_indices.shape, boundaries.shape)

		exit(0)