import os
import sys
import numpy as np
import torch
from lapsolver import solve_dense
from multiprocessing import Pool

def guard_mean_shift(ms, embedding, quantile, iterations, kernel_type="gaussian"):
	"""
	Some times if band width is small, number of cluster can be larger than 50, that
	but we would like to keep max clusters 50 as it is the max number in our dataset.
	In that case you increase the quantile to increase the band width to decrease
	the number of clusters.
	"""
	while True:
		_, center, bandwidth, cluster_ids = ms.mean_shift(
			embedding, 10000, quantile, iterations, kernel_type=kernel_type
		)
		if torch.unique(cluster_ids).shape[0] > 49:
			quantile *= 1.2
		else:
			break
	return center, bandwidth, cluster_ids

def mean_IOU_primitive_segment(matching, predicted_labels, labels, pred_prim, gt_prim):
	"""
	Primitive type IOU, this is calculated over the segment level.
	First the predicted segments are matched with ground truth segments,
	then IOU is calculated over these segments.
	:param matching
	:param pred_labels: N x 1, pred label id for segments
	:param gt_labels: N x 1, gt label id for segments
	:param pred_prim: K x 1, pred primitive type for each of the predicted segments
	:param gt_prim: N x 1, gt primitive type for each point
	"""
	batch_size = labels.shape[0]
	IOU = []
	IOU_prim = []

	for b in range(batch_size):
		iou_b = []
		iou_b_prim = []
		iou_b_prims = []
		len_labels = np.unique(predicted_labels[b]).shape[0]
		rows, cols = matching[b]
		count = 0
		for r, c in zip(rows, cols):
			pred_indices = predicted_labels[b] == r
			gt_indices = labels[b] == c

			# use only matched segments for evaluation
			if (np.sum(gt_indices) == 0) or (np.sum(pred_indices) == 0):
				continue

			# also remove the gt labels that are very small in number
			if np.sum(gt_indices) < 100:
				continue

			iou = np.sum(np.logical_and(pred_indices, gt_indices)) / (
						np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
			iou_b.append(iou)

			# evaluation of primitive type prediction performance
			gt_prim_type_k = gt_prim[b][gt_indices][0]
			try:
				predicted_prim_type_k = pred_prim[b][r]
			except:
				import ipdb;
				ipdb.set_trace()

			iou_b_prim.append(gt_prim_type_k == predicted_prim_type_k)
			iou_b_prims.append([gt_prim_type_k, predicted_prim_type_k])

		# find the mean of IOU over this shape
		IOU.append(np.mean(iou_b))
		IOU_prim.append(np.mean(iou_b_prim))
	return np.mean(IOU), np.mean(IOU_prim), iou_b_prims

def to_one_hot(target_t, maxx=500):
	N = target_t.shape[0]
	maxx = np.max(target_t) + 1
	if maxx <= 0:
		maxx = 1
	target_one_hot = np.zeros((N, maxx))

	for i in range(target_t.shape[0]):
		if target_t[i] >= 0:
			target_one_hot[i][target_t[i]] = 1
	#target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)

	target_one_hot = torch.from_numpy(target_one_hot)
	return target_one_hot

def relaxed_iou_fast(pred, gt, max_clusters=500):
	batch_size, N, K = pred.shape
	normalize = torch.nn.functional.normalize
	one = torch.ones(1)

	norms_p = torch.unsqueeze(torch.sum(pred, 1), 2)
	norms_g = torch.unsqueeze(torch.sum(gt, 1), 1)
	cost = []

	for b in range(batch_size):
		p = pred[b]
		g = gt[b]
		c_batch = []
		dots = p.transpose(1, 0) @ g
		r_iou = dots
		r_iou = r_iou / (norms_p[b] + norms_g[b] - dots + 1e-7)
		cost.append(r_iou)
	cost = torch.stack(cost, 0)
	return cost

def primitive_type_segment_torch(pred, weights):
	"""
	Returns the primitive type for every segment in the predicted shape.
	:param pred: N x L
	:param weights: N x k
	"""
	d = torch.unsqueeze(pred, 2).float() * torch.unsqueeze(weights, 1).float()
	d = torch.sum(d, 0)
	return torch.max(d, 0)[1]

def SIOU_matched_segments(target, pred_labels, primitives_pred, primitives, weights):
	"""
	Computes iou for segmentation performance and primitive type
	prediction performance.
	First it computes the matching using hungarian matching
	between predicted and ground truth labels.
	Then it computes the iou score, starting from matching pairs
	coming out from hungarian matching solver. Note that
	it is assumed that the iou is only computed over matched pairs.
	That is to say, if any column in the matched pair has zero
	number of points, that pair is not considered.
	
	It also computes the iou for primitive type prediction. In this case
	iou is computed only over the matched segments.
	"""
	# 2 is open spline and 9 is close spline
	primitives[primitives == 0] = 9
	primitives[primitives == 6] = 9
	primitives[primitives == 7] = 9
	primitives[primitives == 8] = 2

	primitives_pred[primitives_pred == 0] = 9
	primitives_pred[primitives_pred == 6] = 9
	primitives_pred[primitives_pred == 7] = 9
	primitives_pred[primitives_pred == 8] = 2

	labels_one_hot = to_one_hot(target)
	cluster_ids_one_hot = to_one_hot(pred_labels)

	cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
	cost_ = 1.0 - cost.data.cpu().numpy()
	matching = []

	for b in range(1):
		rids, cids = solve_dense(cost_[b])
		matching.append([rids, cids])

	primitives_pred_hot = to_one_hot(primitives_pred, 10).float()

	# this gives you what primitive type the predicted segment has.
	prim_pred = primitive_type_segment_torch(primitives_pred_hot, weights).data.cpu().numpy()
	target = np.expand_dims(target, 0)
	pred_labels = np.expand_dims(pred_labels, 0)
	prim_pred = np.expand_dims(prim_pred, 0)
	primitives = np.expand_dims(primitives, 0)

	segment_iou, primitive_iou, iou_b_prims = mean_IOU_primitive_segment(matching, pred_labels, target, prim_pred,
																		 primitives)
	return segment_iou, primitive_iou, matching, iou_b_prims

data_path = sys.argv[1]
files = [data_path + '/' + f for f in os.listdir(data_path) if f[-3:] == 'npz']

s_ious = []
p_ious = []

def SaveRelation(i):
	global files
	f = files[i]
	data = np.load(f)
	V, L, L_gt, S, S_gt = data['V'], data['L'], data['L_gt'], data['S'], data['S_gt']

	weights = to_one_hot(L, np.unique(L).shape[0])
	s_iou, p_iou, _, _ = SIOU_matched_segments(
		L_gt,
		L,
		S,
		S_gt,
		weights,
	)

	if np.isnan(s_iou) or np.isnan(p_iou):
		return

	result = np.array([s_iou, p_iou])

	np.savez_compressed('results/relation-iou/%d.npz'%(i), result=result)
with Pool(4) as p:
	p.map(SaveRelation, [i for i in range(len(files))])

files = os.listdir('results/relation-iou')
s_ious = []
p_ious = []
for f in files:
	r = np.load('results/relation-iou/%s'%(f))['result']
	s_ious.append(r[0])
	p_ious.append(r[1])

fp = open('results/statistics/iou.txt','w')
fp.write("SIOU=%f LIOU=%f\n"%(np.mean(s_ious), np.mean(p_ious)))
