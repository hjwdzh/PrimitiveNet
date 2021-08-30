import os
import sys
import numpy as np

import pickle
from multiprocessing import Pool

def Visualize(V, F, L, L_gt):
	face_labels = L
	gt_face_labels = L_gt
	num_labels = np.max(face_labels) + 1
	colors = np.random.rand(num_labels, 3)
	fp = open('Visualize/face.obj', 'w')
	voffset = 1
	for i in range(F.shape[0]):
		if face_labels[i] < 0:
			c = np.array([0,0,0])
		else:
			c = colors[face_labels[i]]
		for j in range(3):
			v = V[F[i][j]]
			fp.write('v %f %f %f %f %f %f\n'%(v[0], v[1], v[2], c[0], c[1], c[2]))
		fp.write('f %d %d %d\n'%(voffset, voffset + 1, voffset + 2))
		voffset += 3
	fp.close()

	num_labels = np.max(gt_face_labels) + 1
	colors = np.random.rand(num_labels, 3)
	fp = open('Visualize/face_gt.obj', 'w')
	voffset = 1
	for i in range(F.shape[0]):
		if gt_face_labels[i] < 0:
			c = np.array([0,0,0])
		else:
			c = colors[gt_face_labels[i]]
		for j in range(3):
			v = V[F[i][j]]
			fp.write('v %f %f %f %f %f %f\n'%(v[0], v[1], v[2], c[0], c[1], c[2]))
		fp.write('f %d %d %d\n'%(voffset, voffset + 1, voffset + 2))
		voffset += 3
	fp.close()

path = sys.argv[1]
files = [path + '/' + f for f in os.listdir(path)][:]

overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
min_region_size = 20

def SaveRelation(i):
	global files
	f = files[i]
	data = np.load(f)
	V, L, L_gt = data['V'], data['L'], data['L_gt']

	# build match
	M = np.max(L) + 1
	N = np.max(L_gt) + 1
	if M < 0:
		M = 0
	if N < 0:
		N = 0

	relation_pred = [{} for i in range(M)]
	relation_gt = [{} for i in range(N)]

	label_count = np.zeros((M))
	label_count_gt = np.zeros((N))

	for i in range(L.shape[0]):
		if L[i] >= 0 and L_gt[i] >= 0:
			if not L_gt[i] in relation_pred[L[i]]:
				relation_pred[L[i]][L_gt[i]] = 1
			else:
				relation_pred[L[i]][L_gt[i]] += 1

			if not L[i] in relation_gt[L_gt[i]]:
				relation_gt[L_gt[i]][L[i]] = 1
			else:
				relation_gt[L_gt[i]][L[i]] += 1

		if L[i] >= 0:
			label_count[L[i]] += 1
		if L_gt[i] >= 0:
			label_count_gt[L_gt[i]] += 1

	with open('results/relation/%s.pkl'%(f.split('/')[-1][:-4]), 'wb') as handle:
		pickle.dump((relation_pred, relation_gt, label_count, label_count_gt, L, L_gt), handle, protocol = pickle.HIGHEST_PROTOCOL)

if sys.argv[-1] == '1':
	with Pool(32) as p:
		p.map(SaveRelation, [i for i in range(len(files))])
	exit(0)

files = ['results/relation/' + f for f in os.listdir('results/relation')][:]

aps = []
for oi, overlap_th in enumerate(overlaps):
	hard_false_negatives = 0
	y_true = np.empty(0)
	y_score = np.empty(0)
	c = 0
	for fi, f in enumerate(files):
		c += 1
		with open(f, 'rb') as handle:
			relation_pred, relation_gt, label_count, label_count_gt, L, L_gt = pickle.load(handle)
		M = len(relation_pred)
		N = len(relation_gt)

		cur_true = np.ones((N))
		cur_score = np.ones((N)) * -1
		cur_match = np.zeros((N), dtype=np.bool)

		for gt_idx in range(N):
			found_match = False
			if label_count_gt[gt_idx] < min_region_size:
				continue
			for pred_idx, intersection in relation_gt[gt_idx].items():
				overlap = intersection / (label_count[pred_idx] + label_count_gt[gt_idx] - intersection + 0.0)
				if overlap > overlap_th:
					confidence = overlap
					if cur_match[gt_idx]:
						max_score = np.max([cur_score[gt_idx], confidence])
						min_score = np.min([cur_score[gt_idx], confidence])
						cur_score[gt_idx] = max_score

						cur_true = np.append(cur_true, 0)
						cur_score = np.append(cur_score, min_score)
						cur_match = np.append(cur_match, True)
					else:
						found_match = True
						cur_match[gt_idx] = True
						cur_score[gt_idx] = confidence

			if not found_match:
				hard_false_negatives += 1

		cur_true = cur_true[cur_match == True]
		cur_score = cur_score[cur_match == True]

		for pred_idx in range(M):
			if label_count[pred_idx] < min_region_size or label_count[pred_idx] < np.sum(label_count) * 0.1:
				continue
			found_gt = False
			for gt_idx, intersection in relation_pred[pred_idx].items():
				overlap = intersection / (label_count[pred_idx] + label_count_gt[gt_idx] - intersection + 0.0)
				if overlap > overlap_th:
					found_gt = True
					break
			if not found_gt:
				num_ignore = np.sum((L_gt[L == pred_idx] < 0).astype('int32'))
				for gt_idx, intersection in relation_pred[pred_idx].items():
					if label_count_gt[gt_idx] < min_region_size:
						num_ignore += intersection

				if num_ignore / (label_count[pred_idx] + 0.0) < overlap_th:
					cur_true = np.append(cur_true, 0)
					confidence = 0
					cur_score = np.append(cur_score, confidence)


		y_true = np.append(y_true, cur_true)
		y_score = np.append(y_score, cur_score)

	score_arg_sort      = np.argsort(y_score)
	y_score_sorted      = y_score[score_arg_sort]
	y_true_sorted       = y_true[score_arg_sort]
	y_true_sorted_cumsum = np.cumsum(y_true_sorted)

	# unique thresholds
	(thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
	num_prec_recall = len(unique_indices) + 1

	# prepare precision recall
	num_examples      = len(y_score_sorted)
	num_true_examples = y_true_sorted_cumsum[-1]
	precision         = np.zeros(num_prec_recall)
	recall            = np.zeros(num_prec_recall)

	# deal with the first point
	y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
	# deal with remaining
	for idx_res,idx_scores in enumerate(unique_indices):
		cumsum = y_true_sorted_cumsum[idx_scores-1]
		tp = num_true_examples - cumsum
		fp = num_examples      - idx_scores - tp
		fn = cumsum + hard_false_negatives
		p  = float(tp)/(tp+fp)
		r  = float(tp)/(tp+fn)
		precision[idx_res] = p
		recall   [idx_res] = r

	# first point in curve is artificial
	precision[-1] = 1.
	recall   [-1] = 0.

	# compute average of precision-recall curve
	recall_for_conv = np.copy(recall)
	recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
	recall_for_conv = np.append(recall_for_conv, 0.)

	stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
	# integrate is now simply a dot product
	ap_current = np.dot(precision, stepWidths)

	aps.append(ap_current)
	print(overlap_th, ap_current)

aps = np.array(aps)
fp = open(sys.argv[-2], 'w')
fp.write('ap50 = %f\n'%(aps[0]))
fp.write('ap75 = %f\n'%(aps[5]))
fp.write('ap25 = %f\n'%(aps[9]))
fp.write('map = %f\n'%(np.average(aps[:8])))
fp.close()
