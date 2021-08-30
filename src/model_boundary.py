import torch
import torch.nn as nn

import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
import numpy as np

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output

class PointConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_fn):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            norm_fn(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            norm_fn(output_dim)
        )

    def forward(self, input):
        return self.conv(input)

class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_fn):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(len(output_dim)):
            layers.append(nn.Linear(prev_dim, output_dim[i]))
            layers.append(norm_fn(output_dim[i]))
            if i < len(output_dim) - 1:
                layers.append(nn.ReLU())
            prev_dim = output_dim[i]

        self.conv = nn.Sequential(
            *layers
        )

    def forward(self, input):
        return self.conv(input)

class SemanticPrediction(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords and cfg.normal == 1:
            input_c += 3

        #### backbone
        self.point_conv1 = PointConvBlock(input_c, m, norm_fn)
        self.point_conv2 = PointConvBlock(m, m, norm_fn)
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(m, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        #### semantic segmentation
        self.linear = nn.Linear(m * 2, 32) # bias(default): True
        self.linear_semantics = nn.Linear(m * 2, classes)
        self.mlp = MLPBlock(32, [32, 2], norm_fn)

        self.apply(self.set_bn_init)

        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, voxel_coords, v2p_map, p2v_map, feats, coords, edge_idx, spatial_shape, epoch, cfg, max_edge, boundary_edge_idx = None):
        '''
        :param input_map: (N), int, cuda
        :param feats: (N, k), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        feats = self.point_conv1(feats)
        feats = torch.max(feats[edge_idx], dim=1)[0]
        feats = self.point_conv2(feats)
        feats = torch.max(feats[edge_idx], dim=1)[0]

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)
        input_map = p2v_map
        batch_idxs = coords[:,0].int()

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]
        output_feats = torch.cat([feats, output_feats], dim = 1)

        #### semantic segmentation
        primitive_embedding = self.linear(output_feats)   # (N, nClass), float
        primitive_pred = self.linear_semantics(output_feats)
        pred_o = primitive_embedding[:,:3]
        pred_n = primitive_embedding[:,3:6]
        l = torch.sqrt(1e-9+torch.sum(pred_n*pred_n, dim=1)).view(-1, 1)
        pred_n = pred_n / torch.cat([l, l, l], dim=1)

        primitive_latent_code = primitive_embedding[:,6:]

        if boundary_edge_idx is None:
            orig_idx = torch.arange(edge_idx.shape[0], device='cuda').view(-1, 1)
            orig_idx = torch.cat([orig_idx for i in range(edge_idx.shape[1])], dim=1)
            i_tensor = orig_idx.view(-1)
            j_tensor = edge_idx.view(-1)
    
            if max_edge >= 0:
                perm = torch.randperm(i_tensor.size(0),device='cuda')
                idx = perm[:max_edge]
                i_tensor = i_tensor[idx]
                j_tensor = j_tensor[idx]
        else:
            i_tensor = boundary_edge_idx[:,0]
            j_tensor = boundary_edge_idx[:,1]

        primitive_info = torch.cat([pred_o, primitive_embedding[:,3:]], dim=1)
        #info = torch.cat([primitive_info[j_tensor], primitive_info[i_tensor]], dim=1)
        info = torch.max(primitive_info[j_tensor], primitive_info[i_tensor])
        boundary = self.mlp(info)

        ret['pred_o'] = pred_o
        ret['pred_n'] = pred_n
        ret['boundary'] = boundary
        ret['primitive_pred'] = primitive_pred
        ret['edges'] = torch.cat([i_tensor.view(-1,1), j_tensor.view(-1,1)], dim=1)
        ret['feat'] = primitive_latent_code
        return ret


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    #semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    #score_criterion = nn.BCELoss(reduction='none').cuda()
    softmax = torch.nn.Softmax(dim=1)
    bce_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()#torch.nn.BCELoss()
    mse_criterion = torch.nn.MSELoss()

    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        normals_float = batch['normals'].cuda()
        labels = batch['boundaries'].cuda()                        # (N), long, cuda

        spatial_shape = batch['spatial_shape']

        gt_boundary = batch['boundaries'].cuda()
        gt_semantic = batch['semantics_gt'].cuda()

        edge_idx = batch['locs_indices'].cuda()
        boundary_edge_idx = batch['edge_indices'].cuda()

        if cfg.normal == 1:
            feats = torch.cat((coords_float, normals_float), 1)
        else:
            feats = coords_float

        ret = model(voxel_coords, v2p_map, p2v_map, feats, coords, edge_idx, spatial_shape, epoch, cfg, max_edge=1024*512, boundary_edge_idx = boundary_edge_idx)
        #semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        
        pred_o = ret['pred_o']
        pred_n = ret['pred_n']
        boundary = ret['boundary']
        pred_semantic = ret['primitive_pred']

        loss_inp = {}
        loss_inp['o'] = (pred_o, batch['locs_float_gt'].cuda())
        loss_inp['n'] = (pred_n, batch['normals_gt'].cuda())
        loss_inp['b'] = (boundary, gt_boundary)

        loss_inp['p'] = (pred_semantic, gt_semantic)

        loss, loss_out = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['o'] = pred_o
            preds['n'] = pred_n
            preds['e'] = ret['edges']
            preds['b'] = ret['boundary']

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, loss_out, preds, visual_dict, meter_dict

    def test_model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        normals_float = batch['normals'].cuda()

        spatial_shape = batch['spatial_shape']

        edge_idx = batch['locs_indices'].cuda()
        boundary_edge_idx = batch['edge_indices'].cuda()

        if cfg.normal == 1:
            feats = torch.cat((coords_float, normals_float), 1)
        else:
            feats = coords_float

        ret = model(voxel_coords, v2p_map, p2v_map, feats, coords, edge_idx, spatial_shape, epoch, cfg, max_edge=1024*512, boundary_edge_idx = boundary_edge_idx)
        #semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        
        pred_o = ret['pred_o']
        pred_n = ret['pred_n']
        boundary = ret['boundary']
        pred_semantic = ret['primitive_pred']

        return {'o': pred_o, 'n': pred_n, 'b': softmax(boundary), 'p': softmax(pred_semantic)}

    def loss_fn(loss_inp, epoch):

        loss_out = {}

        '''semantic loss'''
        o, gt_o = loss_inp['o']
        loss_out['o_loss'] = (mse_criterion(o, gt_o) / 3600.0, o.shape[0])

        n, gt_n = loss_inp['n']
        loss_out['n_loss'] = (mse_criterion(n, gt_n), o.shape[0])

        b, gt_b = loss_inp['b']
        loss_out['b_loss'] = (bce_criterion(b, gt_b), b.shape[0])

        p, gt_p = loss_inp['p']
        loss_out['p_loss'] = (bce_criterion(p, gt_p), p.shape[0])


        '''total loss'''
        loss = loss_out['o_loss'][0] + loss_out['n_loss'][0] + loss_out['b_loss'][0] + loss_out['p_loss'][0]
        return loss, loss_out

    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn

if __name__ == '__main__':
    model_fn_decorator()