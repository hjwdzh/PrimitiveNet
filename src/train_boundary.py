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

def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train_boundary.py {}'.format(backup_dir))
    os.system('cp {} {}'.format('model_boundary.py', backup_dir))
    os.system('cp {} {}'.format('dataset.py', backup_dir))
    os.system('cp {} {}'.format('config/abc.yaml', backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

def Visualize(batch_ids, xyz, label, prefix):
    max_id = np.max(batch_ids)
    for i in range(max_id + 1):
        idx = np.where(batch_ids == i)[0]
        fp = open('Visualize/%s_%02d.obj'%(prefix, i), 'w')
        xyz_o = xyz[idx]
        label_o = label[idx]
        for j in range(xyz_o.shape[0]):
            r, g, b = 0, 0, 0
            if label_o[j] == 0:
                g = 255
            else:
                r = 255
            p = xyz_o[j]
            fp.write('v %f %f %f %d %d %d\n'%(p[0], p[1], p[2], r, g, b))
        fp.close()

iterations = 0
def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    t1 = time.time()
    fp = open(cfg.exp_path + '/train_logs.txt', 'w')
    global iterations
    for i, batch in enumerate(train_loader):
        fp.write('%d %d\n'%(epoch, i))
        for j in batch['file_names']:
            fp.write('%s\n'%(j))
        fp.flush()
        t2 = time.time()
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        ##### prepare input and forward
        loss, loss_out, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

        writer.add_scalar('train/o', loss_out['o_loss'][0], iterations)
        writer.add_scalar('train/n', loss_out['n_loss'][0], iterations)
        writer.add_scalar('train/b', loss_out['b_loss'][0], iterations)
        writer.flush()
        iterations += 1
        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        t3 = time.time()
        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        sys.stdout.write(
            "epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
            (epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val, am_dict['loss'].avg,
             data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
        if (i == len(train_loader) - 1): print()
        t4 = time.time()
        t1 = time.time()

    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

    utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)

    #for k in am_dict.keys():
    #    if k in visual_dict.keys():
    #        writer.add_scalar(k+'_train', am_dict[k].avg, epoch)

if __name__ == '__main__':
    ##### init
    init()

    ##### model
    logger.info('=> creating model ...')

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda

    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    ##### model_fn (criterion)
    model_fn = model_fn_decorator()

    ##### dataset
    dataset = ABCDataset()

    ##### resume
    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, 0, False, cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        #if utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch):
        #    eval_epoch(dataset.val_data_loader, model, model_fn, epoch)
