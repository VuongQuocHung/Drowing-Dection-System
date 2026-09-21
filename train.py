import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import time

from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, CollateFunc_seq, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from config import build_default_config, build_model_config
from models import build_model

from utils.misc import parse_args
from utils.misc_eval import add_entry, reset_log_stats


GLOBAL_SEED = 42





def train():
    beginning_time = time.time()
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # config
    d_cfg = build_default_config(args)
    m_cfg = build_model_config(args)
    if not os.path.exists(d_cfg['save_folder']):
        os.makedirs(d_cfg['save_folder'])
    log_path = os.path.join(d_cfg['save_folder'], 'log_loss.txt')
    with open(log_path, 'w') as f:
        f.write('')
    reset_log_stats(d_cfg['save_folder'])

    # dist
    world_size = distributed_utils.get_world_size()
    per_gpu_batch = d_cfg['batch_size'] // world_size
    print('World size: {}'.format(world_size))
    if d_cfg['distributed']:
        distributed_utils.init_distributed_mode(d_cfg, args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(d_cfg['save_folder'])
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if d_cfg['cuda']:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # CHANGES ########################################################################
    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=True)
    if d_cfg['mode']=='reference':
        collate_fn = CollateFunc()
    elif d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='2Dseq' or d_cfg['mode']=='3Dseq':
        collate_fn = CollateFunc_seq()
    else:
        raise NotImplementedError
    # CHANGES ########################################################################


    # dataloader
    dataloader = build_dataloader(d_cfg, args, dataset, per_gpu_batch, collate_fn, is_train=True)


    # CHANGES ########################################################################
    # build model
    model, criterion = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes, 
        trainable=True,
        resume=d_cfg['resume']
        )
    # CHANGES ########################################################################
    model = model.to(device).train()

    # DDP
    model_without_ddp = model
    if d_cfg['distributed']:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # SyncBatchNorm
    if d_cfg['sybn'] and d_cfg['distributed']:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # # Compute FLOPs and Params
    # if distributed_utils.is_main_process():
    #     model_copy = deepcopy(model_without_ddp)
    #     FlOPs, Prams =FLOPs_and_Params(
    #         model=model_copy,
    #         img_size=d_cfg['test_size'],
    #         T_len=d_cfg['T_len'],
    #         mode=d_cfg['mode'],
    #         device=device)
    #     del model_copy
    #     add_entry('FLOPs', FlOPs, d_cfg['save_folder'])
    #     add_entry('Params', Prams, d_cfg['save_folder'])

    # optimizer
    base_lr = d_cfg['base_lr']
    accumulate = d_cfg['accumulate']
    optimizer, start_epoch = build_optimizer(d_cfg, model_without_ddp, base_lr, d_cfg['resume'])

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, d_cfg['lr_epoch'], d_cfg['lr_decay_ratio'])

    # warmup scheduler
    warmup_scheduler = build_warmup(d_cfg, base_lr=base_lr)

    # training configuration
    max_epoch = d_cfg['max_epoch']
    epoch_size = len(dataloader)
    warmup = True

    # eval before training
    if d_cfg['eval_first'] and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        eval_one_epoch(d_cfg, args, model_without_ddp, evaluator, 0, path_to_save)

    # start to train
    t0 = time.time()
    for epoch in range(start_epoch, max_epoch):
        if d_cfg['distributed']:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size

            # warmup
            if ni < d_cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == d_cfg['wp_iter'] and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, lr=base_lr, base_lr=base_lr)

            # to device
            video_clips = video_clips.to(device)

            # inference
            outputs = model(video_clips)


            # CHANGES ########################################################################
            # loss
            # loss_dict = {}
            # losses = 0
            if d_cfg['mode']=='reference':
                loss_dict = criterion(outputs, targets)
                losses = loss_dict['losses']
            elif d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='2Dseq' or d_cfg['mode']=='3Dseq':
                for outputs, targets, frame_ids in zip(outputs, targets, frame_ids):
                    loss_dict = criterion(outputs, targets)
                    losses = loss_dict['losses']
                    losses =+ losses
            else:
                raise NotImplementedError
            # CHANGES ########################################################################


            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward
            losses /= accumulate
            losses.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                    
            # Display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                print_log(cur_lr, epoch,  max_epoch, iter_i, epoch_size,loss_dict_reduced, t1-t0, accumulate, log_path)
            
                t0 = time.time()

        lr_scheduler.step()
        
        # evaluation
        # eval_iteration = ((epoch + 1) == max_epoch)
        eval_iteration = True
        eval_one_epoch(d_cfg, args, model_without_ddp, evaluator, epoch, path_to_save, eval_iteration)
    training_time = time.time() - beginning_time
    add_entry('training_time', training_time, d_cfg['save_folder'])


def eval_one_epoch(d_cfg, args, model_eval, evaluator, epoch, path_to_save, eval_iteration=True):
    # check evaluator
    if distributed_utils.is_main_process():
        
        # save model
        print('Saving state, epoch:', epoch + 1)
        weight_name = '{}_epoch_{}.pth'.format(d_cfg['version'], epoch+1)
        checkpoint_path = os.path.join(path_to_save, weight_name)
        torch.save({'model': model_eval.state_dict(),
                    'epoch': epoch,
                    'args': args}, 
                    checkpoint_path)   

        if evaluator is None:
            print('No evaluator ... save model and go on training.')
            
        elif eval_iteration == True:
            print('eval ...')
            # set eval mode
            model_eval.trainable = False
            model_eval.eval()

            # evaluate
            evaluator.weight_name = os.path.splitext(weight_name)[0]
            evaluator.evaluate_frame_map(model_eval, epoch + 1)
                
            # set train mode.
            model_eval.trainable = True
            model_eval.train()
          

    if d_cfg['distributed']:
        # wait for all processes to synchronize
        dist.barrier()


def print_log(lr, epoch, max_epoch, iter_i, epoch_size, loss_dict, time, accumulate, log_path):
    # basic infor
    log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
    log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
    log += '[lr: {:.6f}]'.format(lr[0])
    # loss infor
    for k in loss_dict.keys():
        if k == 'losses':
            log += '[{}: {:.2f}]'.format(k, loss_dict[k] * accumulate)
        else:
            log += '[{}: {:.2f}]'.format(k, loss_dict[k])

    # other infor
    log += '[time: {:.2f}]'.format(time)

    # print log information
    print(log, flush=True)
    add_log(log, log_path)



def add_log(log, log_path):
    with open(log_path, 'a') as f:
        f.write(log + '\n')



if __name__ == '__main__':
    train()
