import argparse

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from dataset_utils.LifeguardDataset import LifeguardDataset
from dataset_utils.LifeguardDataset_2Dseq import LifeguardDataset_2Dseq
from dataset_utils.LifeguardDataset_3Dseq import LifeguardDataset_3Dseq


from evaluator.lifeguard_evaluator import Lifeguard_Evaluator
from evaluator.lifeguard_evaluator_2Dseq import Lifeguard_Evaluator_2Dseq
from evaluator.lifeguard_evaluator_3Dseq import Lifeguard_Evaluator_3Dseq




def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=None,
                        help='use cuda.')
    parser.add_argument('--config', default='default_config', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--weight', type=str)

    return parser.parse_args()





def build_dataset(d_cfg, args, is_train=False):
    """
        d_cfg: dataset config
    """

    # dataset
    if d_cfg['mode']=='reference':
        dataset = LifeguardDataset(
            cfg=d_cfg,
            mode='train'
            )
        evaluator = Lifeguard_Evaluator(
        d_cfg=d_cfg,
        iou_thresh=0.5, # default 0.5
        collate_fn=CollateFunc()
        )
    elif d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='3Dseq':
        dataset = LifeguardDataset_3Dseq(
            cfg=d_cfg,
            mode='train'
            )
        evaluator = Lifeguard_Evaluator_3Dseq(
        d_cfg=d_cfg,
        iou_thresh=0.5, # default 0.5
        collate_fn=CollateFunc_seq()
        )
    elif d_cfg['mode']=='2Dseq':
        dataset = LifeguardDataset_2Dseq(
            cfg=d_cfg,
            mode='train'
            )
        evaluator = Lifeguard_Evaluator_2Dseq(
        d_cfg=d_cfg,
        iou_thresh=0.5, # default 0.5
        collate_fn=CollateFunc_seq()
        )
    else:
        raise NotImplementedError

    num_classes = dataset.num_classes

    
    print('==============================')
    print('Training model on the lifeguard dataset:')
    print('The dataset size:', len(dataset))

    if not d_cfg['eval']:
        # no evaluator during training stage
        evaluator = None

    return dataset, evaluator, num_classes


def build_dataloader(d_cfg, args, dataset, batch_size, collate_fn=None, is_train=False):
    if is_train:
        # distributed
        if d_cfg['distributed']:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                            batch_size, 
                                                            drop_last=True)
        # train dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn, 
            num_workers=d_cfg['num_workers'],
            pin_memory=True
            )
    else:
        # test dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=d_cfg['num_workers'],
            drop_last=False,
            pin_memory=True
            )
    
    return dataloader
    

def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


# from batch with where one videoclip and its target are together to stacked videoclip and stacked target for one batch
class CollateFunc(object):
    def __call__(self, batch):
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_key_target



class CollateFunc_seq(object):
    def __call__(self, batch):
        ids = []
        clips = []
        targets = []
        for sequence in batch:
            ids.append(sequence[0])
            clips.append(sequence[1])
            targets.append(sequence[2])

        # padding
        S_len = max([clip_seq.size()[0] for clip_seq in clips])
        # 2Dseq: List [B, S, 3, H, W] -> [S, B, 3, H, W]
        # 3Dseq: List [B, S, 3, T, H, W] -> [S, B, 3, T, H, W]
        batch_video_clips = pad_sequence(clips)
        for id_seq in ids:
            id_seq.extend([None]*(S_len - len(id_seq)))
        for target_seq in targets:
            target_seq.extend([None]*(S_len - len(target_seq)))

        # transpose
        batch_frame_id = swap_first_two_dimensions(ids)
        batch_key_target = swap_first_two_dimensions(targets)
        
        return batch_frame_id, batch_video_clips, batch_key_target
    


def swap_first_two_dimensions(input_list):
    # Get the dimensions of the input list
    outer_dim = len(input_list)
    inner_dim = len(input_list[0])
    
    # Create a new list with swapped dimensions
    swapped_list = [[None] * outer_dim for _ in range(inner_dim)]
    
    for i in range(outer_dim):
        for j in range(inner_dim):
            swapped_list[j][i] = input_list[i][j]
    
    return swapped_list