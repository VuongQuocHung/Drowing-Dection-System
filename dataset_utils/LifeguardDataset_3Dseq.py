#!/usr/bin/python
# encoding: utf-8
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset_utils.LifeguardUtils import FileLoader
from dataset_utils.transforms import Augmentation, BaseTransform


class LifeguardDataset_3Dseq(Dataset):

    def __init__(self,
                 cfg,
                 mode='train',
                 ):
        
        self.T_len = cfg['T_len']
        self.S_len = cfg['S_len']
        self.T_distance = cfg['T_distance']
        self.S_distance = cfg['S_distance']
        self.sequence_distance = cfg['sequence_distance']
        self.mode = mode
        if self.mode == 'train':
            self.transform = Augmentation(
                img_size=cfg['train_size'],
                jitter=cfg['jitter'],
                hue=cfg['hue'],
                saturation=cfg['saturation'],
                exposure=cfg['exposure']
            )
        elif self.mode == 'test':
            self.transform = BaseTransform(
                img_size=cfg['test_size'],
            )


        self.num_classes = 2
        self._load_data(cfg, mode)
    
    def _load_data(self, cfg, mode):
        self.keyframe_seq_indices = []
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        if mode == 'train':
            self.splitlist_file = os.path.join(current_file_directory, 'splitlists', cfg['train_splitlist'])
            self.img_size = cfg['train_size']
            
            with open(self.splitlist_file, 'r') as file:
                self.snippets = [line.strip().split(',')[0] for line in file]

            for snippet in self.snippets:
                num_frames = FileLoader.get_num_frames(snippet)
                keyframe = (self.T_len * self.T_distance) - (self.T_distance-1)
                while keyframe + (self.S_len-1)*self.S_distance <= num_frames:   
                    self.keyframe_seq_indices.append([snippet, list(range(keyframe, keyframe+self.S_len*self.S_distance, self.S_distance))])
                    keyframe += self.sequence_distance

        elif mode == 'test':
            self.splitlist_file = os.path.join(current_file_directory, 'splitlists', cfg['test_splitlist'])
            self.img_size = cfg['test_size']

            with open(self.splitlist_file, 'r') as file:
                self.snippets = [line.strip().split(',')[0] for line in file]

            self.keyframe_indices = []
            for snippet in self.snippets:
                num_frames = FileLoader.get_num_frames(snippet)
                keyframe = (self.T_len * self.T_distance) - (self.T_distance-1)
                while keyframe <= num_frames:   
                    self.keyframe_indices.append([snippet, keyframe])
                    keyframe += self.S_distance
                self.keyframe_seq_indices = [[snippet, [keyframe]] for [snippet, keyframe] in self.keyframe_indices]

    def __len__(self):
        return len(self.keyframe_seq_indices)

    def __getitem__(self, index):
        # load a data
        frame_idx, video_clip, target = self.pull_item(index)

        return frame_idx, video_clip, target

    # video_clip: first 0-1 PIL values then via transform 0-255 values and torch.tensor
    # target: first transformed from list to np.array. then in transform from np.array and pixel-location to torch.tensor and 0-1 values
    def pull_item(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        snippet, keyframe_indices = self.keyframe_seq_indices[index]
        #print('==============================')
        #print('snippet: {}'.format(snippet))
        #print('keyframe_indices: {}'.format(keyframe_indices))

        ids = []
        target_seq = []
        video_clip = []
        # additional dimension S, sequence_position
        for keyframe in keyframe_indices:

            # get seq and images
            # +self.T_distance, because then we get 2,4,6,8 instead 0,2,4,6 for range(0,8,2), keyfram=8, T_len=4, T_distance=2
            seq = list(range(keyframe - self.T_len*self.T_distance + self.T_distance, keyframe+self.T_distance, self.T_distance))
            video_clip.extend(FileLoader.get_video_clip(snippet, seq))
            ow, oh = self.img_size, self.img_size

            # get boxes and labels: target: [N, 4 + 1]
            box_and_label_list = FileLoader.get_box_and_label_list(snippet, keyframe)
            if box_and_label_list == []:
                target_seq.append(None)
            else:
                target_seq.append(np.array(box_and_label_list))

            ids.append([snippet, keyframe])


        # transform
        video_clip, target_seq = self.transform(video_clip, target_seq)
        # List [T, 3, H, W] -> List[S, List [T, 3, H, W]]. needed the other format for transform
        video_clip_seq = [video_clip[i:i + self.T_len] for i in range(0, len(video_clip), self.T_len)]

        # List[S, List [T, 3, H, W]] -> List[S,[3, T, H, W]]
        video_clip_seq = [torch.stack(video_clip, dim=1) for video_clip in video_clip_seq]

        # reformat target_seq
        target_seq = [
            {
                'boxes': torch.empty(0, 4),  # [N, 4]
                'labels': torch.empty(0),  # [N,]
                'orig_size': [ow, oh],
                'video_idx': snippet
            } if target.nelement() == 0 else {
                'boxes': target[:, :4].float(),  # [N, 4]
                'labels': target[:, -1].long(),  # [N,]
                'orig_size': [ow, oh],
                'video_idx': snippet
            }
            for target in target_seq
        ]

        video_clip_seq = torch.stack(video_clip_seq)
        return ids, video_clip_seq, target_seq # [S, 3, T, H, W]
