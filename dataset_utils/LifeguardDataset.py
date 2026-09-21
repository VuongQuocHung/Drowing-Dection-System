#!/usr/bin/python
# encoding: utf-8
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset_utils.LifeguardUtils import FileLoader
from dataset_utils.transforms import Augmentation, BaseTransform



class LifeguardDataset(Dataset):

    def __init__(self,
                 cfg,
                 mode='train',
                 ):
        
        self.T_len = cfg['T_len']
        self.T_distance = cfg['T_distance']
        self.S_distance = cfg['S_distance']
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
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        if mode == 'train':
            self.splitlist_file = os.path.join(current_file_directory, 'splitlists', cfg['train_splitlist'])
            self.img_size = cfg['train_size']
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

    def __len__(self):
        return len(self.keyframe_indices)

    def __getitem__(self, index):
        # load a data
        frame_idx, video_clip, target = self.pull_item(index)

        return frame_idx, video_clip, target

    # video_clip: first 0-1 PIL values then via transform 0-255 values and torch.tensor
    # target: first transformed from list to np.array. then in transform from np.array and pixel-location to torch.tensor and 0-1 values
    def pull_item(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        snippet, keyframe = self.keyframe_indices[index]
        #print('==============================')
        #print('snippet: {}'.format(snippet))
        #print('keyframe_indices: {}'.format(keyframe))

        # get seq and images
        # +self.T_distance, because then we get 2,4,6,8 instead 0,2,4,6 for range(0,8,2), keyfram=8, T_len=4, T_distance=2
        seq = list(range(keyframe - self.T_len*self.T_distance + self.T_distance, keyframe+self.T_distance, self.T_distance))
        video_clip = FileLoader.get_video_clip(snippet, seq)
        ow, oh = self.img_size, self.img_size

        # get boxes and labels: target: [N, 4 + 1]
        box_and_label_list = FileLoader.get_box_and_label_list(snippet, keyframe)
        if box_and_label_list == []:
            target = None
        else:
            target = np.array(box_and_label_list)

        # transform
        video_clip, target = self.transform(video_clip, target)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)

        # reformat target
        if target.nelement() == 0:
            target = {
                'boxes': torch.empty(0, 4),  # [N, 4]
                'labels': torch.empty(0),  # [N,]
                'orig_size': [ow, oh],
                'video_idx': snippet
            }
        else:
            target = {
                'boxes': target[:, :4].float(),  # [N, 4]
                'labels': target[:, -1].long(),  # [N,]
                'orig_size': [ow, oh],
                'video_idx': snippet
            }
        return [snippet, keyframe], video_clip, target  # [3, T, H, W]
