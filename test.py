import argparse
import cv2
import os
import time
import numpy as np
import torch


from dataset_utils.LifeguardDataset import LifeguardDataset
from dataset_utils.LifeguardDataset_2Dseq import LifeguardDataset_2Dseq
from dataset_utils.LifeguardDataset_3Dseq import LifeguardDataset_3Dseq

from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import convert_tensor_to_cv2img, vis_detection

from config import build_default_config, build_model_config
from models import build_model

from utils.misc import parse_args


    
@torch.no_grad()
def inference_lifeguard(d_cfg, args, model, device, dataset, class_names=None, class_colors=None):
    # path to save 
    if d_cfg['save']:
        save_path = os.path.join(
            d_cfg['test_save_path'], 'video_clips')
        os.makedirs(save_path, exist_ok=True)

    # inference
    # CHANGES ########################################################################
    if d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='3Dseq':
        hidden_state = None
        old_snippet = None
    elif d_cfg['mode']=='2Dseq':
        hidden_state = [[None,None,None],[None,None,None]]
        old_snippet = None
    # CHANGES ########################################################################
    for index in range(d_cfg['start_index'], len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        frame_id, video_clip, target = dataset[index]
        

        # prepare
        video_clip = video_clip.unsqueeze(0) # [B, 3, T, H, W], B=1   add batch dimension  # new: [B, S, 3, H, W]

        # CHANGES ########################################################################
        if d_cfg['mode']=='reference':
            orig_size = target['orig_size']  # width, height
            video_clip = video_clip.to(device)
        elif d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='2Dseq' or d_cfg['mode']=='3Dseq':
            orig_size = target[0]['orig_size']  # width, height
            video_clip = torch.transpose(video_clip,0,1).to(device)  # [B, S, ...]  -> [S, B, ...] (now format like after collate_fn, can feed this into model)
        else:
            raise NotImplementedError
        # CHANGES ########################################################################



        t0 = time.time()
        # CHANGES ########################################################################
        # inference
        if d_cfg['mode']=='reference':
            batch_scores, batch_labels, batch_bboxes = model(video_clip)
        elif d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='3Dseq':
            snippet, keyframe = frame_id[0]
            if snippet != old_snippet:
                    hidden_state = None
            old_snippet = snippet
            # the result will be of same dim because our frame_seq is a video_clip of len 1
            batch_scores, batch_labels, batch_bboxes, hidden_state = model(video_clip, hidden_state)
        elif d_cfg['mode']=='2Dseq':
            snippet, keyframe = frame_id[0]
            if snippet != old_snippet:
                    hidden_state = [[None,None,None],[None,None,None]]
            old_snippet = snippet
            # the result will be of same dim because our frame_seq is a video_clip of len 1
            batch_scores, batch_labels, batch_bboxes, hidden_state = model(video_clip, hidden_state)
        else:
            raise NotImplementedError
        # CHANGES ########################################################################
        print("inference time ", time.time() - t0, "s")

        # batch size = 1
        scores = batch_scores[0]
        labels = batch_labels[0]
        bboxes = batch_bboxes[0]
        
        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        
        # CHANGES ########################################################################
        if d_cfg['mode']=='reference' or d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='3Dseq':
            # old format after collate fn: [B, 3, T, H, W]
            key_frame_tensor = video_clip[0, :, -1, :, :]
        elif d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='3Dseq':
            # reminder our seq is of len 1, remove dimension S
            video_clip = video_clip[0]
            key_frame_tensor = video_clip[0, :, -1, :, :]
        elif d_cfg['mode']=='2Dseq':
            # new format after collate fn: [S, B, 3, H, W]
            # needs to be tranformed to old format [B, 3, T, H, W]
            # in model, this is handled by differend forward function (iterating over S instead taking at T=-1)
            video_clip = video_clip.permute(1, 2, 0, 3, 4)
            key_frame_tensor = video_clip[0, :, -1, :, :]
        else:
            raise NotImplementedError
        # CHANGES ########################################################################
        key_frame = convert_tensor_to_cv2img(key_frame_tensor)

        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        # visualize detection
        vis_results = vis_detection(
            frame=key_frame,
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            vis_thresh=d_cfg['vis_thresh'],
            class_names=class_names,
            class_colors=class_colors
            )

        if d_cfg['show']:
            cv2.imshow('key-frame detection', vis_results)
            cv2.waitKey(0)

        if d_cfg['save']:
            # save result
            cv2.imwrite(os.path.join(save_path,
            '{:0>5}_{}.jpg'.format(index,str(frame_id))), vis_results)
        



if __name__ == '__main__':
    args = parse_args()

    # config
    d_cfg = build_default_config(args)
    m_cfg = build_model_config(args)

    # cuda
    if d_cfg['cuda']:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # CHANGES ########################################################################
    # dataset
    if d_cfg['mode']=='reference':
        dataset = LifeguardDataset(
            cfg=d_cfg,
            mode='test'
            )
    elif d_cfg['mode']=='2D3Dseq' or d_cfg['mode']=='3Dseq':
        dataset = LifeguardDataset_3Dseq(
            cfg=d_cfg,
            mode='test'
            )
    elif d_cfg['mode']=='2Dseq':
        dataset = LifeguardDataset_2Dseq(
            cfg=d_cfg,
            mode='test'
            )
    else:
        raise NotImplementedError
    # CHANGES ########################################################################
    class_names = d_cfg['label_map']
    num_classes = dataset.num_classes


    np.random.seed(100)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=os.path.join(d_cfg['save_folder'],d_cfg['weight']))

    # to eval
    model = model.to(device).eval()

    # run
    inference_lifeguard(
        d_cfg=d_cfg,
        args=args,
        model=model,
        device=device,
        dataset=dataset,
        class_names=class_names,
        class_colors=class_colors
    )
