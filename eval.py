import argparse
import torch
import os

from evaluator.lifeguard_evaluator import Lifeguard_Evaluator
from evaluator.lifeguard_evaluator_2Dseq import Lifeguard_Evaluator_2Dseq
from evaluator.lifeguard_evaluator_3Dseq import Lifeguard_Evaluator_3Dseq

from utils.misc import load_weight, CollateFunc_seq, CollateFunc

from config import build_default_config, build_model_config
from models import build_model

from utils.misc import parse_args


def lifeguard_eval(args, d_cfg, model):
    # CHANGES ########################################################################
    if d_cfg['mode']=='reference':
        evaluator = Lifeguard_Evaluator(
            d_cfg=d_cfg,
            iou_thresh=0.5, # default 0.5
            collate_fn=CollateFunc(),
        )
    elif d_cfg['mode']=='2D3Dseq':
        evaluator = Lifeguard_Evaluator_3Dseq(
            d_cfg=d_cfg,
            iou_thresh=0.5, # default 0.5
            collate_fn=CollateFunc_seq(),
        )
    elif d_cfg['mode']=='2Dseq':
        evaluator = Lifeguard_Evaluator_2Dseq(
            d_cfg=d_cfg,
            iou_thresh=0.5, # default 0.5
            collate_fn=CollateFunc_seq(),
        )
    elif d_cfg['mode']=='3Dseq':
        evaluator = Lifeguard_Evaluator_3Dseq(
            d_cfg=d_cfg,
            iou_thresh=0.5, # default 0.5
            collate_fn=CollateFunc_seq(),
        )
    else:
        raise NotImplementedError
    # CHANGES ########################################################################
    # evaluate
    evaluator.evaluate_frame_map(model, show_pr_curve=False) # default: True


if __name__ == '__main__':
    args = parse_args()
    num_classes = 2

    # config
    d_cfg = build_default_config(args)
    m_cfg = build_model_config(args)

    # cuda
    if d_cfg['cuda']:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    


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
    lifeguard_eval(
        args=args,
        d_cfg=d_cfg,
        model=model,
        )

