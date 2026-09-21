import torch
from .yowo import YOWO
from .yowo_2Dseq import YOWO_2Dseq
from .yowo_2D3Dseq import YOWO_2D3Dseq
from .yowo_3Dseq import YOWO_3Dseq
from .loss import build_criterion


# build YOWO detector
def build_yowo(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(d_cfg['version'].upper()))

    # build YOWO
    if d_cfg['mode']=='reference':
        model = YOWO(
            cfg = m_cfg,
            device = device,
            num_classes = num_classes,
            conf_thresh = d_cfg['conf_thresh'],
            nms_thresh = d_cfg['nms_thresh'],
            topk = d_cfg['topk'],
            trainable = trainable,
            multi_hot = d_cfg['multi_hot'],
            )
    elif d_cfg['mode']=='2D3Dseq':    
        model = YOWO_2D3Dseq(
            cfg = m_cfg,
            d_cfg = d_cfg,
            device = device,
            num_classes = num_classes,
            conf_thresh = d_cfg['conf_thresh'],
            nms_thresh = d_cfg['nms_thresh'],
            topk = d_cfg['topk'],
            trainable = trainable,
            multi_hot = d_cfg['multi_hot'],
            )
    elif d_cfg['mode']=='2Dseq':
        model = YOWO_2Dseq(
            cfg = m_cfg,
            d_cfg = d_cfg,
            device = device,
            num_classes = num_classes,
            conf_thresh = d_cfg['conf_thresh'],
            nms_thresh = d_cfg['nms_thresh'],
            topk = d_cfg['topk'],
            trainable = trainable,
            multi_hot = d_cfg['multi_hot'],
            )
    elif d_cfg['mode']=='3Dseq':
        model = YOWO_3Dseq(
            cfg = m_cfg,
            d_cfg = d_cfg,
            device = device,
            num_classes = num_classes,
            conf_thresh = d_cfg['conf_thresh'],
            nms_thresh = d_cfg['nms_thresh'],
            topk = d_cfg['topk'],
            trainable = trainable,
            multi_hot = d_cfg['multi_hot'],
            )
    else:
        raise NotImplementedError

    if trainable:
        # Freeze backbone
        if d_cfg['freeze_backbone_2d']:
            print('Freeze 2D Backbone ...')
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if d_cfg['freeze_backbone_3d']:
            print('Freeze 3D Backbone ...')
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # keep training       
        if resume is not None:
            print('keep training: ', resume)
            checkpoint = torch.load(resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        # build criterion
        criterion = build_criterion(
            d_cfg, args, d_cfg['train_size'], num_classes, d_cfg['multi_hot'])
    
    else:
        criterion = None
                        
    return model, criterion
