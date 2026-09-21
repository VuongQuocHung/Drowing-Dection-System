config = {

    # FUNDAMENTAL
    'mode': 'reference', # 'reference', '2D3Dseq', '2Dseq', '3Dseq'
    'freeze_backbone_2d': False,
    'freeze_backbone_3d': False,
    'vis_thresh': 0.75, # default: 0.4
    'batch_size': 16,   # default for train.py: 16, default for eval.py: 8, maybe later need to split into ||  16!!
    'batch_size_inference': 1,    # for rnn needs to be 1 because due to test.py data for test.py and eval.py is not loaded with Dataset_Seq, but Dataset -> no sequences are loaded, which means loaded data are from different t. hidden_state contains hidden_state for every sequence in batch. when squence have different lengths, probably the hidden_state needs to be complicated transformed
    # IMPORTANT
    'T_len': 32,             # not for 2Dseq
    'T_distance': 6,         # not for 2Dseq
    'S_len': 5,              # seq
    'S_distance': 6,         # 
    'sequence_distance': 6,  # seq
    'layers': 3,             # seq
    # DATA
    'fold': 3,
    'version': 'yowo_v2_nano',
    'epoch_weight': 5,
    # ARGS-EVALUATION-train.py
    'eval': True,
    'eval_first': False,






    # PATHS AND FILES
    # do not change the following values. they will be generated from the 3 previous values. in case the following hard coded values are needed, comment out some lines in   def build_default_config
    'train_splitlist': '8_train.csv',
    'test_splitlist': '8_test.csv',
    'test_save_path': './temp/default_config/8/',    # './temp/det_results/'
    'eval_save_path': './temp/default_config/8/',   # './temp/eval_results/'
    'save_folder': './temp/default_config/8/',   # './temp/weights/'
    'weight': 'yowo_v2_nano_epoch_7.pth', # './temp/weights/yowo_v2_nano/yowo_v2_nano_epoch_7.pth'

    # ARGS-MODEL-all.py
    'conf_thresh': 0.1,
    'nms_thresh': 0.5,
    'topk': 40,
    'resume': None,

    # ARGS-CUDA-all.py
    'cuda': False,

    # ARGS-BASIC-test.py
    'show': True, # default: False
    'save': False,
    'start_index': 0,

    # ARGS-VISUALIZATION-train.py
    'tfboard': False,

    # ARGS-BATCHSIZE-train.py
    'accumulate': 1,
    'base_lr': 0.0001,
    'lr_decay_ratio': 0.5,
    # ARGS-EPOCH-train.py
    'max_epoch': 10,
    'lr_epoch': [2,3,4],
    # ARGS-DATASET-train.py
    'num_workers': 4,  # default: 4
    # ARGS-MATCHER-train.py
    'center_sampling_radius': 2.5,
    'topk_candicate': 10,
    # ARGS-LOSS-train.py
    'loss_conf_weight': 1,
    'loss_cls_weight': 1,
    'loss_reg_weight': 5,
    'focal_loss': False,
    # ARGS-DPP-train.py
    'distributed': False,
    'dist_url': 'env://',
    'world_size': 1,
    'sybn': False,

    # input size
    'train_size': 224,
    'test_size': 224,
    # transform
    'jitter': 0.2,
    'hue': 0.1,
    'saturation': 1.5,
    'exposure': 1.5,
    # cls label
    'multi_hot': False,  # one hot
    # optimizer
    'optimizer': 'adamw',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    # warmup strategy
    'warmup': 'linear',
    'warmup_factor': 0.00066667,
    'wp_iter': 500,
    # class names
    'valid_num_classes': 2,
    'label_map': (
                'swim',     'drown'
            ),
}
