import torch
from thop import profile


def FLOPs_and_Params(model, img_size, T_len, mode, device):
    # generate init video clip
    if mode == 'reference':
        video_clip = torch.randn(1, 3, T_len, img_size, img_size).to(device)
    elif mode == '2Dseq':
        video_clip = torch.randn(1, 1, 3, img_size, img_size).to(device)
    elif mode == '2D3Dseq':
        video_clip = torch.randn(1, 1, 3, T_len, img_size, img_size).to(device)
    elif mode == '3Dseq':
        video_clip = torch.randn(1, 1, 3, T_len, img_size, img_size).to(device)
    

    # set eval mode
    model.trainable = False
    model.eval()

    print('==============================')
    flops, params = profile(model, inputs=(video_clip, ))
    print('==============================')
    FLOPs = 'FLOPs : {:.2f} G'.format(flops / 1e9)
    Params = 'Params : {:.2f} M'.format(params / 1e6)
    print(FLOPs)
    print(Params)
    
    # set train mode.
    model.trainable = True
    model.train()

    return FLOPs, Params

if __name__ == "__main__":
    pass
