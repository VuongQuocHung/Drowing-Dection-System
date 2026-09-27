import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image




# only need to change for target, because format changed only from     List [T, 3, H, W] to     List [S, 3, H, W],     see LifeguardDataset_2Dseq

class Augmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5,
                 exposure=1.5, zoom_out_prob=0.0, zoom_out_min_scale=0.5,
                 zoom_out_fill=114):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.zoom_out_prob = zoom_out_prob
        self.zoom_out_min_scale = zoom_out_min_scale
        self.zoom_out_fill = zoom_out_fill

        if not 0.0 <= self.zoom_out_prob <= 1.0:
            raise ValueError("zoom_out_prob must be between 0 and 1")
        if not 0.0 < self.zoom_out_min_scale <= 1.0:
            raise ValueError("zoom_out_min_scale must be in (0, 1]")


    def rand_scale(self, s):
        scale = random.uniform(1, s)

        if random.randint(0, 1): 
            return scale

        return 1./scale


    def random_distort_image(self, video_clip):
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self.rand_scale(self.saturation)
        dexp = self.rand_scale(self.exposure)
        
        video_clip_ = []
        for image in video_clip:
            image = image.convert('HSV')
            cs = list(image.split())
            cs[1] = cs[1].point(lambda i: i * dsat)
            cs[2] = cs[2].point(lambda i: i * dexp)
            
            def change_hue(x):
                x += dhue * 255
                if x > 255:
                    x -= 255
                if x < 0:
                    x += 255
                return x

            cs[0] = cs[0].point(change_hue)
            image = Image.merge(image.mode, tuple(cs))

            image = image.convert('RGB')

            video_clip_.append(image)

        return video_clip_


    def random_crop(self, video_clip, width, height):
        dw =int(width * self.jitter)
        dh =int(height * self.jitter)

        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)

        swidth =  width - pleft - pright
        sheight = height - ptop - pbot

        sx = float(swidth)  / width
        sy = float(sheight) / height
        
        dx = (float(pleft) / width)/sx
        dy = (float(ptop) / height)/sy

        # random crop
        cropped_clip = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in video_clip]

        return cropped_clip, dx, dy, sx, sy


    def random_zoom_out(self, video_clip):
        """Shrink a whole temporal clip with one shared scale and offset.

        The same geometry is used for every frame so motion remains coherent.
        Returned offsets and scale are normalized to the output canvas and are
        also applied to every target belonging to the clip.
        """
        if random.random() >= self.zoom_out_prob:
            return video_clip, None

        scale = random.uniform(self.zoom_out_min_scale, 1.0)
        resized_size = max(1, min(self.img_size, round(self.img_size * scale)))
        max_offset = self.img_size - resized_size
        left = random.randint(0, max_offset)
        top = random.randint(0, max_offset)

        zoomed_clip = []
        for image in video_clip:
            resized = image.resize((resized_size, resized_size), Image.BILINEAR)
            canvas = Image.new(
                'RGB',
                (self.img_size, self.img_size),
                color=(self.zoom_out_fill,) * 3,
            )
            canvas.paste(resized, (left, top))
            zoomed_clip.append(canvas)

        actual_scale = resized_size / self.img_size
        return zoomed_clip, (
            actual_scale,
            left / self.img_size,
            top / self.img_size,
        )


    def apply_zoom_out_bbox(self, target, zoom_params):
        if zoom_params is None or target.size == 0:
            return target

        scale, offset_x, offset_y = zoom_params
        target[..., [0, 2]] = target[..., [0, 2]] * scale + offset_x
        target[..., [1, 3]] = target[..., [1, 3]] * scale + offset_y
        return target


    def apply_bbox(self, target, ow, oh, dx, dy, sx, sy):
        sx, sy = 1./sx, 1./sy
        # apply deltas on bbox
        target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx)) 
        target[..., 1] = np.minimum(0.999, np.maximum(0, target[..., 1] / oh * sy - dy)) 
        target[..., 2] = np.minimum(0.999, np.maximum(0, target[..., 2] / ow * sx - dx)) 
        target[..., 3] = np.minimum(0.999, np.maximum(0, target[..., 3] / oh * sy - dy)) 

        # refine target
        refine_target = []
        for i in range(target.shape[0]):
            tgt = target[i]
            bw = (tgt[2] - tgt[0]) * ow
            bh = (tgt[3] - tgt[1]) * oh

            if bw < 1. or bh < 1.:
                continue
            
            refine_target.append(tgt)

        refine_target = np.array(refine_target).reshape(-1, target.shape[-1])

        return refine_target
        

    def to_tensor(self, video_clip):
        return [F.to_tensor(image) * 255. for image in video_clip]


    def __call__(self, video_clip, target):
        # Initialize Random Variables
        oh = video_clip[0].height  
        ow = video_clip[0].width
        
        # random crop
        video_clip, dx, dy, sx, sy = self.random_crop(video_clip, ow, oh)

        # resize
        video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]

        # Zoom-out parameters are shared by the complete temporal clip.
        video_clip, zoom_params = self.random_zoom_out(video_clip)

        # random flip
        flip = random.randint(0, 1)
        if flip:
            video_clip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in video_clip]

        # distort
        video_clip = self.random_distort_image(video_clip)


        if isinstance(target, list):
            # process target [CHANGED]
            for i in range(len(target)):
                if target[i] is not None:
                    target[i] = self.apply_bbox(target[i], ow, oh, dx, dy, sx, sy)
                    target[i] = self.apply_zoom_out_bbox(target[i], zoom_params)
                    if flip:
                        target[i][..., [0, 2]] = 1.0 - target[i][..., [2, 0]]
                else:
                    target[i] = np.array([])

                
            # to tensor [CHANGED]
            video_clip = self.to_tensor(video_clip)
            target = [torch.as_tensor(t).float() for t in target]

        else:
            # process target
            if target is not None:
                target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
                target = self.apply_zoom_out_bbox(target, zoom_params)
                if flip:
                    target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
            else:
                target = np.array([])
                
            # to tensor
            video_clip = self.to_tensor(video_clip)
            target = torch.as_tensor(target).float()

        return video_clip, target 





# Transform for Testing
class BaseTransform(object):
    def __init__(self, img_size=224, ):
        self.img_size = img_size


    def to_tensor(self, video_clip):
        return [F.to_tensor(image) * 255. for image in video_clip]


    def __call__(self, video_clip, target=None, normalize=True):
        oh = video_clip[0].height
        ow = video_clip[0].width

        # resize
        video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]

        if isinstance(target, list):
            # normalize target [CHANGED]
            for i in range(len(target)):
                if target[i] is not None:
                    if normalize:
                        target[i][..., [0, 2]] /= ow
                        target[i][..., [1, 3]] /= oh

                else:
                    target[i] = np.array([])

            # to tensor [CHANGED]
            video_clip = self.to_tensor(video_clip)
            target = [torch.as_tensor(t).float() for t in target]

        else:
            # normalize target
            if target is not None:
                if normalize:
                    target[..., [0, 2]] /= ow
                    target[..., [1, 3]] /= oh

            else:
                target = np.array([])

            # to tensor
            video_clip = self.to_tensor(video_clip)
            target = torch.as_tensor(target).float()

        return video_clip, target 
