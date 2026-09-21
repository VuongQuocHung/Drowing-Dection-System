import os
import torch
import time

from dataset_utils.LifeguardDataset_2Dseq import LifeguardDataset_2Dseq
from utils.box_ops import rescale_bboxes
from utils.misc_eval import add_entry

from .cal_frame_mAP import evaluate_frameAP


class Lifeguard_Evaluator_2Dseq(object):
    def __init__(self,
                 d_cfg,
                 iou_thresh=0.5,
                 collate_fn=None):
        self.d_cfg = d_cfg
        self.model_name = d_cfg['version']
        self.img_size = d_cfg['test_size']
        self.T_len = d_cfg['T_len']
        self.batch_size = d_cfg['batch_size_inference']
        self.save_path = d_cfg['eval_save_path']
        self.weight_name, _ = os.path.splitext(d_cfg['weight'])
        self.iou_thresh = iou_thresh
        self.collate_fn = collate_fn

        # CHANGES ########################################################################
        # dataset
        self.testset = LifeguardDataset_2Dseq(
            cfg=d_cfg,
            mode='test')
        self.num_classes = self.testset.num_classes
        # CHANGES ########################################################################




    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):
        print("Metric: Frame mAP")
        inference_time_done = False
        # dataloader
        self.testloader = torch.utils.data.DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )

        epoch_size = len(self.testloader)

        det_boxes = []
        # CHANGES ########################################################################
        # inference
        hidden_state = [[None,None,None],[None,None,None]]
        old_snippet = None
        # CHANGES ########################################################################
        for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(self.testloader):
                # CHANGES ########################################################################
                batch_frame_id=batch_frame_id[0] # eliminate sequence dimension
                batch_target=batch_target[0] # eliminate sequence dimension
                snippet, keyframe = batch_frame_id[0] # eliminate batch dimension, only one batch
                if snippet != old_snippet:
                        hidden_state = [[None,None,None],[None,None,None]]
                old_snippet = snippet
                # CHANGES ########################################################################
                # to device
                batch_video_clip = batch_video_clip.to(model.device)

                with torch.no_grad():
                    # CHANGES ########################################################################
                    # inference
                    if inference_time_done is False:
                        start_time = time.time()
                        batch_scores, batch_labels, batch_bboxes, hidden_state = model(batch_video_clip, hidden_state)
                        inference_time = time.time() - start_time
                        add_entry('inference_time', inference_time, self.save_path)
                        inference_time_done = True
                    else:
                        batch_scores, batch_labels, batch_bboxes, hidden_state = model(batch_video_clip, hidden_state)

                    # CHANGES ########################################################################

                    # process batch
                    for bi in range(len(batch_scores)):
                        snippet = batch_frame_id[bi][0]
                        keyframe = batch_frame_id[bi][1]

                        scores = batch_scores[bi]
                        labels = batch_labels[bi]
                        bboxes = batch_bboxes[bi]
                        target = batch_target[bi]

                        # rescale bbox
                        orig_size = target['orig_size']
                        bboxes = rescale_bboxes(bboxes, orig_size)

                        for score, label, bbox in zip(scores, labels, bboxes):
                            xtl = round(bbox[0])
                            ytl = round(bbox[1])
                            xbr = round(bbox[2])
                            ybr = round(bbox[3])
                            cls_id = int(label)

                            det_boxes.append([snippet, keyframe, xtl, ytl, xbr, ybr, cls_id, score])
                    if iter_i % 100 == 0:
                        log_info = "[%d / %d]" % (iter_i, epoch_size)
                        print(log_info, flush=True)

        print('calculating Frame mAP ...')
        metric_list = evaluate_frameAP(self.testset.keyframe_indices, det_boxes, self.weight_name, self.iou_thresh,
                                       self.save_path, show_pr_curve)
        for metric in metric_list:
            print(metric)


if __name__ == "__main__":
    pass
