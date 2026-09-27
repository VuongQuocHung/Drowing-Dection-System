import argparse
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from config.YOWOv2_tiny import config as yowo_tiny_config
from config.yowo_v2_config import yowo_v2_config
from models import build_model


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT = (
    REPO_ROOT
    / "YOWOv2_tiny_fold3_best"
    / "yowo_v2_tiny_epoch_9.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOWOv2-tiny drowning detection on a video."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "result.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=DEFAULT_WEIGHT,
        help="Checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum score drawn on the output video.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video while inference is running.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Optional pool region in source-video pixels.",
    )
    return parser.parse_args()


def select_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access the NVIDIA GPU."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def resolve_roi(roi, frame_width, frame_height):
    if roi is None:
        return 0, 0, frame_width, frame_height

    x, y, width, height = roi
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive.")
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError(
            f"ROI {roi} is outside the {frame_width}x{frame_height} video frame."
        )
    return x, y, width, height


def prepare_frame(frame, roi, image_size):
    x, y, width, height = roi
    cropped = frame[y : y + height, x : x + width]
    resized = cv2.resize(cropped, (image_size, image_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def make_clip(sampled_frames, device):
    frames = [
        torch.from_numpy(frame).permute(2, 0, 1).float()
        for frame in sampled_frames
    ]
    # [T, C, H, W] -> [1, C, T, H, W]
    return torch.stack(frames, dim=1).unsqueeze(0).to(device)


def draw_detections(frame, detections, roi, class_names, confidence):
    x_offset, y_offset, roi_width, roi_height = roi
    colors = {
        0: (0, 220, 0),
        1: (0, 0, 255),
    }
    drowning_detected = False

    for score, label, box in detections:
        if score < confidence:
            continue

        label = int(label)
        x1 = int(x_offset + box[0] * roi_width)
        y1 = int(y_offset + box[1] * roi_height)
        x2 = int(x_offset + box[2] * roi_width)
        y2 = int(y_offset + box[3] * roi_height)
        color = colors.get(label, (255, 180, 0))
        name = class_names[label] if label < len(class_names) else str(label)
        text = f"{name}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_y = max(y1, text_size[1] + baseline + 4)
        cv2.rectangle(
            frame,
            (x1, text_y - text_size[1] - baseline - 4),
            (x1 + text_size[0] + 4, text_y),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 2, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        drowning_detected = drowning_detected or label == 1

    if drowning_detected:
        message = "WARNING: DROWNING DETECTED"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 190), -1)
        cv2.putText(
            frame,
            message,
            (12, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    weight_path = args.weight.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1.")

    device = select_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    d_cfg = deepcopy(yowo_tiny_config)
    d_cfg["conf_thresh"] = min(d_cfg["conf_thresh"], args.confidence)
    model_config = yowo_v2_config[d_cfg["version"]]

    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=model_config,
        device=device,
        num_classes=len(d_cfg["label_map"]),
        trainable=False,
    )
    load_checkpoint(model, weight_path)
    model = model.to(device).eval()
    print(f"Checkpoint: {weight_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps):
        fps = 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = resolve_roi(args.roi, frame_width, frame_height)

    # Training used every sixth frame from 30 FPS source clips (one sample/0.2 s).
    sample_period = max(1, round(fps * d_cfg["T_distance"] / 30.0))
    temporal_length = d_cfg["T_len"]
    sampled_frames = []
    latest_detections = []
    inference_windows = 0
    visible_predictions = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV cannot create video: {output_path}")

    print(
        f"Video: {frame_width}x{frame_height}, {fps:.2f} FPS, "
        f"{frame_count if frame_count > 0 else '?'} frames"
    )
    print(
        f"Temporal input: {temporal_length} sampled frames, "
        f"one sample every {sample_period} source frames"
    )
    if args.roi is not None:
        print(f"ROI: {roi}")

    started_at = time.perf_counter()
    frame_index = 0
    stopped_by_user = False

    if args.show:
        cv2.namedWindow("YOWOv2 Drowning Detection", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_started_at = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % sample_period == 0:
                sampled_frames.append(
                    prepare_frame(frame, roi, d_cfg["test_size"])
                )
                if len(sampled_frames) > temporal_length:
                    sampled_frames.pop(0)

                if len(sampled_frames) == temporal_length:
                    clip = make_clip(sampled_frames, device)
                    with torch.inference_mode():
                        batch_scores, batch_labels, batch_boxes = model(clip)

                    latest_detections = list(
                        zip(
                            batch_scores[0],
                            batch_labels[0],
                            batch_boxes[0],
                        )
                    )
                    inference_windows += 1
                    for score, label, _ in latest_detections:
                        if score >= args.confidence:
                            visible_predictions[d_cfg["label_map"][int(label)]] += 1

            draw_detections(
                frame,
                latest_detections,
                roi,
                d_cfg["label_map"],
                args.confidence,
            )

            if len(sampled_frames) < temporal_length:
                cv2.putText(
                    frame,
                    f"Collecting temporal context: {len(sampled_frames)}/{temporal_length}",
                    (12, frame_height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_index += 1

            if args.show:
                cv2.imshow("YOWOv2 Drowning Detection", frame)
                processing_ms = (time.perf_counter() - frame_started_at) * 1000.0
                wait_ms = max(1, round(1000.0 / fps - processing_ms))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    stopped_by_user = True
                    break

            if frame_index % 300 == 0:
                elapsed = time.perf_counter() - started_at
                progress = (
                    f"{100.0 * frame_index / frame_count:.1f}%"
                    if frame_count > 0
                    else f"{frame_index} frames"
                )
                print(f"Progress: {progress}, elapsed: {elapsed:.1f}s")
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    if stopped_by_user:
        print("Stopped by user (Q/Esc).")
    print(f"Finished: {frame_index} frames in {elapsed:.1f}s")
    print(f"Inference windows: {inference_windows}")
    print(
        "Visible predictions: "
        + ", ".join(
            f"{name}={visible_predictions.get(name, 0)}"
            for name in d_cfg["label_map"]
        )
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
