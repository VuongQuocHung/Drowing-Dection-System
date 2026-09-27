import argparse
import csv
import glob
import subprocess
from pathlib import Path

import cv2
from yt_dlp import YoutubeDL

from dataset_utils.LifeguardUtils import LifeguardUtils


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("generalization_candidates.csv")
DEFAULT_SOURCE_DIR = REPO_ROOT / "videos" / "training_candidates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "videos" / "annotation_candidates"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create 224x224/30 FPS clips ready for manual CVAT annotation."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--video-id",
        action="append",
        help="Only prepare this video ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download a missing source video from YouTube by its manifest ID.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_source(source_dir, video_id, download_missing=False):
    matches = [Path(path) for path in glob.glob(str(source_dir / f"*{video_id}*"))]
    if not matches and download_missing:
        source_dir.mkdir(parents=True, exist_ok=True)
        options = {
            "format": "bv*[height<=720][ext=mp4]/bv*[height<=720]/bestvideo",
            "outtmpl": str(source_dir / "%(id)s_%(title)s.%(ext)s"),
        }
        with YoutubeDL(options) as downloader:
            downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
        matches = [
            Path(path)
            for path in glob.glob(str(source_dir / f"*{video_id}*"))
        ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one source containing {video_id!r} in "
            f"{source_dir}, found {len(matches)}."
        )
    return matches[0]


def video_size(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid source dimensions for {path}")
    return width, height


def prepare_candidate(
        row, source_dir, output_dir, overwrite, download_missing=False):
    video_id = row["video_id"]
    source = find_source(source_dir, video_id, download_missing)
    frame_width, frame_height = video_size(source)

    x_px = round(float(row["x"]) * frame_width)
    y_px = round(float(row["y"]) * frame_height)
    crop_size = round(float(row["width"]) * frame_height)
    if (
        crop_size <= 0
        or x_px < 0
        or y_px < 0
        or x_px + crop_size > frame_width
        or y_px + crop_size > frame_height
    ):
        raise ValueError(
            f"Candidate {video_id} has an out-of-frame crop: "
            f"x={x_px}, y={y_px}, size={crop_size}, "
            f"frame={frame_width}x{frame_height}"
        )

    snippet = LifeguardUtils.snippet_name(
        video_id,
        row["start"],
        row["end"],
        row["x"],
        row["y"],
        row["width"],
    )
    output = output_dir / f"{snippet}.mp4"
    if output.exists() and not overwrite:
        print(f"Exists, skipping: {output}")
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ss",
        row["start"],
        "-to",
        row["end"],
        "-vf",
        f"crop={crop_size}:{crop_size}:{x_px}:{y_px},scale=224:224,fps=30",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(f"Created [{row['kind']}]: {output}")
    print(
        "  datalist row after reviewed XML is ready: "
        + ",".join(
            [
                video_id,
                row["start"],
                row["end"],
                row["x"],
                row["y"],
                row["width"],
                "y",
            ]
        )
    )
    return output


def main():
    args = parse_args()
    selected_ids = set(args.video_id or [])
    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if selected_ids:
        rows = [row for row in rows if row["video_id"] in selected_ids]
    if not rows:
        raise RuntimeError("No matching annotation candidates were found.")

    for row in rows:
        prepare_candidate(
            row,
            args.source_dir,
            args.output_dir,
            args.overwrite,
            args.download_missing,
        )


if __name__ == "__main__":
    main()
