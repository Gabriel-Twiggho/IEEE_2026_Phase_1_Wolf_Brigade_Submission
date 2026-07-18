from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .progress import ProgressBar


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


def read_video_info(video: Path) -> VideoInfo:
    if _is_git_lfs_pointer(video):
        raise RuntimeError(
            f"{video} is a Git LFS pointer, not the actual video. "
            "Install Git LFS and run `git lfs pull` from the repository clone."
        )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    try:
        return VideoInfo(
            path=video,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 62.5),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def extract_frame_cache(
    video: Path,
    frame_indices: Iterable[int],
    cache_dir: Path,
    png_compression: int = 3,
) -> dict[int, Path]:
    requested = sorted({int(index) for index in frame_indices if int(index) >= 0})
    if not requested:
        return {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[int, Path] = {}
    missing: list[int] = []
    for frame_index in requested:
        path = cache_dir / f"frame_{frame_index:06d}.png"
        cached[frame_index] = path
        if not path.exists():
            missing.append(frame_index)

    if not missing:
        print(f"[mission] shared frame cache: using {len(cached)} cached frames from {cache_dir}")
        return cached

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    print(f"[mission] shared frame cache: extracting {len(missing)} frames to {cache_dir}")
    progress = ProgressBar("shared frame extraction", len(missing))
    try:
        for progress_index, frame_index in enumerate(missing, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(
                    str(cached[frame_index]),
                    frame,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)],
                )
            progress.update(progress_index)
    finally:
        cap.release()
        progress.finish()

    return cached


def extract_frame_memory_cache(
    video: Path,
    frame_indices: Iterable[int],
) -> dict[int, np.ndarray]:
    requested = sorted({int(index) for index in frame_indices if int(index) >= 0})
    if not requested:
        return {}

    requested_set = set(requested)
    last_requested = requested[-1]
    cached: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    print(f"[mission] shared frame cache: loading {len(requested)} frames into memory")
    progress = ProgressBar("shared frame loading", len(requested))
    try:
        frame_index = -1
        while frame_index < last_requested:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index not in requested_set:
                continue
            cached[frame_index] = frame
            progress.update(len(cached))
            if len(cached) == len(requested):
                break
    finally:
        cap.release()
        progress.finish()

    if len(cached) != len(requested):
        print(
            "[WARN] shared frame cache: "
            f"loaded {len(cached)}/{len(requested)} requested frames from {video}"
        )
    return cached


def read_cached_frame(frame_cache: dict[int, Path | np.ndarray] | None, frame_index: int) -> object | None:
    if not frame_cache:
        return None
    cached = frame_cache.get(int(frame_index))
    if cached is None:
        return None
    if isinstance(cached, np.ndarray):
        return cached.copy()
    if not cached.exists():
        return None
    return cv2.imread(str(cached), cv2.IMREAD_COLOR)
