# scripts/smoke/audit_activitynet_timestamp_alignment.py

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import sequential_loader as sl
import torch
from PIL import Image


FRAMES_PER_CHUNK = 16


@dataclass
class NearestFrame:
    target_time: float
    delta: float = float("inf")
    timestamp: float | None = None
    frame_index: int | None = None
    frame: torch.Tensor | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit alignment between ActivityNet annotation times and "
            "Sequential Loader timestamps."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="ActivityNet root containing json/ and v1-3/",
    )
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="training",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=3,
        help="Number of videos to inspect when --video-id is not specified.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help=(
            "ActivityNet video_id to inspect. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--annotations-per-video",
        type=int,
        default=2,
        help="Number of annotations to inspect per video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/activitynet_timestamp_audit"),
    )
    parser.add_argument(
        "--origin-tolerance",
        type=float,
        default=0.5,
        help="Tolerance in seconds when checking whether timestamps start near 0.",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=float,
        default=0.5,
        help="Maximum acceptable distance between target annotation time and nearest frame.",
    )
    return parser.parse_args()


def load_activitynet_database(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if data.get("version") != "VERSION 1.3":
        raise RuntimeError(
            f"Unexpected ActivityNet version: {data.get('version')!r}"
        )

    database = data.get("database")
    if not isinstance(database, dict):
        raise RuntimeError("ActivityNet JSON has no valid database object")

    return database


def parse_annotations(
    database: dict[str, Any],
    video_id: str,
    split: str,
    limit: int,
) -> list[dict[str, Any]]:
    metadata = database.get(video_id)
    if not isinstance(metadata, dict):
        raise RuntimeError(f"No annotation entry for video_id={video_id!r}")

    actual_subset = metadata.get("subset")
    if actual_subset != split:
        raise RuntimeError(
            f"Subset mismatch for {video_id}: "
            f"expected={split!r}, actual={actual_subset!r}"
        )

    raw_annotations = metadata.get("annotations")
    if not isinstance(raw_annotations, list):
        raise RuntimeError(
            f"Invalid annotations for video_id={video_id!r}"
        )

    parsed = []

    for index, annotation in enumerate(raw_annotations[:limit]):
        if not isinstance(annotation, dict):
            raise RuntimeError(
                f"Invalid annotation at index {index} for {video_id}"
            )

        label = annotation.get("label")
        segment = annotation.get("segment")

        if (
            not isinstance(label, str)
            or not isinstance(segment, list)
            or len(segment) != 2
        ):
            raise RuntimeError(
                f"Invalid annotation structure: {annotation!r}"
            )

        start = float(segment[0])
        end = float(segment[1])

        if not (start >= 0.0 and end > start):
            raise RuntimeError(
                f"Invalid annotation interval: {segment!r}"
            )

        parsed.append(
            {
                "index": index,
                "label": label,
                "start": start,
                "end": end,
                "midpoint": (start + end) / 2.0,
            }
        )

    return parsed


def seconds_from_av_time(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value) / float(av.time_base)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream: {path}")

        stream = container.streams.video[0]

        container_start = seconds_from_av_time(container.start_time)
        container_duration = seconds_from_av_time(container.duration)

        if stream.start_time is not None and stream.time_base is not None:
            stream_start = float(stream.start_time * stream.time_base)
        else:
            stream_start = None

        if stream.duration is not None and stream.time_base is not None:
            stream_duration = float(stream.duration * stream.time_base)
        else:
            stream_duration = None

        fps = None
        if stream.average_rate is not None:
            try:
                fps = float(stream.average_rate)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = None

        return {
            "container_start_time": container_start,
            "container_duration": container_duration,
            "stream_start_time": stream_start,
            "stream_duration": stream_duration,
            "average_fps": fps,
            "time_base": (
                str(stream.time_base)
                if stream.time_base is not None
                else None
            ),
        }


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def save_frame(
    frame: torch.Tensor,
    output_path: Path,
) -> None:
    if frame.dtype != torch.uint8:
        raise RuntimeError(
            f"Expected uint8 frame, got {frame.dtype}"
        )

    image = (
        frame.detach()
        .cpu()
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    Image.fromarray(image).save(output_path)


def inspect_video(
    *,
    source: sl.SequenceSource,
    annotations: list[dict[str, Any]],
    output_dir: Path,
    origin_tolerance: float,
    frame_tolerance: float,
) -> dict[str, Any]:

    dataset = sl.SequentialDataset(
        sources=(source,),
        reader=sl.SequentialVideoReader(),
        chunk_config=sl.FixedChunkConfig(
            frames_per_chunk=FRAMES_PER_CHUNK
        ),
    )
    loader = sl.build_sequential_dataloader(dataset=dataset)

    targets: dict[str, NearestFrame] = {}

    for annotation in annotations:
        index = annotation["index"]

        targets[f"{index}_start"] = NearestFrame(
            target_time=annotation["start"]
        )
        targets[f"{index}_mid"] = NearestFrame(
            target_time=annotation["midpoint"]
        )
        targets[f"{index}_end"] = NearestFrame(
            target_time=annotation["end"]
        )

    fully_contained_chunk_counts = {
        annotation["index"]: 0
        for annotation in annotations
    }

    overlapping_chunk_counts = {
        annotation["index"]: 0
        for annotation in annotations
    }

    first_timestamp: float | None = None
    last_timestamp: float | None = None

    first_frame_index: int | None = None
    last_frame_index: int | None = None

    previous_timestamp: float | None = None

    timestamp_regressions = 0
    largest_regression = 0.0
    valid_frame_count = 0
    chunk_count = 0

    with sl.sequential_sample_stream(loader) as samples:
        for sample in samples:
            chunk_count += 1

            valid_mask = sample.valid_mask
            valid_timestamps = sample.timestamps[valid_mask]
            valid_indices = sample.frame_indices[valid_mask]
            valid_frames = sample.frames[valid_mask]

            if valid_timestamps.numel() == 0:
                continue

            timestamps = valid_timestamps.tolist()
            indices = valid_indices.tolist()

            if first_timestamp is None:
                first_timestamp = float(timestamps[0])
                first_frame_index = int(indices[0])

            last_timestamp = float(timestamps[-1])
            last_frame_index = int(indices[-1])

            valid_frame_count += len(timestamps)

            # -----------------------------------------
            # Timestamp regression audit
            # -----------------------------------------
            for timestamp in timestamps:
                timestamp = float(timestamp)

                if (
                    previous_timestamp is not None
                    and timestamp < previous_timestamp
                ):
                    timestamp_regressions += 1
                    regression = previous_timestamp - timestamp
                    largest_regression = max(
                        largest_regression,
                        regression,
                    )

                previous_timestamp = timestamp

            # -----------------------------------------
            # Find nearest real frame to annotation
            # start / midpoint / end.
            # -----------------------------------------
            for frame_position, timestamp in enumerate(timestamps):
                timestamp = float(timestamp)

                for nearest in targets.values():
                    delta = abs(timestamp - nearest.target_time)

                    if delta < nearest.delta:
                        nearest.delta = delta
                        nearest.timestamp = timestamp
                        nearest.frame_index = int(
                            indices[frame_position]
                        )
                        nearest.frame = (
                            valid_frames[frame_position]
                            .detach()
                            .cpu()
                            .clone()
                        )

            # -----------------------------------------
            # Chunk <-> annotation relationship
            # -----------------------------------------
            chunk_start = float(min(timestamps))
            chunk_end = float(max(timestamps))

            for annotation in annotations:
                segment_start = annotation["start"]
                segment_end = annotation["end"]
                annotation_index = annotation["index"]

                # Any temporal overlap
                if (
                    chunk_end >= segment_start
                    and chunk_start <= segment_end
                ):
                    overlapping_chunk_counts[
                        annotation_index
                    ] += 1

                # Every valid frame must be inside the segment.
                if all(
                    segment_start <= float(timestamp) <= segment_end
                    for timestamp in timestamps
                ):
                    fully_contained_chunk_counts[
                        annotation_index
                    ] += 1

    if first_timestamp is None or last_timestamp is None:
        raise RuntimeError(
            f"No valid frame decoded for {source.sequence_id}"
        )

    # -----------------------------------------
    # Save visual audit frames
    # -----------------------------------------
    video_output_dir = output_dir / source.sequence_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    target_results = {}

    for name, nearest in targets.items():
        if (
            nearest.frame is None
            or nearest.timestamp is None
            or nearest.frame_index is None
        ):
            target_results[name] = None
            continue

        safe_name = sanitize_filename(name)

        image_path = (
            video_output_dir
            / (
                f"{safe_name}"
                f"_target={nearest.target_time:.3f}"
                f"_actual={nearest.timestamp:.3f}"
                f"_frame={nearest.frame_index}.png"
            )
        )

        save_frame(nearest.frame, image_path)

        target_results[name] = {
            "target_time": nearest.target_time,
            "actual_timestamp": nearest.timestamp,
            "delta_seconds": nearest.delta,
            "frame_index": nearest.frame_index,
            "image": str(image_path),
            "within_tolerance": nearest.delta <= frame_tolerance,
        }

    # -----------------------------------------
    # Numeric alignment judgement
    # -----------------------------------------
    first_timestamp_near_zero = (
        abs(first_timestamp) <= origin_tolerance
    )

    nearest_frames_ok = all(
        result is not None
        and result["within_tolerance"]
        for result in target_results.values()
    )

    annotation_results = []

    for annotation in annotations:
        annotation_results.append(
            {
                **annotation,
                "fully_contained_chunk_count":
                    fully_contained_chunk_counts[
                        annotation["index"]
                    ],
                "overlapping_chunk_count":
                    overlapping_chunk_counts[
                        annotation["index"]
                    ],
            }
        )

    return {
        "video_id": source.sequence_id,
        "video_path": str(source.source),
        "first_frame_index": first_frame_index,
        "last_frame_index": last_frame_index,
        "first_loader_timestamp": first_timestamp,
        "last_loader_timestamp": last_timestamp,
        "valid_frame_count": valid_frame_count,
        "chunk_count": chunk_count,
        "timestamp_regression_count": timestamp_regressions,
        "largest_timestamp_regression_seconds":
            largest_regression,
        "first_timestamp_near_zero":
            first_timestamp_near_zero,
        "nearest_annotation_frames_ok":
            nearest_frames_ok,
        "annotations": annotation_results,
        "nearest_frames": target_results,
    }


def main() -> None:
    args = parse_args()

    if args.num_videos <= 0:
        raise ValueError("--num-videos must be positive")

    if args.annotations_per_video <= 0:
        raise ValueError(
            "--annotations-per-video must be positive"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = sl.ActivityNetAdapter(
        dataset_root=args.dataset_root
    )

    database = load_activitynet_database(
        adapter.annotation_path
    )

    all_sources = adapter.sequence_sources(args.split)

    if args.video_id:
        source_by_id = {
            source.sequence_id: source
            for source in all_sources
        }

        missing = [
            video_id
            for video_id in args.video_id
            if video_id not in source_by_id
        ]

        if missing:
            raise ValueError(
                f"Unknown video IDs in {args.split}: {missing}"
            )

        sources = [
            source_by_id[video_id]
            for video_id in args.video_id
        ]

    else:
        sources = list(
            all_sources[: args.num_videos]
        )

    report = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "videos": [],
    }

    for source in sources:
        print("=" * 80)
        print(f"video_id: {source.sequence_id}")
        print(f"path:     {source.source}")

        annotations = parse_annotations(
            database=database,
            video_id=source.sequence_id,
            split=args.split,
            limit=args.annotations_per_video,
        )

        print("annotations:")

        for annotation in annotations:
            print(
                f"  [{annotation['index']}] "
                f"{annotation['label']!r}: "
                f"{annotation['start']:.3f} "
                f"-> {annotation['end']:.3f} sec"
            )

        media_info = probe_video(
            Path(source.source)
        )

        print("media:")
        for key, value in media_info.items():
            print(f"  {key}: {value}")

        loader_info = inspect_video(
            source=source,
            annotations=annotations,
            output_dir=args.output_dir,
            origin_tolerance=args.origin_tolerance,
            frame_tolerance=args.frame_tolerance,
        )

        print("loader:")
        print(
            "  first timestamp:",
            loader_info["first_loader_timestamp"],
        )
        print(
            "  last timestamp:",
            loader_info["last_loader_timestamp"],
        )
        print(
            "  timestamp regressions:",
            loader_info["timestamp_regression_count"],
        )
        print(
            "  first timestamp near zero:",
            loader_info["first_timestamp_near_zero"],
        )

        for annotation in loader_info["annotations"]:
            print(
                f"  annotation[{annotation['index']}] "
                f"{annotation['label']!r}: "
                f"fully-contained chunks="
                f"{annotation['fully_contained_chunk_count']}, "
                f"overlapping chunks="
                f"{annotation['overlapping_chunk_count']}"
            )

        print("saved visual check frames:")

        for name, result in loader_info[
            "nearest_frames"
        ].items():
            if result is None:
                print(f"  {name}: NOT FOUND")
                continue

            print(
                f"  {name}: "
                f"target={result['target_time']:.3f}, "
                f"actual={result['actual_timestamp']:.3f}, "
                f"delta={result['delta_seconds']:.4f}, "
                f"frame={result['frame_index']}"
            )
            print(f"    {result['image']}")

        report["videos"].append(
            {
                "video_id": source.sequence_id,
                "media": media_info,
                "loader": loader_info,
            }
        )

    report_path = (
        args.output_dir
        / f"alignment_report_{args.split}.json"
    )

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print(f"report: {report_path}")
    print()
    print("Manual verification:")
    print(
        "  1. Open the saved *_mid_*.png images."
    )
    print(
        "  2. Confirm that each midpoint image actually shows "
        "the annotated activity."
    )
    print(
        "  3. Check start/end images for obvious temporal offset."
    )


if __name__ == "__main__":
    main()
