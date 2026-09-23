from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".m4v",
}


def normalize_video_id(path: Path) -> str:
    """
    ActivityNetのlocal video filenameをannotation JSONのIDへ変換する。

    例:
        v_-1EC1ZP6aC4.mp4
        -> -1EC1ZP6aC4

        v__-4ngMPCA9A.mp4
        -> _-4ngMPCA9A

    先頭の 'v_' だけを取り除く。
    """
    stem = path.stem

    if stem.startswith("v_"):
        return stem[2:]

    return stem


def scan_root(root: Path):
    """
    root以下を再帰的に走査する。

    Returns:
        video_paths:
            対象extensionのvideo file一覧。

        id_to_paths:
            normalized video ID -> local path一覧。

        extension_counts:
            root内に存在する全fileのextension集計。
    """
    if not root.is_dir():
        return [], defaultdict(list), Counter()

    all_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
    ]

    extension_counts = Counter(
        path.suffix.lower() if path.suffix else "<no extension>"
        for path in all_files
    )

    video_paths = [
        path
        for path in all_files
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    id_to_paths = defaultdict(list)

    for path in video_paths:
        video_id = normalize_video_id(path)
        id_to_paths[video_id].append(path)

    return video_paths, id_to_paths, extension_counts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "activitynet_root",
        type=Path,
        help="ActivityNet root directory",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "ActivityNet v1.3 annotation JSON. "
            "省略時は <root>/json/activity_net.v1-3.min.json"
        ),
    )

    parser.add_argument(
        "--show-examples",
        type=int,
        default=10,
        help="missing / duplicate等で表示するexample数",
    )

    args = parser.parse_args()

    activitynet_root = args.activitynet_root.resolve()

    annotation_path = (
        args.json.resolve()
        if args.json is not None
        else activitynet_root
        / "json"
        / "activity_net.v1-3.min.json"
    )

    roots = {
        "manual_crawling": (
            activitynet_root
            / "manual_crawling_from_youtube"
            / "video"
        ),
        "v1-3/train_val": (
            activitynet_root
            / "v1-3"
            / "train_val"
        ),
        "v1-3/test": (
            activitynet_root
            / "v1-3"
            / "test"
        ),
    }

    # =========================================================
    # Annotation JSON
    # =========================================================

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"annotation JSON does not exist: {annotation_path}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        annotation = json.load(file)

    if "database" not in annotation:
        raise ValueError(
            "annotation JSON does not contain 'database'"
        )

    database = annotation["database"]

    json_ids = set(database.keys())

    subset_to_ids = defaultdict(set)

    for video_id, metadata in database.items():
        subset = metadata.get("subset")

        if subset is None:
            subset = "<missing subset>"

        subset_to_ids[subset].add(video_id)

    # =========================================================
    # Local roots
    # =========================================================

    all_local_id_to_paths = defaultdict(list)

    print("=" * 80)
    print("ActivityNet inventory")
    print("=" * 80)
    print()

    print(f"ActivityNet root : {activitynet_root}")
    print(f"Annotation JSON  : {annotation_path}")
    print(f"JSON video IDs   : {len(json_ids)}")
    print()

    print("=" * 80)
    print("1. Rootごとの情報")
    print("=" * 80)

    root_results = {}

    for root_name, root_path in roots.items():
        (
            video_paths,
            id_to_paths,
            extension_counts,
        ) = scan_root(root_path)

        local_ids = set(id_to_paths.keys())
        matched_ids = local_ids & json_ids
        local_only_ids = local_ids - json_ids

        root_results[root_name] = {
            "video_paths": video_paths,
            "id_to_paths": id_to_paths,
            "local_ids": local_ids,
            "matched_ids": matched_ids,
            "local_only_ids": local_only_ids,
            "extension_counts": extension_counts,
        }

        for video_id, paths in id_to_paths.items():
            all_local_id_to_paths[video_id].extend(paths)

        print()
        print(f"[{root_name}]")
        print(f"path                 : {root_path}")
        print(f"exists               : {root_path.is_dir()}")
        print(f"video files          : {len(video_paths)}")
        print(f"normalized unique IDs: {len(local_ids)}")
        print(f"JSON matched IDs     : {len(matched_ids)}")
        print(f"local-only IDs       : {len(local_only_ids)}")

        print("extensions:")
        for extension, count in sorted(
            extension_counts.items()
        ):
            print(
                f"  {extension:15s} {count}"
            )

    # =========================================================
    # Global local union
    # =========================================================

    all_local_ids = set(all_local_id_to_paths.keys())

    matched_ids = json_ids & all_local_ids
    missing_ids = json_ids - all_local_ids
    local_only_ids = all_local_ids - json_ids

    duplicate_ids = {
        video_id: paths
        for video_id, paths
        in all_local_id_to_paths.items()
        if len(paths) > 1
    }

    print()
    print("=" * 80)
    print("2. 全rootを統合した結果")
    print("=" * 80)
    print()

    print(
        f"local unique IDs : {len(all_local_ids)}"
    )
    print(
        f"JSON matched IDs : {len(matched_ids)}"
    )
    print(
        f"missing IDs      : {len(missing_ids)}"
    )
    print(
        f"local-only IDs   : {len(local_only_ids)}"
    )
    print(
        f"duplicate IDs    : {len(duplicate_ids)}"
    )

    # =========================================================
    # Subset
    # =========================================================

    print()
    print("=" * 80)
    print("3. subsetごとの利用可能本数")
    print("=" * 80)
    print()

    for subset in sorted(subset_to_ids):
        subset_ids = subset_to_ids[subset]

        available = subset_ids & all_local_ids
        missing = subset_ids - all_local_ids

        print(f"[{subset}]")
        print(
            f"metadata total : {len(subset_ids)}"
        )
        print(
            f"local available: {len(available)}"
        )
        print(
            f"missing        : {len(missing)}"
        )
        print()

    # =========================================================
    # Examples
    # =========================================================

    n = args.show_examples

    print("=" * 80)
    print("4. missing ID examples")
    print("=" * 80)

    for video_id in sorted(missing_ids)[:n]:
        subset = database[video_id].get(
            "subset",
            "<missing subset>",
        )
        print(
            f"{video_id}  subset={subset}"
        )

    print()
    print("=" * 80)
    print("5. local-only ID examples")
    print("=" * 80)

    for video_id in sorted(local_only_ids)[:n]:
        print(video_id)

        for path in all_local_id_to_paths[
            video_id
        ]:
            print(f"  {path}")

    print()
    print("=" * 80)
    print("6. duplicate ID examples")
    print("=" * 80)

    for video_id in sorted(
        duplicate_ids
    )[:n]:
        print(video_id)

        for path in duplicate_ids[video_id]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
