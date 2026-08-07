"""
Merge the raw Kaggle download (per-class Train/Val subfolders, class_id
always 0 inside each label file) into a single YOLO-format dataset with a
proper 80/10/10 stratified train/valid/test split and correct class_ids.

Usage:
    python -m src.data.merge_dataset --src data/raw --dst data/yolo
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_MAP = {
    "Glioma": 0,
    "Meningioma": 1,
    "Pituitary": 2,
    "No Tumor": 3,
}


def collect_items(src: Path):
    items = []
    for split_dir in ["Train", "Val"]:
        for class_name, class_id in CLASS_MAP.items():
            img_dir = src / split_dir / class_name / "images"
            lbl_dir = src / split_dir / class_name / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    items.append((img_path, lbl_path, class_id))
    return items


def stratified_split(items, train_frac=0.8, valid_frac=0.1, seed=42):
    random.seed(seed)
    by_class = defaultdict(list)
    for item in items:
        by_class[item[2]].append(item)

    train, valid, test = [], [], []
    for class_id, group in by_class.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_valid = int(n * valid_frac)
        train += group[:n_train]
        valid += group[n_train:n_train + n_valid]
        test += group[n_train + n_valid:]
    return train, valid, test


def write_split(items, dst: Path, split_name: str):
    img_out = dst / split_name / "images"
    lbl_out = dst / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path, class_id in items:
        shutil.copy(img_path, img_out / img_path.name)
        lines = lbl_path.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            parts[0] = str(class_id)  # rewrite with the correct class id
            new_lines.append(" ".join(parts))
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(new_lines))


def main(src_dir: str, dst_dir: str) -> None:
    src, dst = Path(src_dir), Path(dst_dir)
    items = collect_items(src)
    print(f"Total images found: {len(items)}")

    train, valid, test = stratified_split(items)
    print(f"Train: {len(train)}  Valid: {len(valid)}  Test: {len(test)}")

    write_split(train, dst, "train")
    write_split(valid, dst, "valid")
    write_split(test, dst, "test")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw")
    parser.add_argument("--dst", default="data/yolo")
    args = parser.parse_args()
    main(args.src, args.dst)
