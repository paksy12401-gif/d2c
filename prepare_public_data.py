"""Download the public AITEE circuit dataset only; never uploads local data.

Run once with internet access before offline training. The archive is treated
as data: only PNG/JPEG/label text is read, and no downloaded code is executed.
"""
import hashlib
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen, Request
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "public" / "aitee"
CLASSES = ["E", "GND", "I", "P", "R", "V", "ident-i", "ident-i-dir", "ident-res", "ident-v", "ident-v-dir"]


def download(url, path, max_bytes=300_000_000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    temporary = path.with_suffix(path.suffix + ".partial")
    total = 0
    with urlopen(Request(url, headers={"User-Agent": "local-harness-research"}), timeout=60) as response:
        with temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Public download exceeded the size limit")
                output.write(chunk)
    temporary.replace(path)
    return path


def prepare(offline=False, rebuild=False):
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "provenance.json"
    if manifest_path.exists() and not rebuild:
        print(manifest_path.read_text(encoding="utf-8"))
        return DEST / "dataset.yaml"
    # Pin all downloaded dataset files to one observed source commit.
    commit = None
    archive = DEST / "aitee_dataset.zip"
    if offline:
        if not archive.is_file() or not (DEST / "LICENSE").is_file():
            raise ValueError("Offline preparation requires the downloaded ZIP and LICENSE")
    else:
        with urlopen(Request("https://api.github.com/repos/CKnievel/aitee-dataset/commits/main",
                             headers={"User-Agent": "local-harness-research"}), timeout=60) as response:
            commit = json.load(response)["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Invalid repository revision")
        raw = f"https://raw.githubusercontent.com/CKnievel/aitee-dataset/{commit}/"
        for name in ("LICENSE", "README.md", "CITATION.cff"):
            download(raw + name, DEST / name)
        archive = download(raw + "aitee_dataset.zip", archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError("Dataset archive is not a ZIP (possibly an LFS pointer)")
    with zipfile.ZipFile(archive) as z:
        image_entries, label_entries = {}, {}
        for info in z.infolist():
            parts = Path(info.filename.replace("\\", "/")).parts
            if "split" in parts or info.is_dir():
                continue
            name = Path(info.filename)
            if "images" in parts and name.suffix.lower() in (".png", ".jpg", ".jpeg"):
                image_entries[name.stem] = info
            elif "labels" in parts and name.suffix.lower() == ".txt":
                label_entries[name.stem] = info
        if not image_entries or set(image_entries) - set(label_entries):
            raise ValueError("Public image/label pairing does not match the expected archive")
        orphan_labels = sorted(set(label_entries) - set(image_entries))
        # Keep filename-encoded circuit families together. This is stronger than
        # an image-random split, but unlabelled visual duplicates remain possible.
        groups = {}
        for stem in sorted(image_entries):
            match = re.search(r"[-_]s[-_](\d+)", stem)
            group = match[1] if match else stem
            groups.setdefault(group, []).append(stem)
        group_names = sorted(groups)
        if len(group_names) < 5:
            raise ValueError("Insufficient independent groups for a useful trial split")
        random.Random(42).shuffle(group_names)
        group_names.sort(key=lambda g: len(groups[g]), reverse=True)
        targets = {"train": len(image_entries)*0.70, "val": len(image_entries)*0.15,
                   "test": len(image_entries)*0.15}
        counts = {"train": 0, "val": 0, "test": 0}
        assignments, hashes = {}, {}
        for index, group in enumerate(group_names):
            split = max(counts, key=lambda s: targets[s] - counts[s])
            for stem in groups[group]:
                image_info, label_info = image_entries[stem], label_entries[stem]
                if image_info.file_size > 20_000_000 or label_info.file_size > 1_000_000:
                    raise ValueError("Unexpected oversized dataset member")
                image_bytes = z.read(image_info)
                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest in hashes:
                    # Identical copies are removed rather than leaking across splits.
                    continue
                hashes[digest] = split
                for folder in ("images", "labels"):
                    (DEST / folder / split).mkdir(parents=True, exist_ok=True)
                (DEST / "images" / split / (stem + Path(image_info.filename).suffix.lower())).write_bytes(image_bytes)
                (DEST / "labels" / split / f"{stem}.txt").write_bytes(z.read(label_info))
                for other in counts:
                    if other == split:
                        continue
                    for folder, name in (("images", stem + Path(image_info.filename).suffix.lower()),
                                         ("labels", f"{stem}.txt")):
                        old_file = (DEST / folder / other / name).resolve()
                        if not old_file.is_relative_to(DEST.resolve()):
                            raise ValueError("Generated path escaped the dataset root")
                        old_file.unlink(missing_ok=True)  # Only our previously generated split copy.
                counts[split] += 1
                assignments[stem] = {"split": split, "group": group, "sha256": digest}
    config = {"path": DEST.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASSES}
    (DEST / "dataset.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {"dataset": "AITEE Electrical Circuit Dataset", "source": "https://github.com/CKnievel/aitee-dataset",
                "commit": commit, "license": "Apache-2.0", "task": "detect",
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "split_rule": "seed42; filename s-family grouped; largest groups assigned by 70/15/15 image-count deficit; exact duplicates removed",
                "counts": counts, "groups": len(group_names), "classes": CLASSES,
                "excluded_labels_without_image": orphan_labels,
                "limitations": "Educational circuits, not harness diagrams. Filename families do not guarantee all visual duplicates are separated."}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEST / "split_manifest.json").write_text(json.dumps(assignments, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return DEST / "dataset.yaml"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Only process the existing local archive; no network")
    parser.add_argument("--rebuild", action="store_true", help="Recreate only the generated split copies")
    args = parser.parse_args()
    prepare(offline=args.offline, rebuild=args.rebuild)
    if not args.offline:
        download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                 ROOT / "models" / "yolo11n.pt", max_bytes=30_000_000)
