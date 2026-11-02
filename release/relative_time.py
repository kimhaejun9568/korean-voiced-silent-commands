"""Replace absolute motion timestamps with times from each participant's first recording.

In every facial and head-motion file the UnixTimeNs column becomes TimeNs, the number of
nanoseconds since the first motion row of the participant's first recording, and in
recording_index.csv the column unix_time_ns becomes start_ns. Intervals and row order are
unchanged. The origin of each participant is written to ORIGINS_JSON, which is kept private.

    python relative_time.py DATASET_ROOT ORIGINS_JSON [WORKERS]
"""
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BOM = b"\xef\xbb\xbf"
NL = "\r\n"


def read_lines(path):
    raw = path.read_bytes()
    bom = raw.startswith(BOM)
    text = (raw[len(BOM):] if bom else raw).decode("utf-8")
    if NL not in text:
        raise ValueError(f"{path}: expected CRLF line endings")
    lines = text.split(NL)
    trailing = lines[-1] == ""
    return bom, lines[:-1] if trailing else lines, trailing


def write_lines(path, bom, lines, trailing):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((BOM if bom else b"") + (NL.join(lines) + (NL if trailing else "")).encode("utf-8"))
    os.replace(tmp, path)


def convert_motion(path, origin):
    leftover = path.with_name(path.name + ".tmp")
    if leftover.exists():
        leftover.unlink()
    bom, lines, trailing = read_lines(path)
    header = lines[0].split(",")
    if header[1] == "TimeNs":
        return 0
    if header[:2] != ["Timestamp", "UnixTimeNs"]:
        raise ValueError(f"{path}: unexpected header")
    out = [lines[0].replace("UnixTimeNs", "TimeNs", 1)]
    for line in lines[1:]:
        first, ns, rest = line.split(",", 2)
        relative = int(ns) - origin
        if relative < 0:
            raise ValueError(f"{path}: row earlier than the participant origin")
        out.append(f"{first},{relative},{rest}")
    write_lines(path, bom, out, trailing)
    return len(out) - 1


def convert_participant(args):
    root, participant, origin, files = args
    rows = sum(convert_motion(Path(root) / name, origin) for name in files)
    return participant, len(files), rows


def first_time(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        f.readline()
        return int(f.readline().split(",")[1])


def main():
    root, origins_path = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    with open(root / "recording_index.csv", encoding="utf-8-sig", newline="") as f:
        index = list(csv.DictReader(f))
    if "start_ns" in index[0]:
        print("recording_index.csv is already converted")
        return

    origins = defaultdict(lambda: None)
    for row in index:
        ns = int(row["unix_time_ns"])
        p = row["participant_id"]
        origins[p] = ns if origins[p] is None else min(origins[p], ns)
    origins = dict(sorted(origins.items()))
    if origins_path.exists():
        saved = json.loads(origins_path.read_text(encoding="utf-8"))
        if saved != origins:
            raise ValueError("saved origins differ from the index")
    else:
        origins_path.parent.mkdir(parents=True, exist_ok=True)
        origins_path.write_text(json.dumps(origins, indent=2) + "\n", encoding="utf-8")

    files = defaultdict(list)
    for row in index:
        files[row["participant_id"]] += [row["face_csv"], row["imu_csv"]]
    jobs = [(str(root), p, origins[p], files[p]) for p in origins]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for participant, n, rows in pool.map(convert_participant, jobs):
            print(f"{participant}: {n} files, {rows} rows converted", flush=True)

    for row in index:
        expected = int(row["unix_time_ns"]) - origins[row["participant_id"]]
        if first_time(root / row["face_csv"]) != expected or first_time(root / row["imu_csv"]) != expected:
            raise ValueError(f"{row['face_csv']}: start time does not match")

    path = root / "recording_index.csv"
    bom, lines, trailing = read_lines(path)
    header = lines[0].split(",")
    k = header.index("unix_time_ns")
    header[k] = "start_ns"
    out = [",".join(header)]
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) != len(header):
            raise ValueError("unexpected comma in recording_index.csv")
        fields[k] = str(int(fields[k]) - origins[fields[0]])
        out.append(",".join(fields))
    write_lines(path, bom, out, trailing)
    print(f"recording_index.csv: {len(out) - 1} rows converted")


if __name__ == "__main__":
    main()
