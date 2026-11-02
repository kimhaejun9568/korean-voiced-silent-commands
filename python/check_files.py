"""Check every file listed in recording_index.csv.

Writes per_recording.csv and summary.json to OUTPUT_DIR. The summary holds the numbers reported in
the Technical Validation section of the Data Descriptor.

    python check_files.py DATASET OUTPUT_DIR [WORKERS]
"""
import csv
import json
import sys
import wave
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CLOCKS = ["Timestamp", "TimeNs"]
STATUS = ["ConfidenceLower", "ConfidenceUpper", "EyeFollowingValid"]


def read_table(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    numeric = all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes)
    return frame, numeric


def repeated_rows(values):
    """Rows whose values all equal those of the previous row."""
    return (values[1:] == values[:-1]).all(axis=1)


def check_participant(args):
    root, participant, rows = args
    root = Path(root)
    out, shared, face_headers, head_headers = [], [], Counter(), Counter()
    previous = None
    for row in sorted(rows, key=lambda r: int(r["start_ns"])):
        face, face_ok = read_table(root / row["face_csv"])
        head, head_ok = read_table(root / row["imu_csv"])
        face_headers[tuple(face.columns)] += 1
        head_headers[tuple(head.columns)] += 1
        weights = face.drop(columns=CLOCKS + STATUS)
        t = face["TimeNs"].to_numpy(dtype=np.int64)
        gaps = np.diff(t)
        span_s = (t[-1] - t[0]) / 1e9
        values = np.concatenate([face.to_numpy(dtype=float).ravel(), head.to_numpy(dtype=float).ravel()])
        face_same = repeated_rows(weights.to_numpy(dtype=float))
        head_values = head.drop(columns=CLOCKS).to_numpy(dtype=float)
        stats = {
            "participant_id": participant, "mode": row["mode"], "sentence_num": int(row["sentence_num"]),
            "rows_face": len(face),
            "numeric": face_ok and head_ok,
            "nonfinite": int((~np.isfinite(values)).sum()),
            "weights_outside_0_1": int(((weights < 0) | (weights > 1)).sum().sum()),
            "clock_mismatch": int(len(face) != len(head) or not (
                np.array_equal(face["TimeNs"].to_numpy(), head["TimeNs"].to_numpy())
                and np.array_equal(face["Timestamp"].to_numpy(), head["Timestamp"].to_numpy()))),
            "clock_not_increasing": int((gaps <= 0).sum() + (np.diff(face["Timestamp"].to_numpy()) <= 0).sum()),
            "first_ns": int(t[0]), "last_ns": int(t[-1]), "span_s": span_s,
            "rate_hz": (len(t) - 1) / span_s if len(t) > 1 else float("nan"),
            "intervals": int(len(gaps)),
            "max_gap_ns": int(gaps.max()) if len(gaps) else 0,
            "face_repeat_rows": int(face_same.sum()),
            "face_update_hz": float((~face_same).sum() / span_s),
            "head_repeat_rows": int(repeated_rows(head_values).sum()),
            "conf_lower_zero": int((face["ConfidenceLower"] == 0).sum()),
            "conf_lower_not_0_or_1": int((~face["ConfidenceLower"].isin([0, 1])).sum()),
            "conf_upper_not_one": int((face["ConfidenceUpper"] != 1).sum()),
            "eye_not_one": int((face["EyeFollowingValid"] != 1).sum()),
        }
        if row["mode"] == "voice":
            with wave.open(str(root / row["audio_wav"])) as w:
                params = (w.getnchannels(), w.getsampwidth() * 8, w.getframerate())
                declared = w.getnframes()
                pcm = np.frombuffer(w.readframes(declared), dtype=np.int16)
            stats.update({"wav_format": "%d/%d/%d" % params, "wav_frames": declared,
                          "wav_decoded": int(len(pcm)), "wav_all_zero": int(len(pcm) == 0 or not pcm.any()),
                          "audio_seconds": declared / params[2]})
        combined = np.concatenate([face.to_numpy(dtype=float), head_values], axis=1)
        if previous is not None:
            prev_times, prev_rows = previous
            common = np.intersect1d(prev_times, t)
            matches = 0
            if len(common):
                a = prev_rows[np.searchsorted(prev_times, common)]
                b = combined[np.searchsorted(t, common)]
                matches = int((a == b).all(axis=1).sum())
            shared.append(matches)
        previous = (t, combined)
        out.append(stats)
    return out, shared, face_headers, head_headers


def main():
    root, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    outdir.mkdir(parents=True, exist_ok=True)
    with open(root / "recording_index.csv", encoding="utf-8-sig", newline="") as handle:
        index = list(csv.DictReader(handle))
    with open(root / "commands.csv", encoding="utf-8-sig", newline="") as handle:
        commands = {int(r["id"]): r for r in csv.DictReader(handle)}
    with open(root / "participant_sessions.csv", encoding="utf-8-sig", newline="") as handle:
        sessions = {r["participant_id"]: r for r in csv.DictReader(handle)}

    keys = Counter((r["participant_id"], r["mode"], int(r["sentence_num"])) for r in index)
    expected = {(f"P{p:02d}", m, c) for p in range(1, 21) for m in ("voice", "silent") for c in range(1, 1001)}
    labels = {
        "index_rows": len(index), "duplicate_keys": sum(n > 1 for n in keys.values()),
        "missing_keys": sorted("-".join(map(str, k)) for k in expected - set(keys)),
        "text_mismatches": sum(r["sentence_text"] != commands[int(r["sentence_num"])]["sentence_text"]
                               or r["topic"] != commands[int(r["sentence_num"])]["topic"] for r in index),
        "id_mismatches": sum(r["sentence_id"] != f"s{int(r['sentence_num']):04d}" for r in index),
        "distinct_texts": len({c["sentence_text"] for c in commands.values()}),
        "day_count_mismatches": sum(
            sum(r["mode"] == "voice" and r["session_day"] == str(d) for r in index if r["participant_id"] == p)
            != int(sessions[p][f"day{d}_commands"]) for p in sessions for d in (1, 2)),
    }

    by_participant = defaultdict(list)
    for r in index:
        by_participant[r["participant_id"]].append(r)
    results, shared, face_headers, head_headers = [], [], Counter(), Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out, sh, fh, hh in pool.map(check_participant, [(str(root), p, rows) for p, rows in sorted(by_participant.items())]):
            results.extend(out)
            shared.extend(sh)
            face_headers.update(fh)
            head_headers.update(hh)
    per = pd.DataFrame(results)
    per.to_csv(outdir / "per_recording.csv", index=False)

    flagged_conf = per[per["conf_lower_zero"] > 0]
    conf_fraction = flagged_conf["conf_lower_zero"] / flagged_conf["rows_face"]
    voiced = per[per["mode"] == "voice"]
    audio_minus_span = voiced["audio_seconds"] - voiced["span_s"]
    starts = per.pivot_table(index=["participant_id", "sentence_num"], columns="mode", values="first_ns")
    complete = starts.dropna()

    summary = {
        "labels": labels,
        "files": {"facial": len(per), "head": len(per), "voiced_wav": int(len(voiced))},
        "facial_headers": {str(len(k)): v for k, v in face_headers.items()},
        "head_headers": {str(len(k)): v for k, v in head_headers.items()},
        "rows_facial_total": int(per["rows_face"].sum()),
        "non_numeric_files": int((~per["numeric"]).sum()), "nonfinite_values": int(per["nonfinite"].sum()),
        "weights_outside_0_1": int(per["weights_outside_0_1"].sum()),
        "clock_mismatch_recordings": int(per["clock_mismatch"].sum()),
        "clock_not_increasing": int(per["clock_not_increasing"].sum()),
        "rate_hz_median": float(per["rate_hz"].median()),
        "longest_gap_s": int(per["max_gap_ns"].max()) / 1e9,
        "facial_update": {
            "repeated_rows_percent_of_intervals": 100 * per["face_repeat_rows"].sum() / per["intervals"].sum(),
            "update_hz_median_by_mode": per.groupby("mode")["face_update_hz"].median().to_dict(),
            "head_repeated_rows": int(per["head_repeat_rows"].sum()),
        },
        "conf_lower_values_other_than_0_1": int(per["conf_lower_not_0_or_1"].sum()),
        "conf_lower_zero_rows": int(per["conf_lower_zero"].sum()),
        "conf_lower_zero_percent_of_rows": 100 * per["conf_lower_zero"].sum() / per["rows_face"].sum(),
        "conf_lower_zero_recordings": int(len(flagged_conf)),
        "conf_lower_zero_fraction_median": float(conf_fraction.median()),
        "conf_lower_zero_fraction_max": float(conf_fraction.max()),
        "conf_upper_not_one_rows": int(per["conf_upper_not_one"].sum()),
        "eye_following_not_one_rows": int(per["eye_not_one"].sum()),
        "wav_formats": voiced["wav_format"].value_counts().to_dict(),
        "wav_frame_mismatches": int((voiced["wav_frames"] != voiced["wav_decoded"]).sum()),
        "wav_empty_or_all_zero": int(voiced["wav_all_zero"].sum()),
        "voiced_audio_seconds": float(voiced["audio_seconds"].sum()),
        "audio_minus_motion_span_s": {"median": float(audio_minus_span.median()),
                                      "max_abs": float(audio_minus_span.abs().max())},
        "transitions": len(shared), "transitions_sharing_rows": int(sum(n > 0 for n in shared)),
        "complete_pairs": int(len(complete)),
        "voiced_first_in_complete_pairs": int((complete["voice"] < complete["silent"]).sum()),
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
