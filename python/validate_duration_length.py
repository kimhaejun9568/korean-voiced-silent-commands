"""Relate the motion span of each recording to the length of its command, separately by mode.

PER_RECORDING_CSV is the output of check_files.py. Writes the correlation of each participant and a
summary to OUTPUT_DIR.

    python validate_duration_length.py DATASET PER_RECORDING_CSV OUTPUT_DIR
"""
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def hangul_blocks(text):
    return sum("가" <= ch <= "힣" for ch in unicodedata.normalize("NFC", text))


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def main():
    dataset, per_recording, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    commands = pd.read_csv(dataset / "commands.csv", encoding="utf-8-sig", keep_default_na=False)
    commands["length"] = commands["sentence_text"].map(hangul_blocks)
    rec = pd.read_csv(per_recording)
    assert len(rec) == 39999
    rec["span_s"] = (rec["last_ns"] - rec["first_ns"]) / 1e9
    rec = rec.merge(commands[["id", "length"]], left_on="sentence_num", right_on="id", how="left")
    assert rec["length"].notna().all()

    summary, per_participant = {}, []
    for mode in ("voice", "silent"):
        group = rec[rec["mode"] == mode]
        rhos = {pid: spearman(g["length"], g["span_s"]) for pid, g in group.groupby("participant_id")}
        per_participant.extend({"participant_id": pid, "mode": mode, "spearman_rho": rho}
                               for pid, rho in sorted(rhos.items()))
        values = np.array(list(rhos.values()))
        summary[mode] = {
            "recordings": int(len(group)),
            "participant_rho_median": float(np.median(values)),
            "participant_rho_min": float(values.min()),
            "participant_rho_max": float(values.max()),
        }

    pd.DataFrame(per_participant).to_csv(output / "duration_length_participants.csv", index=False)
    (output / "duration_length_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
