"""Write commands.csv, recording_index.csv and participant_sessions.csv from the released files.

Collection days are numbered from the recording dates in Korea Standard Time; the dates are not
written to the tables. Tables are UTF-8 with a byte-order mark and CRLF line endings.

    python build_metadata_tables.py DATASET_ROOT COMMANDS_CSV
"""
import csv
import datetime as dt
import sys
import wave
from collections import defaultdict
from pathlib import Path

root, source_commands = Path(sys.argv[1]), Path(sys.argv[2])
KST = dt.timezone(dt.timedelta(hours=9))
MODES = (("Voice", "voice"), ("Silent", "silent"))


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(rows)


with open(source_commands, encoding="utf-8-sig", newline="") as handle:
    commands = {int(row["id"]): row for row in csv.DictReader(handle)}
assert sorted(commands) == list(range(1, 1001))
write_csv(root / "commands.csv", ["id", "sentence_text", "topic"],
          [[cid, commands[cid]["sentence_text"], commands[cid]["topic"]] for cid in sorted(commands)])

records = []
for participant in sorted(p for p in root.iterdir() if p.is_dir()):
    for folder, mode in MODES:
        for face in sorted((participant / folder / "Face").glob(f"s*_{mode}_face.csv")):
            cid = int(face.name[1:5])
            imu = participant / folder / "IMU" / f"s{cid:04d}_{mode}_imu.csv"
            assert imu.exists(), f"missing head-motion file for {participant.name} {mode} {cid}"
            with open(face, encoding="utf-8-sig") as handle:
                handle.readline()
                unix_ns = int(handle.readline().split(",")[1])
            when = dt.datetime.fromtimestamp(unix_ns / 1e9, KST)
            audio_rel, audio_seconds = "", ""
            if mode == "voice":
                wav = participant / folder / f"s{cid:04d}_voice_audio.wav"
                with wave.open(str(wav)) as w:
                    audio_seconds = f"{w.getnframes() / w.getframerate():.6f}"
                audio_rel = wav.relative_to(root).as_posix()
            records.append({
                "participant_id": participant.name, "mode": mode, "sentence_num": cid,
                "sentence_id": f"s{cid:04d}", "sentence_text": commands[cid]["sentence_text"],
                "topic": commands[cid]["topic"], "date": when.date().isoformat(),
                "unix_time_ns": unix_ns,
                "audio_seconds": audio_seconds, "face_csv": face.relative_to(root).as_posix(),
                "imu_csv": imu.relative_to(root).as_posix(), "audio_wav": audio_rel,
            })

dates = defaultdict(set)
for r in records:
    dates[r["participant_id"]].add(r["date"])
assert all(len(d) == 2 for d in dates.values()), "each participant should have two collection days"
day_of = {(p, d): i + 1 for p, ds in dates.items() for i, d in enumerate(sorted(ds))}
records.sort(key=lambda r: (r["participant_id"], r["sentence_num"], r["mode"] != "voice"))
index_header = ["participant_id", "mode", "sentence_num", "sentence_id", "sentence_text", "topic",
                "session_day", "unix_time_ns", "audio_seconds", "face_csv", "imu_csv", "audio_wav"]
write_csv(root / "recording_index.csv", index_header, [
    [r["participant_id"], r["mode"], r["sentence_num"], r["sentence_id"], r["sentence_text"], r["topic"],
     day_of[(r["participant_id"], r["date"])], r["unix_time_ns"],
     r["audio_seconds"], r["face_csv"], r["imu_csv"], r["audio_wav"]] for r in records])

session_rows = []
for p in sorted(dates):
    mine = [r for r in records if r["participant_id"] == p]
    d1, d2 = sorted(dates[p])
    voiced = [r for r in mine if r["mode"] == "voice"]
    session_rows.append([
        p, len(voiced), len(mine) - len(voiced),
        sum(r["date"] == d1 for r in voiced), sum(r["date"] == d2 for r in voiced),
        f"{sum(float(r['audio_seconds']) for r in voiced):.2f}",
    ])
write_csv(root / "participant_sessions.csv",
          ["participant_id", "voice_recordings", "silent_recordings",
           "day1_commands", "day2_commands", "voiced_audio_seconds"], session_rows)

total_audio = sum(float(r["audio_seconds"]) for r in records if r["audio_seconds"])
print(f"commands: {len(commands)} | index rows: {len(records)} "
      f"(voice {sum(r['mode'] == 'voice' for r in records)}, silent {sum(r['mode'] == 'silent' for r in records)})")
print(f"voiced audio: {total_audio:.2f} s ({total_audio / 3600:.2f} h)")
print("day allocations other than 500/500:",
      [(row[0], row[3], row[4]) for row in session_rows if (row[3], row[4]) != (500, 500)])
