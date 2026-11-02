# Paired voiced and silent Korean commands: code

Code for the Data Descriptor "A paired dataset of voiced and silent Korean commands with headset facial and head-motion signals" (Scientific Data, under review).

The data are on Zenodo under controlled access: https://doi.org/10.5281/zenodo.23853036

| Folder | Contents |
|---|---|
| `unity_collection/` | C# scripts of the Unity app used for recording |
| `release/` | Scripts used to prepare the released files |
| `python/` | Scripts that reproduce Technical Validation |

## Data layout

Download the 20 participant archives (`P01.zip` to `P20.zip`) and the three tables, and extract the archives into the folder that holds the tables:

```
DATASET/
  commands.csv
  recording_index.csv
  participant_sessions.csv
  P01/Voice/Face/s0001_voice_face.csv
  P01/Voice/IMU/s0001_voice_imu.csv
  P01/Voice/s0001_voice_audio.wav
  P01/Silent/Face/s0001_silent_face.csv
  P01/Silent/IMU/s0001_silent_imu.csv
  ...
```

- Pair voiced and silent recordings by participant and command ID.
- Motion rows and audio samples have separate time bases.
- `TimeNs` in the motion files counts nanoseconds from the participant's first recording. Absolute dates are not included.
- Each recording runs from the button press to the release, and consecutive recordings share no rows.

## Technical Validation

Install the dependencies with `pip install -r requirements.txt`. We ran the scripts with Python 3.14.

```
cd python
python check_files.py DATASET out
python validate_duration_length.py DATASET out/per_recording.csv out
python validate_classification.py DATASET out/classification
```

| Script | Reproduces |
|---|---|
| `check_files.py` | Completeness, timing and tracking status, and `per_recording.csv` for the next script |
| `validate_duration_length.py` | Correlation between recording duration and command length |
| `validate_classification.py` | Reference classifier and Table 4; needs PyTorch and a CUDA GPU |

## Collection app

The collection app ran in Unity 6000.4.0f1 with the Meta XR Core SDK 85.0.0, on a Windows PC connected to a Meta Quest Pro by Quest Link. The scripts in `unity_collection/` are its recording scripts, cleaned up and adapted for release so that they write files in the released format.

- `SentenceManager.cs` shows the commands in random order. The participant holds the A button while speaking and releases it at the end, first for the voiced and then for the silent production. The thumbstick moves between commands. The order and progress are saved, so a session can be resumed.
- `FacialDataRecorder.cs` writes, while the button is held, a facial row and a head-motion row for every frame (90 Hz) with a valid face state, and, for voiced recordings, saves the microphone audio as a WAV file.
- `ParticipantSession.cs` sets the output folders.
- `ParticipantNameController.cs` takes the participant label from the Inspector.

The app writes absolute times in `UnixTimeNs`; in the released files they were replaced by `TimeNs`. It also writes `session_<time>.json` (random seed and command order) and `progress.json`, which are not part of the dataset.

### Running the app

The Unity project, the Meta SDK and other third-party assets are not included.

1. Set up a Unity project for Quest Link with the Meta XR Core SDK and face tracking enabled (visual data source only).
2. Add an OVRCameraRig and a canvas with four TextMeshPro texts and an Image for the recording indicator.
3. Put `SentenceManager`, `FacialDataRecorder` and `ParticipantNameController` on one GameObject and connect the fields in the Inspector.
4. Copy `commands.csv` to `Assets/StreamingAssets/corpus_1000_xr.csv`.

Recordings are written to `Application.persistentDataPath/Data/<participant label>/`.

## Release scripts

- `build_metadata_tables.py` wrote the three tables from the signal files.
- `relative_time.py` then replaced the absolute timestamps with times from each participant's first recording (`TimeNs` in the motion files, `start_ns` in the index).
- Before these steps, participant folders were renamed to P01 to P20 with a mapping table that is kept private.

## License and citation

The code is under the MIT License. The data are not covered by it; access to the data is governed by the data use agreement in [DATA_USE_AGREEMENT.md](DATA_USE_AGREEMENT.md). To request access, use the Zenodo record and email the signed agreement as described there.

If you use the code or the data, please cite the Data Descriptor (reference to be added) and the dataset.
