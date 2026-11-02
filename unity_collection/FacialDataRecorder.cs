using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;

public class FacialDataRecorder : MonoBehaviour
{
    private const int AudioSampleRate = 44100;
    private const int MicrophoneBufferSeconds = 35;
    private const int MicrophoneChunkFrames = 4096;

    private bool _isRecording;
    private bool _captureAudio;
    private StringBuilder _sb = new StringBuilder(4096);
    private OVRPlugin.FaceState _faceState;
    private float _lastConfidenceLower;
    private float _lastConfidenceUpper;
    private bool _lastIsValid;

    private AudioClip _micClip;
    private string _audioPath;
    private readonly string _micDevice = null;
    private int _micReadPos;
    private int _micChannels = 1;
    private List<float[]> _audioChunks = new List<float[]>(512);

    private readonly BlockingCollection<CsvCommand> _csvQueue = new BlockingCollection<CsvCommand>();
    private readonly BlockingCollection<AudioSaveCommand> _audioQueue = new BlockingCollection<AudioSaveCommand>();
    private readonly ConcurrentQueue<string> _workerErrors = new ConcurrentQueue<string>();
    private Thread _csvThread;
    private Thread _audioThread;

    private enum CsvCommandType
    {
        StartSession,
        FaceRow,
        ImuRow,
        StopSession
    }

    private sealed class CsvCommand
    {
        public CsvCommandType Type;
        public string FacePath;
        public string ImuPath;
        public string FaceHeader;
        public string ImuHeader;
        public string Line;
    }

    private sealed class AudioSaveCommand
    {
        public string Path;
        public List<float[]> Chunks;
        public int Channels;
        public int SampleRate;
    }

    private struct TimestampSample
    {
        public float UnityRealtimeSeconds;
        public long UnixTimeNanoseconds;
    }

    private long _timestampOriginUnixNanoseconds;
    private long _timestampOriginStopwatchTicks;
    private static readonly long UnixEpochTicks = new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero).Ticks;

    private static readonly string[] ExpressionNames =
    {
        "BrowLowererL", "BrowLowererR",
        "CheekPuffL", "CheekPuffR", "CheekRaiserL", "CheekRaiserR", "CheekSuckL", "CheekSuckR",
        "ChinRaiserB", "ChinRaiserT",
        "DimplerL", "DimplerR",
        "EyesClosedL", "EyesClosedR",
        "EyesLookDownL", "EyesLookDownR", "EyesLookLeftL", "EyesLookLeftR",
        "EyesLookRightL", "EyesLookRightR", "EyesLookUpL", "EyesLookUpR",
        "InnerBrowRaiserL", "InnerBrowRaiserR",
        "JawDrop", "JawSidewaysLeft", "JawSidewaysRight", "JawThrust",
        "LidTightenerL", "LidTightenerR",
        "LipCornerDepressorL", "LipCornerDepressorR", "LipCornerPullerL", "LipCornerPullerR",
        "LipFunnelerLB", "LipFunnelerLT", "LipFunnelerRB", "LipFunnelerRT",
        "LipPressorL", "LipPressorR", "LipPuckerL", "LipPuckerR",
        "LipStretcherL", "LipStretcherR",
        "LipSuckLB", "LipSuckLT", "LipSuckRB", "LipSuckRT",
        "LipTightenerL", "LipTightenerR",
        "LipsToward",
        "LowerLipDepressorL", "LowerLipDepressorR",
        "MouthLeft", "MouthRight",
        "NoseWrinklerL", "NoseWrinklerR",
        "OuterBrowRaiserL", "OuterBrowRaiserR",
        "UpperLidRaiserL", "UpperLidRaiserR",
        "UpperLipRaiserL", "UpperLipRaiserR"
    };

    private void Awake()
    {
        DateTimeOffset timestampOriginUtc = DateTimeOffset.UtcNow;
        _timestampOriginUnixNanoseconds = (timestampOriginUtc.Ticks - UnixEpochTicks) * 100L;
        _timestampOriginStopwatchTicks = System.Diagnostics.Stopwatch.GetTimestamp();

        _csvThread = new Thread(CsvWriterLoop)
        {
            IsBackground = true,
            Name = "FacialDataRecorder CSV Writer"
        };
        _csvThread.Start();

        _audioThread = new Thread(AudioWriterLoop)
        {
            IsBackground = true,
            Name = "FacialDataRecorder Audio Writer"
        };
        _audioThread.Start();
    }

    private void Start()
    {
        OVRPlugin.systemDisplayFrequency = 90.0f;
        Debug.Log($"[FacialDataRecorder] Display frequency set to {OVRPlugin.systemDisplayFrequency}Hz");
        StartMicrophone();
    }

    public void StartRecording(int sentenceId, string mode)
    {
        if (_isRecording)
            return;

        ParticipantSession.EnsureParticipantFolders();

        string normalizedMode = ParticipantSession.NormalizeMode(mode);
        string tag = $"s{sentenceId:D4}_{normalizedMode.ToLowerInvariant()}";
        string facePath = Path.Combine(ParticipantSession.GetFaceFolderPath(normalizedMode), $"{tag}_face.csv");
        string imuPath = Path.Combine(ParticipantSession.GetImuFolderPath(normalizedMode), $"{tag}_imu.csv");

        EnqueueCsv(new CsvCommand
        {
            Type = CsvCommandType.StartSession,
            FacePath = facePath,
            ImuPath = imuPath,
            FaceHeader = BuildFaceHeader(),
            ImuHeader = BuildImuHeader()
        });

        _captureAudio = normalizedMode == ParticipantSession.VoiceMode;
        if (_captureAudio)
        {
            _audioPath = Path.Combine(ParticipantSession.GetParticipantRootPath(), normalizedMode, $"{tag}_audio.wav");
            PrepareAudioCapture();
        }

        _isRecording = true;
        Debug.Log($"[FacialDataRecorder] Recording started: {tag}");
    }

    public void StopRecording()
    {
        if (!_isRecording)
            return;

        if (_captureAudio)
            CaptureMicrophoneSamples(true);
        _isRecording = false;

        EnqueueCsv(new CsvCommand { Type = CsvCommandType.StopSession });
        if (_captureAudio)
            QueueAudioSave();
        _captureAudio = false;

        Debug.Log("[FacialDataRecorder] Recording stopped.");
    }

    private void StartMicrophone()
    {
        if (_micClip != null && Microphone.IsRecording(_micDevice))
            return;

        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("[FacialDataRecorder] No microphone found.");
            return;
        }

        _micClip = Microphone.Start(_micDevice, true, MicrophoneBufferSeconds, AudioSampleRate);
        _micReadPos = Microphone.GetPosition(_micDevice);
        _micChannels = _micClip != null ? _micClip.channels : 1;
        Debug.Log($"[FacialDataRecorder] Microphone started: {AudioSampleRate} Hz, {_micChannels} ch");

        StartCoroutine(MicrophoneSelfTest());
    }

    private System.Collections.IEnumerator MicrophoneSelfTest()
    {
        yield return new WaitForSeconds(0.7f);

        if (_micClip == null || !Microphone.IsRecording(_micDevice))
        {
            Debug.LogError("[FacialDataRecorder] Microphone is not recording.");
            yield break;
        }

        int probeFrames = AudioSampleRate / 2;
        int probeSamples = probeFrames * _micChannels;
        var probe = new float[probeSamples];
        int clipFrames = _micClip.samples;
        int currentPos = Microphone.GetPosition(_micDevice);
        int startPos = (currentPos - probeFrames + clipFrames) % clipFrames;
        _micClip.GetData(probe, startPos);

        float peak = 0f;
        for (int i = 0; i < probe.Length; i++)
        {
            float v = probe[i];
            float a = v < 0 ? -v : v;
            if (a > peak) peak = a;
        }

        const float SilentThreshold = 1e-4f;
        if (peak < SilentThreshold)
            Debug.LogError($"[FacialDataRecorder] Microphone returns silence (peak {peak:E3}).");
    }

    private void PrepareAudioCapture()
    {
        StartMicrophone();
        _audioChunks = new List<float[]>(512);
        _micChannels = _micClip != null ? _micClip.channels : 1;

        if (_micClip == null)
        {
            _micReadPos = 0;
            return;
        }

        _micReadPos = Microphone.GetPosition(_micDevice);
    }

    private void CaptureMicrophoneSamples(bool flushPartial)
    {
        if (_micClip == null || !Microphone.IsRecording(_micDevice))
            return;

        int currentPos = Microphone.GetPosition(_micDevice);
        if (currentPos < 0 || currentPos == _micReadPos)
            return;

        int clipFrames = _micClip.samples;
        int availableFrames = currentPos > _micReadPos
            ? currentPos - _micReadPos
            : clipFrames - _micReadPos + currentPos;

        int framesToRead = flushPartial
            ? availableFrames
            : availableFrames - (availableFrames % MicrophoneChunkFrames);

        while (framesToRead > 0)
        {
            int frames = Math.Min(framesToRead, MicrophoneChunkFrames);
            frames = Math.Min(frames, clipFrames - _micReadPos);
            AppendMicrophoneChunk(_micReadPos, frames);
            _micReadPos = (_micReadPos + frames) % clipFrames;
            framesToRead -= frames;
        }
    }

    private void AppendMicrophoneChunk(int startFrame, int frameCount)
    {
        if (frameCount <= 0)
            return;

        int sampleCount = frameCount * _micChannels;
        var chunk = new float[sampleCount];
        _micClip.GetData(chunk, startFrame);
        _audioChunks.Add(chunk);
    }

    private void QueueAudioSave()
    {
        if (_audioChunks.Count == 0)
        {
            Debug.LogWarning("[FacialDataRecorder] No audio data captured.");
            _audioPath = null;
            return;
        }

        float peak = 0f;
        foreach (var c in _audioChunks)
        {
            for (int i = 0; i < c.Length; i++)
            {
                float v = c[i];
                float a = v < 0 ? -v : v;
                if (a > peak) peak = a;
            }
        }
        const float SilentThreshold = 1e-4f;
        if (peak < SilentThreshold)
            Debug.LogWarning($"[FacialDataRecorder] Silent audio: {Path.GetFileName(_audioPath)} (peak {peak:E3})");

        EnqueueAudio(new AudioSaveCommand
        {
            Path = _audioPath,
            Chunks = _audioChunks,
            Channels = _micChannels,
            SampleRate = AudioSampleRate
        });

        _audioChunks = new List<float[]>(512);
        _audioPath = null;
    }

    private static void SaveWav(string path, List<float[]> chunks, int channels, int sampleRate)
    {
        long totalSamples = 0;
        foreach (var chunk in chunks)
            totalSamples += chunk.Length;

        if (totalSamples == 0)
            return;

        using (var stream = new FileStream(path, FileMode.Create))
        using (var writer = new BinaryWriter(stream))
        {
            int bitsPerSample = 16;
            int byteRate = sampleRate * channels * bitsPerSample / 8;
            int dataSize = checked((int)(totalSamples * bitsPerSample / 8));

            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + dataSize);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write((short)(channels * bitsPerSample / 8));
            writer.Write((short)bitsPerSample);

            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(dataSize);

            foreach (var chunk in chunks)
            {
                for (int i = 0; i < chunk.Length; i++)
                {
                    float clamped = chunk[i];
                    if (clamped > 1f) clamped = 1f;
                    else if (clamped < -1f) clamped = -1f;
                    writer.Write((short)(clamped * 32767f));
                }
            }
        }
    }

    public bool GetFaceTrackingStatus(out float confidenceLower, out float confidenceUpper, out bool isValid)
    {
        confidenceLower = _lastConfidenceLower;
        confidenceUpper = _lastConfidenceUpper;
        isValid = _lastIsValid;
        return _lastIsValid;
    }

    private string BuildFaceHeader()
    {
        _sb.Clear();
        _sb.Append("Timestamp,UnixTimeNs");
        foreach (var name in ExpressionNames)
        {
            _sb.Append(',');
            _sb.Append(name);
        }
        _sb.Append(",ConfidenceLower,ConfidenceUpper,EyeFollowingValid");
        return _sb.ToString();
    }

    private static string BuildImuHeader()
    {
        return "Timestamp,UnixTimeNs,HeadPosX,HeadPosY,HeadPosZ,HeadRotX,HeadRotY,HeadRotZ,HeadRotW,HeadVelX,HeadVelY,HeadVelZ,HeadAngVelX,HeadAngVelY,HeadAngVelZ";
    }

    private bool BuildFaceLine(TimestampSample timestamp, out string line)
    {
        line = null;
        if (!OVRPlugin.GetFaceState2(OVRPlugin.Step.Render, -1, ref _faceState))
        {
            _lastConfidenceLower = 0f;
            _lastConfidenceUpper = 0f;
            _lastIsValid = false;
            return false;
        }

        _lastIsValid = _faceState.Status.IsValid;
        if (_faceState.ExpressionWeightConfidences != null && _faceState.ExpressionWeightConfidences.Length >= 2)
        {
            _lastConfidenceLower = _faceState.ExpressionWeightConfidences[0];
            _lastConfidenceUpper = _faceState.ExpressionWeightConfidences[1];
        }

        if (!_lastIsValid)
            return false;

        _sb.Clear();
        _sb.Append(timestamp.UnityRealtimeSeconds.ToString("F6"));
        _sb.Append(','); _sb.Append(timestamp.UnixTimeNanoseconds);
        for (int i = 0; i < ExpressionNames.Length; i++)
        {
            _sb.Append(',');
            _sb.Append(_faceState.ExpressionWeights[i].ToString("F6"));
        }
        _sb.Append(','); _sb.Append(_faceState.ExpressionWeightConfidences[0].ToString("F6"));
        _sb.Append(','); _sb.Append(_faceState.ExpressionWeightConfidences[1].ToString("F6"));
        _sb.Append(','); _sb.Append(_faceState.Status.IsEyeFollowingBlendshapesValid ? "1" : "0");
        line = _sb.ToString();
        return true;
    }

    private string BuildImuLine(TimestampSample timestamp)
    {
        var pose = OVRPlugin.GetNodePoseStateRaw(OVRPlugin.Node.Head, OVRPlugin.Step.Render);
        var angularVelocity = ToTrackingFrame(pose.Pose.Orientation, pose.AngularVelocity);

        _sb.Clear();
        _sb.Append(timestamp.UnityRealtimeSeconds.ToString("F6"));
        _sb.Append(','); _sb.Append(timestamp.UnixTimeNanoseconds);
        _sb.Append(','); _sb.Append(pose.Pose.Position.x.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Pose.Position.y.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Pose.Position.z.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Pose.Orientation.x.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Pose.Orientation.y.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Pose.Orientation.z.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Pose.Orientation.w.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Velocity.x.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Velocity.y.ToString("F6"));
        _sb.Append(','); _sb.Append(pose.Velocity.z.ToString("F6"));
        _sb.Append(','); _sb.Append(angularVelocity.x.ToString("F6"));
        _sb.Append(','); _sb.Append(angularVelocity.y.ToString("F6"));
        _sb.Append(','); _sb.Append(angularVelocity.z.ToString("F6"));
        return _sb.ToString();
    }

    private static OVRPlugin.Vector3f ToTrackingFrame(OVRPlugin.Quatf q, OVRPlugin.Vector3f v)
    {
        float n = (float)Math.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        float x = q.x / n, y = q.y / n, z = q.z / n, w = q.w / n;
        float tx = 2f * (y * v.z - z * v.y);
        float ty = 2f * (z * v.x - x * v.z);
        float tz = 2f * (x * v.y - y * v.x);
        return new OVRPlugin.Vector3f
        {
            x = v.x + w * tx + (y * tz - z * ty),
            y = v.y + w * ty + (z * tx - x * tz),
            z = v.z + w * tz + (x * ty - y * tx)
        };
    }

    private TimestampSample CaptureTimestamp()
    {
        long elapsedStopwatchTicks = System.Diagnostics.Stopwatch.GetTimestamp() - _timestampOriginStopwatchTicks;
        long elapsedNanoseconds = (long)Math.Round(
            elapsedStopwatchTicks * 1000000000.0 / System.Diagnostics.Stopwatch.Frequency);

        return new TimestampSample
        {
            UnityRealtimeSeconds = Time.realtimeSinceStartup,
            UnixTimeNanoseconds = _timestampOriginUnixNanoseconds + elapsedNanoseconds
        };
    }

    private void LateUpdate()
    {
        DrainWorkerErrors();

        if (!_isRecording)
            return;

        TimestampSample ts = CaptureTimestamp();
        bool faceOk = BuildFaceLine(ts, out string faceLine);
        string imuLine = BuildImuLine(ts);

        if (faceOk)
        {
            EnqueueCsv(new CsvCommand { Type = CsvCommandType.FaceRow, Line = faceLine });
            EnqueueCsv(new CsvCommand { Type = CsvCommandType.ImuRow, Line = imuLine });
        }

        if (_captureAudio)
            CaptureMicrophoneSamples(false);
    }

    private void EnqueueCsv(CsvCommand command)
    {
        if (_csvQueue.IsAddingCompleted)
            return;

        try
        {
            _csvQueue.Add(command);
        }
        catch (InvalidOperationException)
        {
        }
    }

    private void EnqueueAudio(AudioSaveCommand command)
    {
        if (_audioQueue.IsAddingCompleted)
            return;

        try
        {
            _audioQueue.Add(command);
        }
        catch (InvalidOperationException)
        {
        }
    }

    private void CsvWriterLoop()
    {
        StreamWriter faceWriter = null;
        StreamWriter imuWriter = null;

        try
        {
            foreach (var command in _csvQueue.GetConsumingEnumerable())
            {
                try
                {
                    switch (command.Type)
                    {
                        case CsvCommandType.StartSession:
                            CloseWriter(ref faceWriter);
                            CloseWriter(ref imuWriter);
                            Directory.CreateDirectory(Path.GetDirectoryName(command.FacePath));
                            Directory.CreateDirectory(Path.GetDirectoryName(command.ImuPath));
                            faceWriter = new StreamWriter(command.FacePath, false, Encoding.UTF8);
                            imuWriter = new StreamWriter(command.ImuPath, false, Encoding.UTF8);
                            faceWriter.WriteLine(command.FaceHeader);
                            imuWriter.WriteLine(command.ImuHeader);
                            break;

                        case CsvCommandType.FaceRow:
                            faceWriter?.WriteLine(command.Line);
                            break;

                        case CsvCommandType.ImuRow:
                            imuWriter?.WriteLine(command.Line);
                            break;

                        case CsvCommandType.StopSession:
                            CloseWriter(ref faceWriter);
                            CloseWriter(ref imuWriter);
                            break;
                    }
                }
                catch (Exception e)
                {
                    _workerErrors.Enqueue($"[FacialDataRecorder] CSV writer error: {e.Message}");
                }
            }
        }
        finally
        {
            CloseWriter(ref faceWriter);
            CloseWriter(ref imuWriter);
        }
    }

    private void AudioWriterLoop()
    {
        foreach (var command in _audioQueue.GetConsumingEnumerable())
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(command.Path));
                SaveWav(command.Path, command.Chunks, command.Channels, command.SampleRate);
            }
            catch (Exception e)
            {
                _workerErrors.Enqueue($"[FacialDataRecorder] Audio writer error: {e.Message}");
            }
        }
    }

    private static void CloseWriter(ref StreamWriter writer)
    {
        if (writer == null)
            return;

        writer.Flush();
        writer.Close();
        writer = null;
    }

    private void DrainWorkerErrors()
    {
        while (_workerErrors.TryDequeue(out string message))
            Debug.LogError(message);
    }

    private void OnDestroy()
    {
        StopRecording();

        _csvQueue.CompleteAdding();
        _audioQueue.CompleteAdding();

        if (_csvThread != null && _csvThread.IsAlive && !_csvThread.Join(3000))
            Debug.LogWarning("[FacialDataRecorder] CSV writer did not finish before shutdown.");

        if (_audioThread != null && _audioThread.IsAlive && !_audioThread.Join(3000))
            Debug.LogWarning("[FacialDataRecorder] Audio writer did not finish before shutdown.");

        if (_micClip != null && Microphone.IsRecording(_micDevice))
            Microphone.End(_micDevice);
    }
}
