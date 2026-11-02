using System;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

public class SentenceManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private FacialDataRecorder recorder;

    [Header("Corpus")]
    [SerializeField] private string corpusFileName = "corpus_1000_xr.csv";
    [SerializeField] private int sentenceLimit = 1000;

    [Header("UI")]
    [SerializeField] private TextMeshProUGUI sentenceText;
    [SerializeField] private TextMeshProUGUI modeText;
    [SerializeField] private TextMeshProUGUI progressText;
    [SerializeField] private TextMeshProUGUI statusText;
    [SerializeField] private Image recordingDot;

    private enum Phase { WaitingVoiced, RecordingVoiced, WaitingSilent, RecordingSilent }

    private struct Sentence
    {
        public int id;
        public string text;
    }

    [Serializable]
    private class ProgressData
    {
        public int seed;
        public int currentIndex;
        public int phase;
        public int[] shuffledOrder;
    }

    private readonly List<Sentence> _sentences = new List<Sentence>();
    private int[] _shuffledOrder;
    private int _currentIndex;
    private Phase _phase = Phase.WaitingVoiced;
    private float _joystickCooldown;
    private int _sessionSeed;
    private bool _allCompleted;
    private bool _aPrev;
    private int _faceTrackingStartAttempts;
    private float _nextFaceTrackingStartTime;
    private bool _hadXrFocus;
    private float _faceTrackingInvalidSince;
    private float _faceTrackingRestartCooldownUntil;
    private OVRPlugin.FaceState _faceTrackingWatchdogState;
    private const float FaceTrackingStartRetrySeconds = 3f;
    private const float FaceTrackingFocusWarmupSeconds = 1f;
    private const float FaceTrackingInvalidRestartSeconds = 4f;
    private const float FaceTrackingRestartCooldownSeconds = 6f;

    private string ProgressFilePath =>
        Path.Combine(ParticipantSession.GetParticipantRootPath(), "progress.json");

    private void Start()
    {
        ParticipantSession.EnsureParticipantFolders();
        LoadCSV();

        if (TryLoadProgress())
        {
            Debug.Log($"[SentenceManager] Resumed: index={_currentIndex}/{_sentences.Count}, phase={_phase}, seed={_sessionSeed}");
        }
        else
        {
            ShuffleOrder();
            SaveSessionMeta();
            SaveProgress();
            Debug.Log($"[SentenceManager] New session: {_sentences.Count} sentences, seed={_sessionSeed}");
        }

        UpdateUI();
    }

    private void LoadCSV()
    {
        string path = Path.Combine(Application.streamingAssetsPath, corpusFileName);
        if (!File.Exists(path))
        {
            Debug.LogError($"[SentenceManager] CSV not found: {path}");
            return;
        }

        string[] lines = File.ReadAllLines(path);
        for (int i = 1; i < lines.Length; i++)
        {
            if (sentenceLimit > 0 && _sentences.Count >= sentenceLimit)
                break;

            string line = lines[i];
            if (string.IsNullOrWhiteSpace(line))
                continue;

            int firstComma = line.IndexOf(',');
            if (firstComma < 0)
                continue;

            int secondComma = line.IndexOf(',', firstComma + 1);
            if (secondComma < 0)
                secondComma = line.Length;

            if (!int.TryParse(line.Substring(0, firstComma), out int id))
                continue;

            string text = line.Substring(firstComma + 1, secondComma - firstComma - 1);
            _sentences.Add(new Sentence { id = id, text = text });
        }

        if (sentenceLimit > 0 && _sentences.Count < sentenceLimit)
            Debug.LogWarning($"[SentenceManager] Loaded {_sentences.Count}/{sentenceLimit} sentences from {path}");
    }

    private void ShuffleOrder()
    {
        _sessionSeed = Environment.TickCount;
        var rng = new System.Random(_sessionSeed);

        _shuffledOrder = new int[_sentences.Count];
        for (int i = 0; i < _shuffledOrder.Length; i++)
            _shuffledOrder[i] = i;

        for (int i = _shuffledOrder.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            int tmp = _shuffledOrder[i];
            _shuffledOrder[i] = _shuffledOrder[j];
            _shuffledOrder[j] = tmp;
        }

        _currentIndex = 0;
    }

    private void SaveSessionMeta()
    {
        string dir = ParticipantSession.GetParticipantRootPath();
        Directory.CreateDirectory(dir);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string path = Path.Combine(dir, $"session_{timestamp}.json");

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"participantName\": \"{EscapeJsonString(ParticipantSession.ParticipantName)}\",");
        sb.AppendLine($"  \"seed\": {_sessionSeed},");
        sb.AppendLine($"  \"timestamp\": \"{timestamp}\",");
        sb.AppendLine($"  \"sentenceCount\": {_sentences.Count},");
        sb.Append("  \"shuffledOrder\": [");
        for (int i = 0; i < _shuffledOrder.Length; i++)
        {
            if (i > 0) sb.Append(",");
            sb.Append(_shuffledOrder[i]);
        }
        sb.AppendLine("]");
        sb.AppendLine("}");

        File.WriteAllText(path, sb.ToString());
        Debug.Log($"[SentenceManager] Session meta saved: {path}");
    }

    private static string EscapeJsonString(string value)
    {
        return string.IsNullOrEmpty(value)
            ? string.Empty
            : value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private bool TryLoadProgress()
    {
        if (!File.Exists(ProgressFilePath))
            return false;

        try
        {
            string json = File.ReadAllText(ProgressFilePath);
            var data = JsonUtility.FromJson<ProgressData>(json);

            if (data.shuffledOrder == null || data.shuffledOrder.Length != _sentences.Count)
                return false;
            if (data.currentIndex < 0 || data.currentIndex >= _sentences.Count)
                return false;

            _sessionSeed = data.seed;
            _currentIndex = data.currentIndex;
            _shuffledOrder = data.shuffledOrder;

            _phase = (Phase)data.phase;
            if (_phase == Phase.RecordingVoiced) _phase = Phase.WaitingVoiced;
            if (_phase == Phase.RecordingSilent) _phase = Phase.WaitingSilent;

            return true;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[SentenceManager] Progress load failed: {e.Message}");
            return false;
        }
    }

    private void SaveProgress()
    {
        var data = new ProgressData
        {
            seed = _sessionSeed,
            currentIndex = _currentIndex,
            phase = (int)_phase,
            shuffledOrder = _shuffledOrder
        };

        string dir = ParticipantSession.GetParticipantRootPath();
        Directory.CreateDirectory(dir);
        File.WriteAllText(ProgressFilePath, JsonUtility.ToJson(data));
    }

    private void DeleteProgress()
    {
        if (File.Exists(ProgressFilePath))
            File.Delete(ProgressFilePath);
    }

    private void Update()
    {
        HandleAButton();
        HandleJoystick();
        UpdateTrackingStatus();
    }

    private static bool ReadXrButton(XRNode hand, InputFeatureUsage<bool> usage)
    {
        var device = InputDevices.GetDeviceAtXRNode(hand);
        return device.isValid && device.TryGetFeatureValue(usage, out bool value) && value;
    }

    private static Vector2 ReadXrAxis2D(XRNode hand, InputFeatureUsage<Vector2> usage)
    {
        var device = InputDevices.GetDeviceAtXRNode(hand);
        return device.isValid && device.TryGetFeatureValue(usage, out Vector2 value) ? value : Vector2.zero;
    }

    private void HandleAButton()
    {
        if (_sentences.Count == 0 || _allCompleted)
        {
            _aPrev = ReadXrButton(XRNode.RightHand, CommonUsages.primaryButton);
            return;
        }

        bool aNow = ReadXrButton(XRNode.RightHand, CommonUsages.primaryButton);
        bool aDown = aNow && !_aPrev;
        bool aUp = !aNow && _aPrev;
        _aPrev = aNow;

        if (aDown)
        {
            recorder.GetFaceTrackingStatus(out _, out _, out bool isValid);
            if (!isValid)
                return;

            var sentence = _sentences[_shuffledOrder[_currentIndex]];

            if (_phase == Phase.WaitingVoiced)
            {
                recorder.StartRecording(sentence.id, ParticipantSession.VoiceMode);
                _phase = Phase.RecordingVoiced;
                UpdateUI();
            }
            else if (_phase == Phase.WaitingSilent)
            {
                recorder.StartRecording(sentence.id, ParticipantSession.SilentMode);
                _phase = Phase.RecordingSilent;
                UpdateUI();
            }
        }

        if (aUp)
        {
            if (_phase == Phase.RecordingVoiced)
            {
                recorder.StopRecording();
                _phase = Phase.WaitingSilent;
                SaveProgress();
                UpdateUI();
            }
            else if (_phase == Phase.RecordingSilent)
            {
                recorder.StopRecording();
                AdvanceToNext();
                if (!_allCompleted) SaveProgress();
                UpdateUI();
            }
        }
    }

    private void HandleJoystick()
    {
        if (_phase != Phase.WaitingVoiced)
            return;

        _joystickCooldown -= Time.deltaTime;
        if (_joystickCooldown > 0)
            return;

        Vector2 stick = ReadXrAxis2D(XRNode.RightHand, CommonUsages.primary2DAxis);

        if (stick.x > 0.7f)
        {
            if (_currentIndex < _shuffledOrder.Length - 1)
            {
                _currentIndex++;
                SaveProgress();
                UpdateUI();
            }
            _joystickCooldown = 0.3f;
        }
        else if (stick.x < -0.7f)
        {
            if (_currentIndex > 0)
            {
                _currentIndex--;
                SaveProgress();
                UpdateUI();
            }
            _joystickCooldown = 0.3f;
        }
    }

    private void AdvanceToNext()
    {
        if (_currentIndex < _shuffledOrder.Length - 1)
        {
            _currentIndex++;
            _phase = Phase.WaitingVoiced;
        }
        else
        {
            _allCompleted = true;
            DeleteProgress();
            Debug.Log("[SentenceManager] All sentences completed!");
        }
    }

    private void UpdateUI()
    {
        if (_sentences.Count == 0)
        {
            if (sentenceText != null) sentenceText.text = "CSV load failed";
            return;
        }

        if (_allCompleted)
        {
            if (sentenceText != null) sentenceText.text = "Collection complete";
            if (modeText != null) { modeText.text = "Complete"; modeText.color = Color.cyan; }
            if (progressText != null) progressText.text = $"Command {_sentences.Count}/{_sentences.Count}";
            if (recordingDot != null) recordingDot.color = new Color(1, 1, 1, 0.2f);
            return;
        }

        var sentence = _sentences[_shuffledOrder[_currentIndex]];

        if (sentenceText != null)
            sentenceText.text = sentence.text;

        if (modeText != null)
        {
            bool isVoiced = _phase == Phase.WaitingVoiced || _phase == Phase.RecordingVoiced;
            modeText.text = isVoiced ? ParticipantSession.VoiceMode : ParticipantSession.SilentMode;
            modeText.color = isVoiced ? Color.green : Color.yellow;
        }

        if (progressText != null)
            progressText.text = $"Command {_currentIndex + 1}/{_sentences.Count}";

        if (recordingDot != null)
        {
            bool recording = _phase == Phase.RecordingVoiced || _phase == Phase.RecordingSilent;
            recordingDot.color = recording ? Color.red : new Color(1, 1, 1, 0.2f);
        }
    }

    private void UpdateTrackingStatus()
    {
        if (recorder == null)
            return;

        bool hasXrFocus = OVRManager.hasVrFocus && OVRManager.hasInputFocus;
        EnsureFaceTrackingStarted();

        if (_phase == Phase.RecordingVoiced || _phase == Phase.RecordingSilent)
        {
            if (statusText != null)
            {
                statusText.text = "Recording... (release A to stop)";
                statusText.color = Color.green;
            }
            return;
        }

        if (!hasXrFocus)
        {
            if (statusText != null)
            {
                statusText.text = "Face Tracking: waiting for XR focus";
                statusText.color = Color.yellow;
            }
            return;
        }

        recorder.GetFaceTrackingStatus(out float lower, out float upper, out bool isValid);

        if (statusText == null)
            return;

        if (!isValid)
        {
            statusText.text = "Face Tracking: unavailable";
            statusText.color = Color.red;
        }
        else if (lower < 0.5f || upper < 0.5f)
        {
            statusText.text = $"Face Tracking: LOW  L:{lower:F2} U:{upper:F2}";
            statusText.color = Color.yellow;
        }
        else
        {
            statusText.text = $"Face Tracking: OK  L:{lower:F2} U:{upper:F2}";
            statusText.color = Color.green;
        }
    }

    private void EnsureFaceTrackingStarted()
    {
        if (Time.unscaledTime < 1.5f || Time.unscaledTime < _nextFaceTrackingStartTime)
            return;

        if (!OVRPlugin.faceTracking2Supported || !OVRManager.isHmdPresent || !OVRManager.hasVrFocus || !OVRManager.hasInputFocus)
        {
            if (_hadXrFocus)
            {
                _hadXrFocus = false;
                _faceTrackingStartAttempts = 0;
                _faceTrackingInvalidSince = 0f;
                _nextFaceTrackingStartTime = Time.unscaledTime + FaceTrackingFocusWarmupSeconds;
            }
            return;
        }

        if (!_hadXrFocus)
        {
            _hadXrFocus = true;
            _faceTrackingStartAttempts = 0;
            _faceTrackingInvalidSince = 0f;
            _nextFaceTrackingStartTime = Time.unscaledTime + FaceTrackingFocusWarmupSeconds;
            return;
        }

        if (OVRPlugin.faceTracking2Enabled)
        {
            EnsureFaceExpressionComponentsEnabled();

            if (HasValidFaceTrackingSample())
            {
                _faceTrackingInvalidSince = 0f;
                return;
            }

            if (_faceTrackingInvalidSince <= 0f)
            {
                _faceTrackingInvalidSince = Time.unscaledTime;
                return;
            }

            if (Time.unscaledTime - _faceTrackingInvalidSince >= FaceTrackingInvalidRestartSeconds &&
                Time.unscaledTime >= _faceTrackingRestartCooldownUntil)
            {
                OVRPlugin.StopFaceTracking2();
                _faceTrackingStartAttempts = 0;
                _faceTrackingInvalidSince = 0f;
                _faceTrackingRestartCooldownUntil = Time.unscaledTime + FaceTrackingRestartCooldownSeconds;
                _nextFaceTrackingStartTime = Time.unscaledTime + 0.5f;
                Debug.LogWarning("[SentenceManager] Face tracking invalid; restarting.");
            }

            return;
        }

        var dataSources = GetRequestedFaceTrackingDataSources();

        if (dataSources.Length == 0)
        {
            Debug.LogWarning("[SentenceManager] No face tracking data source is requested in OculusRuntimeSettings.");
            return;
        }

        _faceTrackingStartAttempts++;
        _nextFaceTrackingStartTime = Time.unscaledTime + FaceTrackingStartRetrySeconds;

        OVRPlugin.StopFaceTracking2();
        bool started = OVRPlugin.StartFaceTracking2(dataSources);
        EnsureFaceExpressionComponentsEnabled();
        Debug.Log($"[SentenceManager] Face tracking start attempt {_faceTrackingStartAttempts}: started={started}");
    }

    private bool HasValidFaceTrackingSample()
    {
        if (OVRPlugin.GetFaceState2(OVRPlugin.Step.Render, -1, ref _faceTrackingWatchdogState) &&
            _faceTrackingWatchdogState.Status.IsValid)
        {
            return true;
        }

        var expressions = Resources.FindObjectsOfTypeAll<OVRFaceExpressions>();
        foreach (var expression in expressions)
        {
            if (expression == null || !expression.gameObject.scene.IsValid() || !expression.gameObject.activeInHierarchy)
                continue;

            if (expression.enabled && expression.ValidExpressions)
                return true;
        }

        return false;
    }

    private void EnsureFaceExpressionComponentsEnabled()
    {
        var expressions = Resources.FindObjectsOfTypeAll<OVRFaceExpressions>();
        foreach (var expression in expressions)
        {
            if (expression == null || !expression.gameObject.scene.IsValid() || !expression.gameObject.activeInHierarchy)
                continue;

            if (!expression.enabled)
                expression.enabled = true;
        }
    }

    private static OVRPlugin.FaceTrackingDataSource[] GetRequestedFaceTrackingDataSources()
    {
        var runtimeSettings = OVRRuntimeSettings.GetRuntimeSettings();
        bool visual = runtimeSettings.RequestsVisualFaceTracking;
        bool audio = runtimeSettings.RequestsAudioFaceTracking;

#if UNITY_EDITOR && UNITY_STANDALONE
        if (visual)
            return new[] { OVRPlugin.FaceTrackingDataSource.Visual };
#endif

        if (visual && audio)
            return new[] { OVRPlugin.FaceTrackingDataSource.Visual, OVRPlugin.FaceTrackingDataSource.Audio };
        if (visual)
            return new[] { OVRPlugin.FaceTrackingDataSource.Visual };
        if (audio)
            return new[] { OVRPlugin.FaceTrackingDataSource.Audio };

        return Array.Empty<OVRPlugin.FaceTrackingDataSource>();
    }
}
