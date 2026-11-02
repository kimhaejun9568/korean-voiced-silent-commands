using System.IO;
using System.Text;
using UnityEngine;

public static class ParticipantSession
{
    public const string PlayerPrefsKey = "ParticipantName";
    public const string VoiceMode = "Voice";
    public const string SilentMode = "Silent";

    private const string DefaultParticipantName = "Unnamed";
    private const string DataRootFolderName = "Data";

    public static string ParticipantName
    {
        get
        {
            string value = PlayerPrefs.GetString(PlayerPrefsKey, DefaultParticipantName);
            return string.IsNullOrWhiteSpace(value) ? DefaultParticipantName : value.Trim();
        }
    }

    public static void SetParticipantName(string participantName)
    {
        string value = string.IsNullOrWhiteSpace(participantName)
            ? DefaultParticipantName
            : participantName.Trim();

        PlayerPrefs.SetString(PlayerPrefsKey, value);
        PlayerPrefs.Save();
    }

    public static string GetParticipantRootPath()
    {
        return Path.Combine(Application.persistentDataPath, DataRootFolderName, GetSafeParticipantFolderName());
    }

    public static string GetFaceFolderPath(string mode)
    {
        return Path.Combine(GetParticipantRootPath(), NormalizeMode(mode), "Face");
    }

    public static string GetImuFolderPath(string mode)
    {
        return Path.Combine(GetParticipantRootPath(), NormalizeMode(mode), "IMU");
    }

    public static void EnsureParticipantFolders()
    {
        Directory.CreateDirectory(GetFaceFolderPath(VoiceMode));
        Directory.CreateDirectory(GetImuFolderPath(VoiceMode));
        Directory.CreateDirectory(GetFaceFolderPath(SilentMode));
        Directory.CreateDirectory(GetImuFolderPath(SilentMode));
    }

    public static string NormalizeMode(string mode)
    {
        if (string.Equals(mode, SilentMode, System.StringComparison.OrdinalIgnoreCase))
            return SilentMode;

        return VoiceMode;
    }

    private static string GetSafeParticipantFolderName()
    {
        string participantName = ParticipantName;
        char[] invalidChars = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(participantName.Length);

        foreach (char ch in participantName)
        {
            bool invalid = false;
            for (int i = 0; i < invalidChars.Length; i++)
            {
                if (ch == invalidChars[i])
                {
                    invalid = true;
                    break;
                }
            }

            sb.Append(invalid ? '_' : ch);
        }

        string safeName = sb.ToString().Trim();
        return string.IsNullOrWhiteSpace(safeName) ? DefaultParticipantName : safeName;
    }
}
