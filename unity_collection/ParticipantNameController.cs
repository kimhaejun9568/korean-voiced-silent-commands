using UnityEngine;

public class ParticipantNameController : MonoBehaviour
{
    [SerializeField] private string participantName;

    private void Awake()
    {
        ApplyParticipantName();
    }

    public void ApplyParticipantName()
    {
        ParticipantSession.SetParticipantName(participantName);
        ParticipantSession.EnsureParticipantFolders();
    }
}
