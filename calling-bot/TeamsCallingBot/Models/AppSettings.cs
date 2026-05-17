namespace TeamsCallingBot.Models;

public class AppSettings
{
    public string MicrosoftAppId { get; set; } = string.Empty;
    public string MicrosoftAppPassword { get; set; } = string.Empty;
    public string TenantId { get; set; } = string.Empty;
    public string CallbackUri { get; set; } = string.Empty;
    public string DefaultMeetingId { get; set; } = string.Empty;
    public string DefaultMeetingPasscode { get; set; } = string.Empty;
    public string? DefaultChatThreadId { get; set; }
    public string? DefaultChatMessageId { get; set; }
    public string ServiceFqdn { get; set; } = string.Empty;
    public string MediaCertificateThumbprint { get; set; } = string.Empty;
    public int MediaInternalPort { get; set; } = 8445;
    public int MediaInstanceExternalPort { get; set; } = 8445;

    /// <summary>
    /// When true, bootstraps the Graph Communications client and Skype media platform (requires a valid
    /// <see cref="MediaCertificateThumbprint"/> in the machine store). Set false for local /api/messages tests without a cert.
    /// </summary>
    public bool InitializeGraphCommunications { get; set; }

    /// <summary>
    /// When true, writes the legacy single-file <c>captures\call_*.wav</c> alongside chunked output.
    /// Default false (chunks only under <c>captures\chunks</c>).
    /// </summary>
    public bool EnableFullCallWavCapture { get; set; }
}
