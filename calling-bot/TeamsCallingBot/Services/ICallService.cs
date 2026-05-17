using Microsoft.Graph.Communications.Calls;
using Microsoft.Graph.Communications.Client;

namespace TeamsCallingBot.Services;

public interface ICallService : IDisposable
{
    /// <summary>Null when graph communications initialization is disabled or skipped in configuration.</summary>
    ICommunicationsClient? Client { get; }

    bool IsGraphClientReady { get; }

    void Initialize();
    Task<ICall> JoinMeetingAsync(
        string? meetingId,
        string? passcode,
        string? threadId,
        string? messageId,
        CancellationToken cancellationToken = default);
}
