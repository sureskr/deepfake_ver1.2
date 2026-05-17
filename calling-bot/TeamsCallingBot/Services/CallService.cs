using System.Net;
using System.Net.Sockets;
using Microsoft.Extensions.Options;
using Microsoft.Graph.Communications.Calls;
using Microsoft.Graph.Communications.Calls.Media;
using Microsoft.Graph.Communications.Client;
using Microsoft.Graph.Communications.Client.Authentication;
using Microsoft.Graph.Communications.Common.Telemetry;
using Microsoft.Graph.Models;
using Microsoft.Skype.Bots.Media;
using TeamsCallingBot.Authentication;
using TeamsCallingBot.Models;

namespace TeamsCallingBot.Services;

public class CallService : ICallService
{
    private readonly ILogger<CallService> _logger;
    private readonly AppSettings _settings;
    private readonly IGraphLogger _graphLogger;
    private readonly IRequestAuthenticationProvider _auth;
    private readonly IChunkMlSink _chunkMlSink;
    private readonly object _lock = new();
    private ICommunicationsClient? _client;
    public CallService(
        ILogger<CallService> logger,
        IOptions<AppSettings> options,
        IGraphLogger graphLogger,
        IRequestAuthenticationProvider auth,
        IChunkMlSink chunkMlSink)
    {
        _logger = logger;
        _settings = options.Value;
        _graphLogger = graphLogger;
        _auth = auth;
        _chunkMlSink = chunkMlSink;
    }

    public ICommunicationsClient? Client => _client;

    public bool IsGraphClientReady => _client != null;

    public void Initialize()
    {
        lock (_lock)
        {
            if (_client != null)
            {
                return;
            }

            if (!_settings.InitializeGraphCommunications)
            {
                _logger.LogWarning(
                    "InitializeGraphCommunications is false: Graph Communications + media platform not started. " +
                    "Set InitializeGraphCommunications to true, install a cert, and set MediaCertificateThumbprint to enable /api/calls processing and /test/join-meeting.");
                return;
            }

            if (string.IsNullOrWhiteSpace(_settings.MicrosoftAppId)
                || string.IsNullOrWhiteSpace(_settings.MicrosoftAppPassword))
            {
                throw new InvalidOperationException("MicrosoftAppId and MicrosoftAppPassword are required in appsettings.");
            }

            if (string.IsNullOrWhiteSpace(_settings.CallbackUri) || !Uri.TryCreate(_settings.CallbackUri, UriKind.Absolute, out var callback))
            {
                throw new InvalidOperationException("CallbackUri must be an absolute URL (e.g. https://your-fqdn/api/calls).");
            }

            if (string.IsNullOrWhiteSpace(_settings.MediaCertificateThumbprint))
            {
                throw new InvalidOperationException(
                    "App-hosted media requires a TLS certificate. Set MediaCertificateThumbprint (and install the cert in the machine store) on the Windows VM.");
            }

            var serviceFqdn = string.IsNullOrWhiteSpace(_settings.ServiceFqdn)
                ? callback.Host
                : _settings.ServiceFqdn;
            if (string.IsNullOrWhiteSpace(serviceFqdn))
            {
                throw new InvalidOperationException(
                    "Set ServiceFqdn in appsettings, or use a CallbackUri with a valid host (public FQDN for media, e.g. your VM).");
            }

            _logger.LogInformation("Initializing Microsoft Graph Communications client (app-hosted media, audio)…");

            var name = GetType().Assembly.GetName().Name ?? "TeamsCallingBot";
            var builder = new CommunicationsClientBuilder(
                name,
                _settings.MicrosoftAppId,
                _graphLogger);

            var mediaSettings = new MediaPlatformSettings
            {
                MediaPlatformInstanceSettings = new MediaPlatformInstanceSettings
                {
                    CertificateThumbprint = _settings.MediaCertificateThumbprint,
                    InstanceInternalPort = _settings.MediaInternalPort,
                    InstancePublicPort = _settings.MediaInstanceExternalPort,
                    InstancePublicIPAddress = GetPublicOrAnyIp(),
                    ServiceFqdn = serviceFqdn
                },
                ApplicationId = _settings.MicrosoftAppId,
                MediaPlatformLogger = new CallMediaLogger(_logger)
            };

            builder.SetAuthenticationProvider(_auth);
            builder.SetNotificationUrl(callback);
            builder.SetMediaPlatformSettings(mediaSettings);
            builder.SetServiceBaseUrl(new Uri(AppConstants.PlaceCallEndpointUrl));

            _client = builder.Build();
            _logger.LogInformation("Graph Communications client initialized. Notification / callback URL: {Url}", _settings.CallbackUri);
        }
    }

    public async Task<ICall> JoinMeetingAsync(
        string? meetingId,
        string? passcode,
        string? threadId,
        string? messageId,
        CancellationToken cancellationToken = default)
    {
        var joinId = !string.IsNullOrWhiteSpace(meetingId) ? meetingId : _settings.DefaultMeetingId;
        var joinPass = !string.IsNullOrWhiteSpace(passcode) ? passcode : _settings.DefaultMeetingPasscode;
        if (string.IsNullOrWhiteSpace(joinId))
        {
            throw new InvalidOperationException("Meeting join id is empty. Set query meetingId= or DefaultMeetingId in configuration.");
        }

        var th = !string.IsNullOrWhiteSpace(threadId) ? threadId : _settings.DefaultChatThreadId;
        var msg = !string.IsNullOrWhiteSpace(messageId) ? messageId : _settings.DefaultChatMessageId ?? "0";

        if (!IsGraphClientReady)
        {
            throw new InvalidOperationException(
                "Graph client is not initialized. Set InitializeGraphCommunications to true, configure MediaCertificateThumbprint (cert in local machine store), CallbackUri, and ServiceFqdn, then restart.");
        }

        var hasThread = !string.IsNullOrWhiteSpace(th);
        _logger.LogInformation(
            "Join request started. JoinMeetingId set: {HasId}, Passcode set: {HasPass}, ChatInfo thread: {WithThread}",
            joinId,
            !string.IsNullOrEmpty(joinPass),
            hasThread);

        var mediaSession = CreateLocalMediaSession();
        var meetingInfo = new JoinMeetingIdMeetingInfo
        {
            Passcode = string.IsNullOrEmpty(joinPass) ? null : joinPass,
            JoinMeetingId = joinId
        };

        var chatInfo = hasThread
            ? new ChatInfo { ThreadId = th, MessageId = msg }
            : new ChatInfo();

        var joinParams = new JoinMeetingParameters(
            chatInfo: chatInfo,
            meetingInfo: meetingInfo,
            mediaSession: mediaSession)
        {
            TenantId = _settings.TenantId
        };

        var scenarioId = Guid.NewGuid();
        try
        {
            var call = await Client.Calls()
                .AddAsync(joinParams, scenarioId, cancellationToken)
                .ConfigureAwait(false);
            _logger.LogInformation("Join request sent (AddAsync completed). Call id: {CallId}", call.Id);
            // Incoming media: ILocalMediaSession.AudioSocket + ICall state for capture start/stop (see CallAudioCapture).
            var audioCapture = new CallAudioCapture(
                call,
                mediaSession,
                _logger,
                _settings.EnableFullCallWavCapture,
                _chunkMlSink);
            audioCapture.StartListening();
            return call;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "JoinMeeting failed: {Full}", ex);
            _graphLogger.Error(ex, "JoinMeeting failed");
            throw;
        }
    }

    private ILocalMediaSession CreateLocalMediaSession(Guid mediaSessionId = default)
    {
        // Pcm16K + receive-only: meeting audio to the socket for AudioSocket.AudioMediaReceived.
        return Client.CreateMediaSession(
            new AudioSocketSettings
            {
                StreamDirections = StreamDirection.Recvonly,
                SupportedAudioFormat = AudioFormat.Pcm16K,
                ReceiveUnmixedMeetingAudio = false
            },
            new VideoSocketSettings
            {
                StreamDirections = StreamDirection.Inactive
            },
            mediaSessionId: mediaSessionId);
    }

    private static IPAddress GetPublicOrAnyIp()
    {
        try
        {
            return Dns
                .GetHostEntry(Dns.GetHostName())
                .AddressList
                .FirstOrDefault(a => a.AddressFamily == AddressFamily.InterNetwork) ?? IPAddress.Any;
        }
        catch
        {
            return IPAddress.Any;
        }
    }

    public void Dispose()
    {
        _client?.Dispose();
        _client = null;
    }

    private sealed class CallMediaLogger(ILogger<CallService> log) : IMediaPlatformLogger
    {
        public void WriteLog(Microsoft.Skype.Bots.Media.LogLevel level, string logStatement) =>
            log.LogDebug("[Skype.Bots.Media] {Level} {Message}", level, logStatement);
    }
}
