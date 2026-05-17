using Microsoft.Graph.Communications.Calls;
using Microsoft.Graph.Communications.Calls.Media;
using Microsoft.Graph.Communications.Resources;
using Microsoft.Graph.Models;
using Microsoft.Skype.Bots.Media;

namespace TeamsCallingBot.Services;

/// <summary>
/// Incoming <see cref="IAudioSocket.AudioMediaReceived"/> (16 kHz, 16-bit, mono PCM) is split into 20-second
/// standalone WAV chunks under <see cref="ChunksRoot"/>; optionally also one full-call WAV under
/// <see cref="CaptureRoot"/> when <paramref name="enableFullCallWavCapture"/> is true.
/// Starts/stops on <see cref="ICall"/> establish/terminate.
/// </summary>
internal sealed class CallAudioCapture : IDisposable
{
    private const string CaptureRoot = @"C:\TeamsCallingBot\captures";
    private const string ChunksRoot = @"C:\TeamsCallingBot\captures\chunks";
    /// <summary>20 s × 16000 Hz × 2 bytes = 640,000 bytes per chunk WAV.</summary>
    private const int ChunkPcmBytes = 16000 * 2 * 20;

    private readonly ICall _call;
    private readonly ILocalMediaSession _localMedia;
    private readonly ILogger _logger;
    private readonly IChunkMlSink? _chunkMlSink;
    private readonly bool _enableFullCallWavCapture;
    private readonly string _callIdSafe;
    /// <summary>Optional single-file recording: <see cref="CaptureRoot"/>.
    /// <c>captures\call_*.wav</c> (debug).</summary>
    private readonly string _fullCallWavPath;
    private readonly object _lock = new();
    private readonly ResourceEventHandler<ICall, Call> _onCallUpdated;
    private readonly EventHandler<AudioMediaReceivedEventArgs> _onAudio;
    /// <summary>Legacy full-call writer; only used when <see cref="_enableFullCallWavCapture"/>.</summary>
    private CallWavWriter? _fullCallWriter;
    private readonly byte[] _chunkBuffer = new byte[ChunkPcmBytes];
    private int _chunkFill;
    /// <summary>Next 1-based index for <c>call_*_chunk_NNN.wav</c>.</summary>
    private int _nextChunkIndex = 1;
    private long _totalBytes;
    private DateTime _lastProgressUtc;
    private bool _captureActive;
    private bool _ended;
    private bool _firstFrameLogged;

    public CallAudioCapture(
        ICall call,
        ILocalMediaSession localMedia,
        ILogger logger,
        bool enableFullCallWavCapture = false,
        IChunkMlSink? chunkMlSink = null)
    {
        _call = call;
        _localMedia = localMedia;
        _logger = logger;
        _chunkMlSink = chunkMlSink;
        _enableFullCallWavCapture = enableFullCallWavCapture;
        _callIdSafe = string.IsNullOrEmpty(call.Id) ? Guid.NewGuid().ToString("N") : SanitizeFileSegment(call.Id);
        _fullCallWavPath = Path.Combine(CaptureRoot, $"call_{_callIdSafe}.wav");
        _onCallUpdated = OnCallUpdated;
        _onAudio = OnAudioFrame;
    }

    public void StartListening()
    {
        _call.OnUpdated += _onCallUpdated;
        _localMedia.AudioSocket.AudioMediaReceived += _onAudio;
    }

    private static string SanitizeFileSegment(string s)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = s.Select(c => invalid.Contains(c) ? '_' : c).ToArray();
        return new string(chars);
    }

    private void OnCallUpdated(object? sender, ResourceEventArgs<Call> e)
    {
        if (e.NewResource?.State is not { } state)
        {
            return;
        }

        if (state == CallState.Established)
        {
            lock (_lock)
            {
                if (_ended) return;
                if (_captureActive) return;
                Directory.CreateDirectory(CaptureRoot);
                Directory.CreateDirectory(ChunksRoot);
                if (_enableFullCallWavCapture)
                {
                    _logger.LogDebug("Full-call WAV (debug) enabled: {Path}", _fullCallWavPath);
                    _fullCallWriter = new CallWavWriter(_fullCallWavPath);
                }
                _chunkFill = 0;
                _nextChunkIndex = 1;
                _captureActive = true;
                _lastProgressUtc = DateTime.UtcNow;
                _logger.LogInformation(
                    "Call established; chunk capture: {ChunkDir} (chunk size {ChunkBytes} bytes PCM).",
                    ChunksRoot,
                    ChunkPcmBytes);
            }
        }
        else if (IsCallTerminal(state))
        {
            StopCapture($"call state: {state}");
        }
    }

    private static bool IsCallTerminal(CallState state) => state == CallState.Terminated;

    private void OnAudioFrame(object? sender, AudioMediaReceivedEventArgs e)
    {
        // Media SDK: dispose AudioMediaBuffer when done to return it to the pool.
        using var buffer = e.Buffer;
        if (buffer is null) return;
        if (buffer.Length == 0) return;
        if (buffer.Data == IntPtr.Zero) return;

        if (!_captureActive) return;
        var len = buffer.Length;
        if (len > int.MaxValue) return;
        var ptr = buffer.Data;
        var frame = new byte[(int)len];
        System.Runtime.InteropServices.Marshal.Copy(ptr, frame, 0, (int)len);

        if (!_firstFrameLogged)
        {
            _firstFrameLogged = true;
            _logger.LogInformation("Audio frame received (len={Length} bytes).", len);
        }

        var now = DateTime.UtcNow;
        lock (_lock)
        {
            if (!_captureActive) return;
            AppendPcmIntoChunks(frame);

            // Optional/debug: same PCM appended to one continuous WAV.
            _fullCallWriter?.WritePcm(frame);

            _totalBytes += len;
            if ((now - _lastProgressUtc).TotalSeconds >= 5.0)
            {
                _lastProgressUtc = now;
                _logger.LogInformation("Audio received (cumulative): {Kb:F0} KB", _totalBytes / 1024.0);
            }
        }
    }

    /// <summary>Fills rolling 20-second buffer and writes standalone WAV files as each fills.</summary>
    private void AppendPcmIntoChunks(ReadOnlySpan<byte> pcm)
    {
        var offset = 0;
        while (offset < pcm.Length)
        {
            var room = ChunkPcmBytes - _chunkFill;
            var take = Math.Min(room, pcm.Length - offset);
            pcm.Slice(offset, take).CopyTo(_chunkBuffer.AsSpan(_chunkFill));
            _chunkFill += take;
            offset += take;
            if (_chunkFill == ChunkPcmBytes)
            {
                WriteChunkWav(closeAfterFull: true);
                _chunkFill = 0;
            }
        }
    }

    private void WriteChunkWav(bool closeAfterFull)
    {
        var n = _nextChunkIndex++;
        var fileName = $"call_{_callIdSafe}_chunk_{n:000}.wav";
        var path = Path.Combine(ChunksRoot, fileName);
        var pcmSpan = closeAfterFull ? _chunkBuffer.AsSpan(0, ChunkPcmBytes) : _chunkBuffer.AsSpan(0, _chunkFill);
        using var w = new CallWavWriter(path);
        w.WritePcm(pcmSpan);
        w.Close();
        var sizeKb = pcmSpan.Length / 1024.0;
        _logger.LogInformation("Chunk saved: chunkIndex={ChunkIndex}, size={SizeKb}, path={Path}", n, sizeKb, path);
        _chunkMlSink?.OnChunkWavWritten(path, n);
    }

    private void FlushPartialChunkIfAny()
    {
        if (_chunkFill <= 0) return;
        WriteChunkWav(closeAfterFull: false);
        _chunkFill = 0;
    }

    private void StopCapture(string reason)
    {
        CallWavWriter? fullDispose = null;
        lock (_lock)
        {
            if (_ended) return;
            _ended = true;
            _captureActive = false;
            FlushPartialChunkIfAny();
            fullDispose = _fullCallWriter;
            _fullCallWriter = null;
        }

        if (_enableFullCallWavCapture)
        {
            _logger.LogInformation("Stopping full-call WAV ({Reason}) if any: {Path}", reason, _fullCallWavPath);
        }
        fullDispose?.Dispose();

        if (_totalBytes > 0)
        {
            _logger.LogInformation("Audio capture stopped ({Reason}); cumulative PCM: {Kb:F0} KB", reason, _totalBytes / 1024.0);
        }

        _call.OnUpdated -= _onCallUpdated;
        _localMedia.AudioSocket.AudioMediaReceived -= _onAudio;
    }

    public void Dispose() => StopCapture("dispose");
}

