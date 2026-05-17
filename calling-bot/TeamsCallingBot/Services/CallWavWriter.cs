using System.Runtime.InteropServices;
using System.Text;

namespace TeamsCallingBot.Services;

/// <summary>PCM 16 kHz, 16-bit, mono: reserve header, stream PCM, fix RIFF sizes in <see cref="Close"/>.</summary>
internal sealed class CallWavWriter : IDisposable
{
    private const int SampleRate = 16_000;
    private const short NumChannels = 1;
    private const short BitsPerSample = 16;
    private const int HeaderSize = 44;
    private readonly FileStream _file;
    private long _dataBytes;
    private bool _closed;

    public CallWavWriter(string filePath)
    {
        _file = new FileStream(
            filePath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.Read,
            bufferSize: 64 * 1024,
            useAsync: false);
        // RIFF header placeholder; final sizes written in Close().
        _file.Write(Encoding.ASCII.GetBytes("RIFF"), 0, 4);
        _file.Write(new byte[4], 0, 4);
        _file.Write(Encoding.ASCII.GetBytes("WAVE"), 0, 4);
        _file.Write(Encoding.ASCII.GetBytes("fmt "), 0, 4);
        WriteLe32(16);
        WriteLe16(1);
        WriteLe16(NumChannels);
        WriteLe32(SampleRate);
        var byteRate = SampleRate * NumChannels * BitsPerSample / 8;
        WriteLe32(byteRate);
        WriteLe16((short)(NumChannels * BitsPerSample / 8));
        WriteLe16(BitsPerSample);
        _file.Write(Encoding.ASCII.GetBytes("data"), 0, 4);
        _file.Write(new byte[4], 0, 4);
        if (HeaderSize != _file.Position)
        {
            _file.Close();
            throw new InvalidOperationException($"WAV header length mismatch: expected {HeaderSize}, was {_file.Position}.");
        }
    }

    /// <summary>Append raw PCM frames (caller ensures 16 kHz mono 16-bit).</summary>
    public void WritePcm(ReadOnlySpan<byte> pcm)
    {
        if (pcm.IsEmpty) return;
        _dataBytes += pcm.Length;
        _file.Write(pcm);
    }

    public void WritePcmIntPtr(nint data, long length)
    {
        if (length == 0 || data == 0) return;
        if (length > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "Frame larger than 2GB.");
        var len = (int)length;
        var buf = new byte[len];
        Marshal.Copy(data, buf, 0, len);
        WritePcm(buf);
    }

    private void WriteLe16(short v) => _file.Write(BitConverter.GetBytes(v));
    private void WriteLe32(int v) => _file.Write(BitConverter.GetBytes(v));

    public void Close()
    {
        if (_closed) return;
        _closed = true;
        // RIFF chunk size = 36 + data size (8-byte RIFF + WAVE + fmt + data subchunk headers)
        var riffChunkSize = 36 + (int)_dataBytes;
        _file.Position = 4;
        _file.Write(BitConverter.GetBytes(riffChunkSize), 0, 4);
        _file.Position = 40;
        _file.Write(BitConverter.GetBytes((int)_dataBytes), 0, 4);
        _file.Close();
    }

    public void Dispose() => Close();
}
