namespace TeamsCallingBot.Services;

/// <summary>
/// Receives notification after a chunk WAV is written (async queue + ML call happens in <see cref="MlInferenceService"/>).
/// </summary>
public interface IChunkMlSink
{
    void OnChunkWavWritten(string absolutePath, int chunkIndex);
}
