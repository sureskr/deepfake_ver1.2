using System.Net.Http.Headers;
using System.Threading.Channels;
using Microsoft.Extensions.Options;
using TeamsCallingBot.Models;

namespace TeamsCallingBot.Services;

/// <summary>
/// Queues chunk file paths and POSTs WAV data to the configured ML predict endpoint.
/// Failures are logged; the bot process is not interrupted.
/// </summary>
public sealed class MlInferenceService : BackgroundService, IChunkMlSink
{
    private readonly ILogger<MlInferenceService> _logger;
    private readonly MlSettings _settings;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly Channel<(string Path, int ChunkIndex)> _channel = Channel.CreateUnbounded<(string, int)>();

    public MlInferenceService(
        IOptions<MlSettings> options,
        IHttpClientFactory httpClientFactory,
        ILogger<MlInferenceService> logger)
    {
        _settings = options.Value;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
    }

    public override Task StartAsync(CancellationToken cancellationToken)
    {
        var predictAbsolute = MlUrlBuilder.CombinePredictUrl(_settings.BaseUrl, _settings.PredictEndpoint);
        _logger.LogInformation("ML Endpoint: {Url}, Enabled: {Enabled}", predictAbsolute, _settings.Enabled);
        return base.StartAsync(cancellationToken);
    }

    public void OnChunkWavWritten(string absolutePath, int chunkIndex)
    {
        if (!_settings.Enabled)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(_settings.BaseUrl))
        {
            _logger.LogWarning("ML is enabled but BaseUrl is empty; skipping chunkIndex={ChunkIndex}", chunkIndex);
            return;
        }

        if (!File.Exists(absolutePath))
        {
            _logger.LogWarning("ML skip: chunk file missing path={Path} chunkIndex={ChunkIndex}", absolutePath, chunkIndex);
            return;
        }

        if (!_channel.Writer.TryWrite((absolutePath, chunkIndex)))
        {
            _logger.LogWarning("ML queue rejected chunk path={Path} chunkIndex={ChunkIndex}", absolutePath, chunkIndex);
        }
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await foreach (var item in _channel.Reader.ReadAllAsync(stoppingToken).ConfigureAwait(false))
        {
            try
            {
                await ProcessOneAsync(item.Path, item.ChunkIndex, stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "ML inference unexpected error path={Path} chunkIndex={ChunkIndex}", item.Path, item.ChunkIndex);
            }
        }
    }

    private async Task ProcessOneAsync(string path, int chunkIndex, CancellationToken stoppingToken)
    {
        if (!_settings.Enabled)
        {
            return;
        }

        var url = MlUrlBuilder.CombinePredictUrl(_settings.BaseUrl, _settings.PredictEndpoint);
        if (string.IsNullOrWhiteSpace(url))
        {
            _logger.LogWarning("ML predict URL empty; skipping chunkIndex={ChunkIndex}", chunkIndex);
            return;
        }

        var timeoutSec = Math.Clamp(_settings.TimeoutSeconds, 1, 3600);
        using var http = _httpClientFactory.CreateClient("Ml");
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
        cts.CancelAfter(TimeSpan.FromSeconds(timeoutSec));

        try
        {
            using var content = new MultipartFormDataContent();
            await using (var fs = File.OpenRead(path))
            {
                var streamContent = new StreamContent(fs);
                streamContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                content.Add(streamContent, "file", Path.GetFileName(path));

                using var response = await http.PostAsync(url, content, cts.Token).ConfigureAwait(false);
                var body = await response.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation(
                        "ML result chunkIndex={ChunkIndex} path={Path} status={Status} body={Body}",
                        chunkIndex,
                        path,
                        (int)response.StatusCode,
                        body);
                }
                else
                {
                    _logger.LogWarning(
                        "ML non-success chunkIndex={ChunkIndex} path={Path} status={Status} body={Body}",
                        chunkIndex,
                        path,
                        (int)response.StatusCode,
                        body);
                }
            }
        }
        catch (OperationCanceledException ex) when (!stoppingToken.IsCancellationRequested)
        {
            _logger.LogWarning(ex, "ML call timed out ({TimeoutSeconds}s) chunkIndex={ChunkIndex} path={Path}", timeoutSec, chunkIndex, path);
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(ex, "ML service unreachable chunkIndex={ChunkIndex} path={Path}", chunkIndex, path);
        }
        catch (IOException ex)
        {
            _logger.LogWarning(ex, "ML could not read chunk file chunkIndex={ChunkIndex} path={Path}", chunkIndex, path);
        }
    }
}
