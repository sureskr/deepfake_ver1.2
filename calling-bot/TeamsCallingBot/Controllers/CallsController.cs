using System.Text;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Graph.Communications.Client;
using TeamsCallingBot.Services;
using TeamsCallingBot.Util;

namespace TeamsCallingBot.Controllers;

[ApiController]
[Route("api/calls")]
public class CallsController : ControllerBase
{
    private readonly ICallService _callService;
    private readonly ILogger<CallsController> _logger;

    public CallsController(ICallService callService, ILogger<CallsController> logger)
    {
        _callService = callService;
        _logger = logger;
    }

    [HttpPost]
    public async Task PostAsync(CancellationToken cancellationToken)
    {
        Request.EnableBuffering();
        string body;
        using (var reader = new StreamReader(Request.Body, Encoding.UTF8, leaveOpen: true))
        {
            body = await reader.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
        }
        _logger.LogInformation("POST /api/calls. Body: {Body}", body);
        Request.Body.Position = 0;

        if (!_callService.IsGraphClientReady || _callService.Client is null)
        {
            _logger.LogWarning("POST /api/calls: Graph client not initialized; returning 200 without ProcessNotificationAsync. Enable InitializeGraphCommunications and media cert for real callbacks.");
            Response.StatusCode = StatusCodes.Status200OK;
            return;
        }

        var httpRequest = HttpHelpers.ToHttpRequestMessage(Request);
        var response = await _callService.Client.ProcessNotificationAsync(httpRequest).ConfigureAwait(false);

        Response.StatusCode = (int)response.StatusCode;
        foreach (var h in response.Headers)
        {
            Response.Headers[h.Key] = h.Value.ToArray();
        }
        if (response.Content == null)
        {
            return;
        }
        foreach (var h in response.Content.Headers)
        {
            Response.Headers[h.Key] = h.Value.ToArray();
        }
        await response.Content.CopyToAsync(Response.Body, cancellationToken).ConfigureAwait(false);
    }
}
