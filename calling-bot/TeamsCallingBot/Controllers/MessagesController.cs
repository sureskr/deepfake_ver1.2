using System.Text;
using Microsoft.AspNetCore.Mvc;

namespace TeamsCallingBot.Controllers;

[ApiController]
[Route("api/messages")]
public class MessagesController : ControllerBase
{
    private readonly ILogger<MessagesController> _logger;

    public MessagesController(ILogger<MessagesController> logger) => _logger = logger;

    [HttpPost]
    public async Task<IActionResult> PostAsync(CancellationToken cancellationToken)
    {
        Request.EnableBuffering();
        string body;
        using (var reader = new StreamReader(Request.Body, Encoding.UTF8, leaveOpen: true))
        {
            body = await reader.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
        }
        _logger.LogInformation("POST /api/messages. Body: {Body}", body);
        return Ok();
    }
}
