using Microsoft.AspNetCore.Mvc;
using TeamsCallingBot.Services;

namespace TeamsCallingBot.Controllers;

[ApiController]
[Route("test")]
public class TestController : ControllerBase
{
    private readonly ICallService _callService;
    private readonly ILogger<TestController> _logger;

    public TestController(ICallService callService, ILogger<TestController> logger)
    {
        _callService = callService;
        _logger = logger;
    }

    [HttpGet("join-meeting")]
    public async Task<IActionResult> JoinMeetingAsync(
        [FromQuery] string? meetingId,
        [FromQuery] string? passcode,
        [FromQuery] string? threadId,
        [FromQuery] string? messageId,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("GET /test/join-meeting");
        try
        {
            var call = await _callService
                .JoinMeetingAsync(meetingId, passcode, threadId, messageId, cancellationToken)
                .ConfigureAwait(false);
            return Ok(new
            {
                call.Id,
                call.ScenarioId
            });
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "Join validation / configuration: {Message}", ex.Message);
            return BadRequest(new { error = ex.Message });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Exception: {Exception}", ex);
            return StatusCode(500, new { error = ex.Message, detail = ex.ToString() });
        }
    }
}
