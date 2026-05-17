namespace TeamsCallingBot.Models;

/// <summary>
/// Bound from configuration section <c>ML</c>. Override with environment variables
/// <c>ML__BaseUrl</c>, <c>ML__Enabled</c>, etc. (ASP.NET Core convention).
/// </summary>
public sealed class MlSettings
{
    public string BaseUrl { get; set; } = string.Empty;
    public string PredictEndpoint { get; set; } = "/predict";

    /// <summary>HTTP client timeout for each predict request.</summary>
    public int TimeoutSeconds { get; set; } = 10;

    public bool Enabled { get; set; } = true;
}
