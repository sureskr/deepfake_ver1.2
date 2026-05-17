namespace TeamsCallingBot.Services;

internal static class MlUrlBuilder
{
    /// <summary>Full predict URL: <paramref name="baseUrl"/> + <paramref name="predictEndpoint"/>.</summary>
    public static string CombinePredictUrl(string baseUrl, string predictEndpoint)
    {
        if (string.IsNullOrWhiteSpace(baseUrl))
        {
            return string.Empty;
        }

        var trimmedBase = baseUrl.TrimEnd('/');
        var path = string.IsNullOrEmpty(predictEndpoint)
            ? string.Empty
            : predictEndpoint.StartsWith('/')
                ? predictEndpoint
                : "/" + predictEndpoint;

        return trimmedBase + path;
    }
}
