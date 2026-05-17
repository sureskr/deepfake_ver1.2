using System.IdentityModel.Tokens.Jwt;
using System.Net.Http.Headers;
using System.Security.Claims;
using Microsoft.Graph.Communications.Client;
using Microsoft.Graph.Communications.Client.Authentication;
using Microsoft.Graph.Communications.Common;
using Microsoft.Graph.Communications.Common.Telemetry;
using Microsoft.Identity.Client;
using Microsoft.IdentityModel.Protocols;
using Microsoft.IdentityModel.Protocols.OpenIdConnect;
using Microsoft.IdentityModel.Tokens;

namespace TeamsCallingBot.Authentication;

public class BotAuthenticationProvider : ObjectRoot, IRequestAuthenticationProvider
{
    private readonly string _appId;
    private readonly string _appSecret;
    private readonly TimeSpan _openIdConfigRefreshInterval = TimeSpan.FromHours(2);
    private DateTime _prevOpenIdConfigUpdateTimestamp = DateTime.MinValue;
    private OpenIdConnectConfiguration? _openIdConfiguration;

    public BotAuthenticationProvider(string appName, string appId, string appSecret, IGraphLogger graphLogger)
        : base(graphLogger.NotNull(nameof(graphLogger)).CreateShim(nameof(BotAuthenticationProvider)))
    {
        _ = appName.NotNullOrWhitespace(nameof(appName));
        _appId = appId.NotNullOrWhitespace(nameof(appId));
        _appSecret = appSecret.NotNullOrWhitespace(nameof(appSecret));
    }

    public async Task AuthenticateOutboundRequestAsync(HttpRequestMessage request, string? tenant)
    {
        const string replaceString = "{tenant}";
        const string oauthV2TokenLink = "https://login.microsoftonline.com/{tenant}";
        const string resource = "https://graph.microsoft.com";
        tenant = string.IsNullOrWhiteSpace(tenant) ? "common" : tenant;
        var tokenLink = oauthV2TokenLink.Replace(replaceString, tenant);
        var scopes = new[] { $"{resource}/.default" };

        GraphLogger.Info("AuthenticationProvider: Generating OAuth token.");
        var app = ConfidentialClientApplicationBuilder.Create(_appId)
            .WithAuthority(new Uri(tokenLink))
            .WithClientSecret(_appSecret)
            .Build();

        AuthenticationResult result;
        try
        {
            result = await app.AcquireTokenForClient(scopes)
                .ExecuteAsync()
                .ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            GraphLogger.Error(ex, $"Failed to generate token for client: {_appId}");
            throw;
        }

        GraphLogger.Info(
            $"AuthenticationProvider: Generated OAuth token. Expires in {result.ExpiresOn.Subtract(DateTimeOffset.UtcNow).TotalMinutes:F1} minutes.");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", result.AccessToken);
    }

    public async Task<RequestValidationResult> ValidateInboundRequestAsync(HttpRequestMessage request)
    {
        var token = request?.Headers?.Authorization?.Parameter;
        if (string.IsNullOrWhiteSpace(token))
        {
            return new RequestValidationResult { IsValid = false };
        }

        if (_openIdConfiguration == null
            || DateTime.UtcNow > _prevOpenIdConfigUpdateTimestamp.Add(_openIdConfigRefreshInterval))
        {
            GraphLogger.Info("Updating OpenID configuration");
            IConfigurationManager<OpenIdConnectConfiguration> configurationManager =
                new ConfigurationManager<OpenIdConnectConfiguration>(
                    AppConstants.AuthDomain,
                    new OpenIdConnectConfigurationRetriever());
            _openIdConfiguration = await configurationManager
                .GetConfigurationAsync(CancellationToken.None)
                .ConfigureAwait(false);
            _prevOpenIdConfigUpdateTimestamp = DateTime.UtcNow;
        }

        var authIssuers = new[]
        {
            "https://graph.microsoft.com",
            "https://api.botframework.com"
        };
        var validationParameters = new TokenValidationParameters
        {
            ValidIssuers = authIssuers,
            ValidAudience = _appId,
            IssuerSigningKeys = _openIdConfiguration?.SigningKeys
        };

        try
        {
            ArgumentNullException.ThrowIfNull(request);
            var handler = new JwtSecurityTokenHandler();
            var claimsPrincipal = handler.ValidateToken(token, validationParameters, out _);
            const string claimType = "http://schemas.microsoft.com/identity/claims/tenantid";
            var tenantClaim = claimsPrincipal.FindFirst(claim => claim.Type.Equals(claimType, StringComparison.Ordinal));
            if (string.IsNullOrEmpty(tenantClaim?.Value))
            {
                return new RequestValidationResult { IsValid = false };
            }
#pragma warning disable CS0618 // HttpRequestMessage.Properties; Graph comms still reads tenant from this bag
            request.Properties[HttpConstants.HeaderNames.Tenant] = tenantClaim.Value;
#pragma warning restore CS0618
            return new RequestValidationResult { IsValid = true, TenantId = tenantClaim.Value };
        }
        catch (Exception ex)
        {
            GraphLogger.Error(ex, $"Failed to validate token for client: {_appId}.");
            return new RequestValidationResult { IsValid = false };
        }
    }
}
