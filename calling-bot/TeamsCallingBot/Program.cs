using Microsoft.Graph.Communications.Client.Authentication;
using Microsoft.Graph.Communications.Common.Telemetry;
using Microsoft.Extensions.Hosting;
using TeamsCallingBot.Authentication;
using TeamsCallingBot.Models;
using TeamsCallingBot.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<AppSettings>(builder.Configuration);
builder.Services.Configure<MlSettings>(builder.Configuration.GetSection("ML"));
builder.Services.AddHttpClient("Ml", client => client.Timeout = Timeout.InfiniteTimeSpan);
builder.Services.AddSingleton<MlInferenceService>();
builder.Services.AddSingleton<IChunkMlSink>(sp => sp.GetRequiredService<MlInferenceService>());
builder.Services.AddHostedService(sp => sp.GetRequiredService<MlInferenceService>());
builder.Services.AddSingleton<IGraphLogger>(_ => new GraphLogger("TeamsCallingBot", redirectToTrace: true));
builder.Services.AddSingleton<IRequestAuthenticationProvider>(sp =>
{
    var app = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<AppSettings>>().Value;
    return new BotAuthenticationProvider("TeamsCallingBot", app.MicrosoftAppId, app.MicrosoftAppPassword, sp.GetRequiredService<IGraphLogger>());
});
builder.Services.AddSingleton<ICallService, CallService>();
builder.Services.AddControllers();
builder.Services.AddLogging();

var app = builder.Build();

app.Services.GetRequiredService<IHostApplicationLifetime>()
    .ApplicationStarted
    .Register(() =>
    {
        app.Logger.LogInformation("Teams calling bot app started. Urls: {Urls}", string.Join(", ", app.Urls));
        Console.WriteLine("Teams calling bot app started. Urls: {0}", string.Join(", ", app.Urls));
    });

using (var scope = app.Services.CreateScope())
{
    scope.ServiceProvider.GetRequiredService<ICallService>().Initialize();
}

app.MapControllers();
await app.RunAsync().ConfigureAwait(false);
