use anthropic::ClientBuilder;
use config::Config;

use crate::anthropic::Client;
use crate::commands::DmCommand;
use crate::errors::Error;
use crate::events::AppEvent;
use crate::input::InputHandler;

mod anthropic;
mod commands;
mod errors;
mod events;
mod input;
mod logger;
mod markdown;
mod obsidian;
#[cfg(test)]
mod test_integration;
mod tui;

fn load_settings() -> Result<Config, Error> {
    use config::{Environment, File};

    let mut config_file = dirs::config_dir().expect("config dir should exist");
    config_file.push("dmcli.toml");

    log::info!("Loading configuration from {}", config_file.display());

    Config::builder()
        .add_source(File::from(config_file))
        .add_source(Environment::with_prefix("DMCLI"))
        .build()
        .map_err(|e| e.into())
}

fn init_logging(settings: &Config) -> Result<(), Error> {
    let mut aggregate_log_builder = crate::logger::AggregateLoggerBuilder::default();

    if let Ok(true) = settings.get_bool("logging.opentelemetry") {
        use opentelemetry_appender_log::OpenTelemetryLogBridge;
        use opentelemetry_otlp::LogExporter;
        use opentelemetry_sdk::logs::{BatchLogProcessor, SdkLoggerProvider};

        let log_exporter = LogExporter::builder().with_http().build()?;
        let log_processor = BatchLogProcessor::builder(log_exporter).build();

        let logger_provider = SdkLoggerProvider::builder()
            .with_log_processor(log_processor)
            .build();

        let otel_log_appender = OpenTelemetryLogBridge::new(&logger_provider);
        aggregate_log_builder = aggregate_log_builder.with(otel_log_appender);
    }

    if let Ok(true) = settings.get_bool("logging.syslog") {
        let formatter = syslog::Formatter3164::default();
        let logger = syslog::unix(formatter)?;

        let logger = syslog::BasicLogger::new(logger);

        aggregate_log_builder = aggregate_log_builder.with(logger);
    }

    log::set_boxed_logger(Box::new(aggregate_log_builder.build()))
        .expect("A logger should not have already been set");

    match settings
        .get_string("logging.level")
        .as_ref()
        .map(|s| s.as_str())
    {
        Ok("info") => log::set_max_level(log::LevelFilter::Info),
        Ok("warn") => log::set_max_level(log::LevelFilter::Warn),
        Ok("error") => log::set_max_level(log::LevelFilter::Error),
        Ok("debug") => log::set_max_level(log::LevelFilter::Debug),
        Ok("trace") => log::set_max_level(log::LevelFilter::Trace),
        _ => log::set_max_level(log::LevelFilter::Info),
    }

    Ok(())
}

async fn create_client(
    config: &Config,
    event_sender: async_channel::Sender<AppEvent>,
) -> Result<Client, Error> {
    let mut builder = ClientBuilder::default()
        .with_api_key(
            config
                .get_string("anthropic.api_key")
                .expect("api_key must be set"),
        )
        .with_event_sender(event_sender);

    if let Ok(model) = config.get_string("anthropic.model") {
        log::info!("Overriding anthropic model to {model}");
        builder = builder.with_model(model);
    }

    if let Ok(max_tokens) = config.get_int("anthropic.max_tokens") {
        log::info!("Overriding anthropic max tokens to {max_tokens}");
        builder = builder.with_max_tokens(max_tokens)
    }

    if let Ok(obsidian_vault) = config.get_string("local.obsidian_vault") {
        log::info!(
            "Adding tools for obsidian vault located at {obsidian_vault}"
        );

        let obsidian = crate::obsidian::Obsidian::new(obsidian_vault.into());

        builder = builder.with_toolkit(obsidian).await;
    };

    builder.build().await
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let settings = load_settings()?;
    init_logging(&settings)?;

    let (event_sender, event_receiver) = async_channel::unbounded::<AppEvent>();
    let mut input_handler = InputHandler::new(event_sender.clone())?;
    let mut client = create_client(&settings, event_sender.clone()).await?;
    let mut tui = crate::tui::Tui::new(&settings, event_sender.clone())?;

    tokio::spawn(async move {
        loop {
            input_handler.read_input().await;
            log::debug!("Got input, reading more");
        }
    });

    tui.render()?;
    while let Ok(event) = event_receiver.recv().await {
        log::debug!("Got event, updating");

        match event {
            AppEvent::UserCommand(DmCommand::Exit {}) => {
                tui.add_message("Good bye!".to_string(), crate::tui::MessageType::System);
                break;
            }
            AppEvent::UserCommand(DmCommand::Reset {}) => {
                client.clear();
                tui.add_message(
                    "Conversation reset".to_string(),
                    crate::tui::MessageType::System,
                );
            }
            AppEvent::UserCommand(DmCommand::Roll { expressions }) => {
                let result = caith::Roller::new(&expressions.join(" "))
                    .unwrap()
                    .roll()
                    .unwrap();
                tui.add_message(format!("🎲 {result}"), crate::tui::MessageType::System);
            }
            AppEvent::UserAgent(line) => {
                if !line.is_empty() {
                    tui.add_message(line.clone(), crate::tui::MessageType::User);
                    client.push(line)?;
                }
            }
            AppEvent::Exit => {
                log::debug!("Exit event received, exiting");
                break;
            }
            AppEvent::AiResponse(msg) => tui.add_message(msg, crate::tui::MessageType::Assistant),
            AppEvent::AiThinking(msg, tools) => {
                tui.add_message(format!("🤔 {msg}"), crate::tui::MessageType::Thinking);
                client.use_tools(tools).await?;
            }
            AppEvent::AiCompact(attempt, max_attempts) => {
                tui.add_message(
                    format!(
                        "Max tokens reached. Removing oldest message and retrying. Attempt {attempt} of {max_attempts}"
                    ),
                    crate::tui::MessageType::System
                );
                client.compact(attempt, max_attempts)?;
            }
            AppEvent::AiError(msg) => {
                tui.add_message(format!("❌ {msg}"), crate::tui::MessageType::Error)
            }
            AppEvent::CommandResult(msg) => tui.add_message(msg, crate::tui::MessageType::System),
            AppEvent::CommandError(msg) => {
                tui.add_message(format!("Error: {msg}"), crate::tui::MessageType::Error)
            }
            AppEvent::InputUpdated { line, cursor } => tui.input_updated(line, cursor),
            AppEvent::WindowResized { width, height } => tui.resized(width, height),
            AppEvent::ScrollBack => tui.handle_scroll_back(),
            AppEvent::ScrollForward => tui.handle_scroll_forward(),
        }

        tui.render()?
    }

    log::info!("Exiting cleanly");
    Ok(())
}
