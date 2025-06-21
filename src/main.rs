use anthropic::ClientBuilder;
use config::Config;
use std::sync::mpsc;

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
mod obsidian;
#[cfg(test)]
mod test_integration;

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
    event_sender: mpsc::Sender<AppEvent>,
) -> Result<Client, Error> {
    let mut builder = ClientBuilder::default()
        .with_api_key(
            config
                .get_string("anthropic.api_key")
                .expect("api_key must be set"),
        )
        .with_event_sender(event_sender);

    if let Ok(model) = config.get_string("anthropic.model") {
        log::info!("Overriding anthropic model to {}", model);
        builder = builder.with_model(model);
    }

    if let Ok(max_tokens) = config.get_int("anthropic.max_tokens") {
        log::info!("Overriding anthropic max tokens to {}", max_tokens);
        builder = builder.with_max_tokens(max_tokens)
    }

    if let Ok(obsidian_vault) = config.get_string("local.obsidian_vault") {
        log::info!(
            "Adding tools for obsidian vault located at {}",
            obsidian_vault
        );

        let obsidian = crate::obsidian::Obsidian::new(obsidian_vault.into());

        builder = builder.with_toolkit(obsidian).await;
    };

    builder.build().await
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut input_handler = InputHandler::new()?;

    let settings = load_settings()?;
    init_logging(&settings)?;

    let (event_sender, event_receiver) = mpsc::channel::<AppEvent>();
    let mut client = create_client(&settings, event_sender).await?;

    loop {
        let event = input_handler.read_input()?;

        match event {
            AppEvent::UserCommand(DmCommand::Exit {}) => {
                println!("Good bye!");
                break;
            }
            AppEvent::UserCommand(DmCommand::Reset {}) => {
                client.clear();
            }
            AppEvent::UserCommand(DmCommand::Roll { expressions }) => {
                let result = caith::Roller::new(&expressions.join(" "))
                    .unwrap()
                    .roll()
                    .unwrap();
                println!("{}", result);
            }
            AppEvent::UserAgent(line) => {
                log::info!("Sending line to AI agent");

                // Start the AI request in a separate task
                println!("Sending line to AI agent");
                client.push(line).await?;
                println!("Done");

                while let Ok(event) = event_receiver.try_recv() {
                    match event {
                        AppEvent::AiResponse(msg) => println!("{}", msg),
                        AppEvent::AiThinking(msg) => println!(":thinking: {}", msg),
                        AppEvent::AiError(msg) => {
                            println!(":error: {}", msg);
                            break;
                        }
                        AppEvent::AiComplete => {
                            break;
                        }
                        AppEvent::CommandResult(msg) => println!("{}", msg),
                        AppEvent::CommandError(msg) => println!("Error: {}", msg),
                        _ => {} // Ignore other event types
                    }
                }
            }
            AppEvent::Exit => {
                log::info!("Exit event received, exiting");
                break;
            }
            _ => {
                log::warn!("Unhandled event: {:?}", event);
            }
        }
    }

    log::info!("Exiting cleanly");
    Ok(())
}
