use anthropic::ClientBuilder;
use config::Config;
use std::sync::Arc;

use crate::anthropic::Client;
use crate::commands::DmCommand;
use crate::conversation::Conversation;
use crate::embeddings::{EmbeddingGenerator, Model2VecEmbeddingGeneratorBuilder};
use crate::errors::Error;
use crate::events::AppEvent;
use crate::input::InputHandler;

mod anthropic;
mod commands;
mod conversation;
mod embeddings;
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

    #[cfg(unix)]
    {
        if let Ok(true) = settings.get_bool("logging.syslog") {
            let formatter = syslog::Formatter3164::default();
            let logger = syslog::unix(formatter)?;

            let logger = syslog::BasicLogger::new(logger);

            aggregate_log_builder = aggregate_log_builder.with(logger);
        }
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
        builder = builder.with_max_tokens(max_tokens);
    }

    if let Ok(obsidian_vault) = config.get_string("local.obsidian_vault") {
        log::info!("Adding tools for obsidian vault located at {obsidian_vault}");

        let obsidian = crate::obsidian::Obsidian::new(obsidian_vault.into());

        builder = builder.with_toolkit(obsidian).await?;
    };

    builder.build().await
}

fn create_embedder(config: &Config) -> Result<Arc<dyn EmbeddingGenerator>, Error> {
    let mut builder = Model2VecEmbeddingGeneratorBuilder::default();

    if let Ok(repo) = config.get_string("embedder.repo") {
        log::info!("Overriding embedder model to {repo}");
        builder = builder.repo(repo);
    }

    if let Ok(token) = config.get_string("embedder.token") {
        log::info!("Applying huggingface token");
        builder = builder.token(token)
    }

    if let Ok(subfolder) = config.get_string("embedder.subfolder") {
        log::info!("Using embedder from subfolder {subfolder}");
        builder = builder.subfolder(subfolder);
    }

    let result: Arc<dyn EmbeddingGenerator> = builder.build().map(Arc::new)?;

    Ok(result)
}

fn create_conversation(
    _config: &Config,
    embedder: Arc<dyn EmbeddingGenerator>,
) -> Result<Conversation, Error> {
    Conversation::builder().with_embedder(embedder).build()
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let settings = load_settings()?;
    init_logging(&settings)?;

    let embedder = create_embedder(&settings)?;
    let mut conversation = create_conversation(&settings, embedder)?;
    let mut input_text = String::new();
    let mut input_cursor = usize::default();

    let (event_sender, event_receiver) = async_channel::unbounded::<AppEvent>();
    let mut client = create_client(&settings, event_sender.clone()).await?;

    // Do as much as possible before these: they set the terminal into raw mode
    let mut input_handler = InputHandler::new(event_sender.clone())?;
    let mut tui = crate::tui::Tui::new(&settings, event_sender.clone())?;

    tokio::spawn(async move {
        loop {
            input_handler.read_input().await;
            log::debug!("Got input, reading more");
        }
    });

    conversation.system("Welcome to dmcli! Type your message and press Enter to send. Send 'roll 2d6' to roll a dice or 'exit' to quit.");

    tui.render(&conversation, &input_text, input_cursor)?;
    while let Ok(event) = event_receiver.recv().await {
        log::debug!("Got event, updating");

        match event {
            AppEvent::UserCommand(DmCommand::Exit {}) => {
                conversation.system("Good bye!");
                break;
            }
            AppEvent::UserCommand(DmCommand::Reset {}) => {
                conversation.system("Conversation reset (not really)");
            }
            AppEvent::UserCommand(DmCommand::Roll { expressions }) => {
                let result = caith::Roller::new(&expressions.join(" "))
                    .unwrap()
                    .roll()
                    .unwrap();
                conversation.system(format!("🎲 {result}"));
                tui.reset_scroll();
            }
            AppEvent::UserAgent(line) => {
                if !line.is_empty() {
                    conversation.user(&line);
                    tui.reset_scroll();
                    client.push(&conversation)?;
                }
            }
            AppEvent::Exit => {
                log::debug!("Exit event received, exiting");
                break;
            }
            AppEvent::AiResponse(msg) => {
                conversation.assistant(&msg);
                tui.reset_scroll();
            }
            AppEvent::AiThinking(msg, tools) => {
                conversation.thinking(format!("🤔 {msg}"), tools);
            }
            AppEvent::AiThinkingDone(tools) => {
                conversation.thinking_done(tools);
            }
            AppEvent::AiError(msg) => {
                conversation.error(format!("❌ {msg}"));
                tui.reset_scroll();
            }
            AppEvent::InputUpdated { line, cursor } => {
                input_text = line.clone();
                input_cursor = cursor;
            }
            AppEvent::WindowResized { width, height } => tui.resized(width, height),
            AppEvent::ScrollBack => tui.handle_scroll_back(),
            AppEvent::ScrollForward => tui.handle_scroll_forward(),
        }

        tui.render(&conversation, &input_text, input_cursor)?
    }

    log::info!("Exiting cleanly");
    Ok(())
}
