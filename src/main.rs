use anthropic::ClientBuilder;
use clap::Parser;
use config::Config;
use shlex::split;

use crate::anthropic::Client;
use crate::commands::{DmCli, DmCommand};
use crate::errors::Error;

mod anthropic;
mod commands;
mod errors;
mod obsidian;
mod tools;

fn read_line(readline: &mut rustyline::DefaultEditor) -> Result<Option<String>, Error> {
    match readline.readline(">> ") {
        Ok(line) => Ok(Some(line)),
        Err(rustyline::error::ReadlineError::Interrupted) => Ok(None),
        Err(x) => Err(x.into()),
    }
}

fn parse_command(line: &str) -> Option<DmCommand> {
    match split(line).map(DmCli::try_parse_from) {
        Some(Ok(DmCli { command })) => Some(command),
        _ => None,
    }
}

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

async fn create_client(config: &Config) -> Result<Client, Error> {
    let mut builder = ClientBuilder::default().with_api_key(
        config
            .get_string("anthropic.api_key")
            .expect("api_key must be set"),
    );

    if let Ok(model) = config.get_string("anthropic.model") {
        log::info!("Overriding anthropic model to {}", model);
        builder = builder.with_model(model);
    }

    if let Ok(obsidian_vault) = config.get_string("local.obsidian_vault") {
        log::info!(
            "Adding tools for obsidian vault located at {}",
            obsidian_vault
        );
        let obsidian = crate::obsidian::LocalVaultBuilder::default()
            .with_directory(obsidian_vault.into())
            .build();

        builder = builder.with_toolkit(obsidian);
    };

    builder.build().await
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut rl = rustyline::DefaultEditor::new().unwrap();

    let settings = load_settings()?;
    let client = create_client(&settings).await?;

    let mut ai_chat = Vec::<anthropic::Message>::new();

    loop {
        let Some(line) = read_line(&mut rl)? else {
            log::info!("No line read, exiting");
            break;
        };

        if line.is_empty() {
            continue;
        }

        rl.add_history_entry(line.as_str())?;

        // Check to see if the entered text can be handled with one of the built in commands. This
        // is much faster, cheaper, and more accurate than feeding it to an AI agent for processing.
        if let Some(command) = parse_command(line.as_str()) {
            log::info!("Line appears to be a paseable command: {}", line);

            match command {
                DmCommand::Exit {} => {
                    println!("Good bye!");
                    break;
                }
                DmCommand::Reset {} => {
                    ai_chat.clear();
                }
                DmCommand::Roll { expressions } => {
                    println!("Rolling {:?}", expressions.join(" "));
                }
            }

            continue;
        }

        log::info!("Sending line to AI agent");
        ai_chat.push(anthropic::Message::user(line));
        client.request(&mut ai_chat).await?;
    }

    log::info!("Exiting cleanly");
    Ok(())
}
