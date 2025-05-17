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
        builder = builder.with_model(model);
    }

    builder.build().await
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut rl = rustyline::DefaultEditor::new().unwrap();

    let settings = load_settings()?;
    let client = create_client(&settings).await?;

    loop {
        let Some(line) = read_line(&mut rl)? else {
            break;
        };

        if line.is_empty() {
            continue;
        }

        rl.add_history_entry(line.as_str())?;

        // Check to see if the entered text can be handled with one of the built in commands. This
        // is much faster, cheaper, and more accurate than feeding it to an AI agent for processing.
        if let Some(command) = parse_command(line.as_str()) {
            match command {
                DmCommand::Exit {} => {
                    println!("Good bye!");
                    break;
                }
                DmCommand::Roll { expressions } => {
                    println!("Rolling {:?}", expressions.join(" "));
                }
            }

            continue;
        }

        // Anything that isn't specifically a command just gets fed to the AI agent. This can take
        // a lot longer to deal with (and is more expensive) but is where the magic like "create an
        // NPC with blue hair" type stuff works.
        //
        // (until the agent actually works, just spit out the line)
        println!(">>> {}", line);

        let response = client
            .request(&serde_json::json!([
                {
                    "role": "user",
                    "content": line,
                }
            ]))
            .await?;

        let body = response.text().await?;
        println!("Body: {:?}", body);
    }

    Ok(())
}
