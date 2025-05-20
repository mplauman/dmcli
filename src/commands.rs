use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[clap(multicall(true), disable_help_flag(true), name(""))]
pub struct DmCli {
    #[clap(subcommand)]
    pub command: DmCommand,
}

#[derive(Subcommand, Debug)]
pub enum DmCommand {
    Exit {},
    Reset {},
    Roll { expressions: Vec<String> },
}
