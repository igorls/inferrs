mod buffer;
mod cache;
mod cli;
mod config;
mod engine;
mod model;
mod sampling;
mod scheduler;
mod server;
mod tokenizer;

use clap::Parser;
use cli::Cli;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    if let Err(e) = cli.run() {
        tracing::error!("{e:#}");
        std::process::exit(1);
    }
}
