mod features { pub mod branding; pub mod commands; }

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "geneloop", version, about = "GeneLoop CLI", disable_help_subcommand = false)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Quick checks for an input (e.g., BAM) and suggestions
    Debug { path: String },
    /// Generate a project/slice from a natural language spec
    Gen { spec: String },
    /// Explain a file/script in plain language
    Explain { path: String },
    /// Refactor a file to a new target and/or library
    Refactor {
        path: String,
        #[arg(long)] target: Option<String>,
        #[arg(long)] library: Option<String>,
    },
}

fn main() -> Result<()> {
    // Banner
    features::branding::ui::banner::show_startup(std::io::stdout())?;

    let cli = Cli::parse();
    match cli.command {
        Some(Command::Debug { path }) => features::commands::debug(&path)?,
        Some(Command::Gen { spec }) => features::commands::gen(&spec)?,
        Some(Command::Explain { path }) => features::commands::explain(&path)?,
        Some(Command::Refactor { path, target, library }) => features::commands::refactor(&path, target.as_deref(), library.as_deref())?,
        None => {
            // No subcommand: show brief help
            eprintln!("\nUsage: geneloop <COMMAND> [ARGS]\n  Commands: debug | gen | explain | refactor\n  Try: geneloop --help\n");
        }
    }
    Ok(())
}

