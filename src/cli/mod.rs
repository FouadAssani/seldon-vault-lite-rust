//! Analyse des arguments et exécution des commandes de la CLI.
//!
//! La CLI est volontairement minimale au jalon M0 : `version` et `help`.
//! Les commandes `serve`, `embed`, `index`, `search` et `verify` seront
//! ajoutées avec leurs fonctionnalités aux jalons suivants. Aucun stub ne
//! simule un comportement inexistant.

use std::ffi::OsString;
use std::io::Write;

use crate::error::{Error, Result};
use crate::{PROGRAM_NAME, TARGET_TRIPLE, VERSION};

/// Texte d'aide affiché par la commande `help`.
const HELP: &str = concat!(
    env!("CARGO_PKG_NAME"),
    r#" — moteur local d'embeddings et de recherche multimodale

USAGE:
    "#,
    env!("CARGO_PKG_NAME"),
    r#" <COMMANDE>

COMMANDES:
    version    Affiche la version et la cible de compilation
    help       Affiche cette aide

OPTIONS:
    -V, --version    Équivalent à `version`
    -h, --help       Équivalent à `help`

CODES DE SORTIE:
    0    succès
    1    erreur d'exécution
    2    erreur d'usage

Les commandes `serve`, `embed`, `index`, `search` et `verify` seront ajoutées
avec les jalons M1 à M6. Aucune commande n'est simulée en attendant.
"#
);

/// Commande reconnue par la CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Affiche la version du programme et la cible de compilation.
    Version,
    /// Affiche l'aide et les codes de sortie.
    Help,
}

/// Analyse les arguments situés après le nom du programme.
///
/// Les arguments non UTF-8 sont rejetés avec une erreur d'usage plutôt que de
/// provoquer une panique : la CLI ne doit jamais s'interrompre sur une entrée
/// hostile.
pub fn parse<I, S>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Err(Error::usage(
            "aucune commande fournie; utilisez `--help` pour afficher l'aide",
        ));
    };
    let first: OsString = first.into();
    let first = first
        .to_str()
        .ok_or_else(|| Error::usage("argument non UTF-8"))?;

    let command = match first {
        "version" | "--version" | "-V" => Command::Version,
        "help" | "--help" | "-h" => Command::Help,
        other => {
            return Err(Error::usage(format!(
                "commande inconnue `{other}`; utilisez `--help`"
            )));
        }
    };

    if args.next().is_some() {
        return Err(Error::usage(format!("arguments superflus après `{first}`")));
    }

    Ok(command)
}

/// Exécute une commande et écrit sa sortie sur `stdout`.
///
/// # Erreurs
///
/// Retourne une erreur d'E/S si l'écriture échoue, notamment lorsque la
/// sortie standard est fermée par le processus appelant.
pub fn run<W: Write>(command: Command, stdout: &mut W) -> Result<()> {
    match command {
        Command::Version => writeln!(stdout, "{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})")?,
        Command::Help => stdout.write_all(HELP.as_bytes())?,
    }
    stdout.flush()?;
    Ok(())
}

/// Point d'entrée de la CLI à partir des arguments bruts du processus.
///
/// Le premier argument est le nom du programme, comme avec
/// [`std::env::args_os`] ; il est ignoré. Le reste est analysé par
/// [`parse`] puis exécuté par [`run`].
pub fn main_entry<I, S>(args: I, stdout: &mut impl Write) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter();
    let _program_name = args.next();
    let command = parse(args)?;
    run(command, stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    fn parse_args(args: &[&str]) -> Result<Command> {
        parse(args.iter().copied())
    }

    #[test]
    fn accepts_version_aliases() {
        for args in [&["version"][..], &["--version"], &["-V"]] {
            assert_eq!(
                parse_args(args).expect("version doit être acceptée"),
                Command::Version
            );
        }
    }

    #[test]
    fn accepts_help_aliases() {
        for args in [&["help"][..], &["--help"], &["-h"]] {
            assert_eq!(
                parse_args(args).expect("help doit être acceptée"),
                Command::Help
            );
        }
    }

    #[test]
    fn rejects_missing_command() {
        let error = parse_args(&[]).expect_err("l'absence de commande doit échouer");
        assert!(matches!(error, Error::Usage(_)));
        assert_eq!(error.exit_code(), 2);
        assert!(error.to_string().contains("aucune commande"));
    }

    #[test]
    fn rejects_unknown_command() {
        let error = parse_args(&["serve"]).expect_err("serve n'existe pas encore");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("commande inconnue"));
    }

    #[test]
    fn rejects_extra_arguments() {
        let error =
            parse_args(&["version", "extra"]).expect_err("les arguments superflus doivent échouer");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("arguments superflus"));
    }

    #[test]
    fn rejects_non_utf8_argument() {
        let args = [OsStr::from_bytes(b"\xff")];
        let error = parse(args.iter().copied()).expect_err("un argument non UTF-8 doit échouer");
        assert!(matches!(error, Error::Usage(_)));
        assert!(error.to_string().contains("non UTF-8"));
    }

    #[test]
    fn version_output_is_stable() {
        let mut output = Vec::new();
        run(Command::Version, &mut output).expect("l'écriture doit réussir");
        assert_eq!(
            String::from_utf8(output).expect("la sortie doit être de l'UTF-8"),
            format!("{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})\n")
        );
    }

    #[test]
    fn help_contains_usage_and_exit_codes() {
        let mut output = Vec::new();
        run(Command::Help, &mut output).expect("l'écriture doit réussir");
        let text = String::from_utf8(output).expect("la sortie doit être de l'UTF-8");
        assert!(text.contains("USAGE:"));
        assert!(text.contains("CODES DE SORTIE:"));
        assert!(text.contains("version"));
    }

    #[test]
    fn main_entry_ignores_program_name() {
        let mut output = Vec::new();
        main_entry(["seldon-vault-lite", "version"], &mut output)
            .expect("la commande doit réussir");
        assert_eq!(
            String::from_utf8(output).expect("la sortie doit être de l'UTF-8"),
            format!("{PROGRAM_NAME} {VERSION} ({TARGET_TRIPLE})\n")
        );
    }
}
