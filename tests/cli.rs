//! Tests d'intégration de la CLI du jalon M0.
//!
//! Ils exécutent le vrai binaire afin de vérifier le contrat observable :
//! sortie standard, sortie d'erreur et codes de sortie.

use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_seldon-vault-lite");

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("le binaire de test doit pouvoir être lancé")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout doit être de l'UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr doit être de l'UTF-8")
}

fn expected_version() -> String {
    format!(
        "seldon-vault-lite {} ({})\n",
        env!("CARGO_PKG_VERSION"),
        seldon_vault_lite::TARGET_TRIPLE
    )
}

#[test]
fn version_prints_name_version_and_target() {
    let output = run(&["version"]);
    assert!(output.status.success());
    assert_eq!(stdout(&output), expected_version());
    assert!(output.stderr.is_empty());
}

#[test]
fn version_aliases_are_equivalent() {
    for args in [&["version"][..], &["--version"], &["-V"]] {
        let output = run(args);
        assert!(output.status.success(), "échec pour {args:?}");
        assert_eq!(
            stdout(&output),
            expected_version(),
            "sortie différente pour {args:?}"
        );
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn help_explains_usage_and_exit_codes() {
    for args in [&["help"][..], &["--help"], &["-h"]] {
        let output = run(args);
        assert!(output.status.success(), "échec pour {args:?}");
        let text = stdout(&output);
        assert!(text.contains("USAGE:"), "aide sans USAGE pour {args:?}");
        assert!(
            text.contains("CODES DE SORTIE:"),
            "aide sans codes pour {args:?}"
        );
        assert!(text.contains("version"));
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn missing_command_is_a_usage_error() {
    let output = run(&[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let text = stderr(&output);
    assert!(text.contains("erreur:"));
    assert!(text.contains("aucune commande"));
}

#[test]
fn unknown_command_is_a_usage_error() {
    // `serve` est planifiée mais n'est pas encore implémentée : la CLI doit
    // refuser explicitement au lieu de simuler un service.
    let output = run(&["serve"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("commande inconnue"));
}

#[test]
fn extra_arguments_are_rejected() {
    let output = run(&["version", "extra"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(stderr(&output).contains("arguments superflus"));
}
