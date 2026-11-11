//! Binaire `seldon-vault-lite`.
//!
//! `main` se contente de traduire le résultat de la CLI en code de sortie.
//! Toute la logique d'analyse et d'exécution vit dans la bibliothèque afin
//! d'être testable sans lancer de processus.

use std::io::Write;
use std::process::ExitCode;

use seldon_vault_lite::cli;

fn main() -> ExitCode {
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();

    match cli::main_entry(std::env::args_os(), &mut stdout) {
        Ok(()) => ExitCode::SUCCESS,
        // Un tuyau fermé est un comportement normal d'outil Unix, pas une panne.
        Err(error) if error.is_broken_pipe() => ExitCode::SUCCESS,
        Err(error) => {
            // L'écriture du diagnostic est best-effort : si stderr est lui aussi
            // fermé, le code de sortie reste la seule information disponible.
            let _ = writeln!(stderr, "erreur: {error}");
            let _ = stderr.flush();
            ExitCode::from(error.exit_code())
        }
    }
}
