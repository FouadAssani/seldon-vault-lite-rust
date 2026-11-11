//! Type d'erreur central de Seldon Vault Lite.
//!
//! Toutes les erreurs du produit convergent ici afin que la CLI, puis l'API
//! HTTP, puissent choisir un code de sortie ou un statut sans inspecter des
//! erreurs ad hoc. Les variantes d'entrée corrompue et de limite dépassée
//! seront ajoutées avec les parseurs du jalon M1.

use std::error::Error as StdError;
use std::fmt;
use std::io;

/// Résultat interne du crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Erreur centrale du produit.
///
/// Les variantes distinguent une erreur d'usage (code de sortie 2) d'une
/// erreur d'exécution (code de sortie 1). La taxonomie sera étendue avec les
/// erreurs d'entrée corrompue et de limite dépassée lorsque les parseurs
/// correspondants existeront.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Commande ou argument invalide fourni par l'utilisateur.
    Usage(String),
    /// Échec d'entrée/sortie.
    Io(io::Error),
}

impl Error {
    /// Construit une erreur d'usage.
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    /// Code de sortie du processus associé à cette erreur.
    ///
    /// `0` est réservé au succès, `1` aux erreurs d'exécution et `2` aux
    /// erreurs d'usage.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) => 2,
            Self::Io(_) => 1,
        }
    }

    /// Indique si l'erreur provient d'un tuyau fermé par le processus appelant.
    ///
    /// Un tuyau fermé n'est pas une panne du programme : la CLI peut alors
    /// terminer avec le code de succès, comme le font les outils Unix.
    pub fn is_broken_pipe(&self) -> bool {
        matches!(self, Self::Io(error) if error.kind() == io::ErrorKind::BrokenPipe)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) => formatter.write_str(message),
            Self::Io(error) => write!(formatter, "{error}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Usage(_) => None,
            Self::Io(error) => Some(error),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_errors_exit_with_code_two() {
        let error = Error::usage("commande invalide");
        assert!(matches!(error, Error::Usage(_)));
        assert_eq!(error.exit_code(), 2);
        assert!(!error.is_broken_pipe());
        assert_eq!(error.to_string(), "commande invalide");
        assert!(StdError::source(&error).is_none());
    }

    #[test]
    fn io_errors_exit_with_code_one() {
        let error = Error::from(io::Error::other("panne"));
        assert_eq!(error.exit_code(), 1);
        assert!(!error.is_broken_pipe());
        assert_eq!(
            StdError::source(&error)
                .expect("une erreur d'E/S doit exposer sa source")
                .to_string(),
            "panne"
        );
    }

    #[test]
    fn broken_pipe_is_detected() {
        let error = Error::from(io::Error::from(io::ErrorKind::BrokenPipe));
        assert!(error.is_broken_pipe());
    }
}
