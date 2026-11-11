//! Fondations de Seldon Vault Lite.
//!
//! Le crate expose la CLI et le type d'erreur central. Les modules
//! d'inférence, de stockage et de médias seront ajoutés au fil des jalons
//! M1 à M9 ; aucun module vide n'est créé à l'avance.
//!
//! Contrat du projet : aucune crate tierce, uniquement `std` et les
//! intrinsics de `core::arch`.

#![warn(missing_docs)]

pub mod cli;
pub mod error;

/// Version du crate, issue de `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Nom du programme affiché par la CLI.
///
/// Il est dérivé du nom du paquet afin que l'aide et la ligne de version ne
/// puissent pas diverger si le paquet est renommé.
pub const PROGRAM_NAME: &str = env!("CARGO_PKG_NAME");

/// Triple de la cible de compilation.
///
/// La phase 1 ne valide que `aarch64-apple-darwin`. Les autres cibles
/// renvoient `"unsupported"` tant que leur portage n'a pas été validé ; cela
/// évite d'annoncer une compatibilité qui n'a pas été testée.
pub const TARGET_TRIPLE: &str = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
    "aarch64-apple-darwin"
} else {
    "unsupported"
};
