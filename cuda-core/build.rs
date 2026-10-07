/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Probes the CUDA headers for the multicast driver API (CUDA 12.1+).
//!
//! The `cuMulticast*` entry points first appeared in CUDA 12.1, and
//! `cuda-bindings` binds whatever the host `cuda.h` declares, so building
//! against a CUDA 12.0 toolkit would otherwise fail to compile all of
//! `cuda-core`. The `cuda_has_multicast` cfg gates the multicast surface of
//! `vmm` to toolkits that declare the API, mirroring the
//! `cuda_has_cuEventElapsedTime_v2` probe in `cuda-bindings`.
//!
//! Toolkit discovery matches `cuda-bindings/build.rs`: the first set
//! variable among `CUDA_TOOLKIT_PATH` and `CUDA_HOME`, else the same
//! default candidate list with the same CUDA 13.0 floor, with both the
//! standard `include/` and the redistributable `targets/<dir>/include/`
//! layouts probed (a non-blank `CUDA_TOOLKIT_TARGET_DIR` names the single
//! `targets/` tree to probe). A missing or unreadable `cuda.h` leaves the
//! cfg unset (multicast unavailable) rather than erroring; `cuda-bindings`
//! reports the authoritative failure for a genuinely broken toolkit.

use std::env;
use std::path::{Path, PathBuf};

const TOOLKIT_ENV_VARS: &[&str] = &["CUDA_TOOLKIT_PATH", "CUDA_HOME"];

/// Overrides the `targets/<dir>` selection with a single directory name,
/// like nvcc's `-target-dir`; matches `cuda-bindings`.
const TOOLKIT_TARGET_DIR_ENV: &str = "CUDA_TOOLKIT_TARGET_DIR";

/// The version floor, matching `cuda-bindings/build.rs` (`MIN_CUDA_VERSION`).
/// Candidates come from [`installed_cuda_toolkits`], a copy of
/// `cuda-bindings/toolkit_installs.rs`: a versioned-only install (no
/// `/usr/local/cuda` symlink), or a symlink pointing at a below-floor tree,
/// must resolve to the same toolkit here as it does there, or this probe reads
/// a different `cuda.h` than the one the bindings were generated from.
const MIN_CUDA_VERSION: u32 = 13000;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_has_multicast)");
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-env-changed={TOOLKIT_TARGET_DIR_ENV}");

    let Some(cuda_h) = find_cuda_header() else {
        return;
    };
    println!("cargo:rerun-if-changed={}", cuda_h.display());
    if std::fs::read_to_string(&cuda_h).is_ok_and(|header| header.contains("cuMulticastCreate")) {
        println!("cargo:rustc-cfg=cuda_has_multicast");
    }
}

/// CUDA toolkit `targets/` directory names to probe for cargo's build
/// target, most specific first.
///
/// Kept in lockstep BY HAND with the selection table in
/// `cuda-bindings/toolkit_target.rs` (`resolve_toolkit_include_candidates`):
/// build scripts cannot import each other's sources across crates. If the
/// selection there changes, mirror it here. CUDA names these layouts after
/// the GPU platform, not the Rust triple, and an aarch64 Linux triple is
/// ambiguous between servers (`sbsa-linux`) and Tegra (`aarch64-linux`), so
/// both are probed in that order.
fn toolkit_target_dirs() -> Vec<String> {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "linux" {
        return vec![];
    }
    match arch.as_str() {
        "x86_64" => vec!["x86_64-linux".to_string()],
        "aarch64" => vec!["sbsa-linux".to_string(), "aarch64-linux".to_string()],
        _ => vec![],
    }
}

/// The include directories to probe for `cuda.h` under one toolkit root. A
/// non-blank [`TOOLKIT_TARGET_DIR_ENV`] names the single `targets/` tree to
/// probe (the top-level `include/` is not consulted, so a cross-build
/// cannot silently bind the host's headers); otherwise the standard
/// `include/` comes first, then the table's `targets/<dir>/include`
/// candidates. Matches `cuda-bindings`.
fn include_candidates(toolkit: &Path) -> Vec<PathBuf> {
    if let Some(dir) = env::var(TOOLKIT_TARGET_DIR_ENV)
        .ok()
        .filter(|dir| !dir.trim().is_empty())
    {
        return vec![toolkit.join("targets").join(dir).join("include")];
    }
    let mut candidates = vec![toolkit.join("include")];
    for target_dir in toolkit_target_dirs() {
        candidates.push(toolkit.join("targets").join(target_dir).join("include"));
    }
    candidates
}

/// The first include candidate under `toolkit` that contains `cuda.h`.
fn find_cuda_header_in(toolkit: &Path) -> Option<PathBuf> {
    include_candidates(toolkit)
        .into_iter()
        .map(|dir| dir.join("cuda.h"))
        .find(|header| header.is_file())
}

/// The `CUDA_VERSION` a `cuda.h` declares, when readable.
fn cuda_version_from_header(cuda_h: &Path) -> Option<u32> {
    let source = std::fs::read_to_string(cuda_h).ok()?;
    source.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next(), parts.next()) {
            (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
            _ => None,
        }
    })
}

/// Returns the path of `cuda.h` for the toolkit `cuda-bindings` resolves:
/// the first set variable among [`TOOLKIT_ENV_VARS`] taken as-is, else the
/// first default candidate whose `cuda.h` meets the version floor.
fn find_cuda_header() -> Option<PathBuf> {
    for var in TOOLKIT_ENV_VARS {
        if let Ok(toolkit) = env::var(var) {
            return find_cuda_header_in(Path::new(&toolkit));
        }
    }
    installed_cuda_toolkits().iter().find_map(|toolkit| {
        let header = find_cuda_header_in(toolkit)?;
        (cuda_version_from_header(&header)? >= MIN_CUDA_VERSION).then_some(header)
    })
}

// Kept in lockstep BY HAND with `cuda-bindings/toolkit_installs.rs`.
/// Every versioned CUDA toolkit installed in the platform's standard location
/// (`CUDA\vX.Y` under Program Files on Windows, `/usr/local/cuda-X.Y`
/// elsewhere), newest version first, then the unversioned Linux links
/// (`/usr/local/cuda-13`, `/usr/local/cuda`). Callers apply their own
/// version floor. Scanning, rather than naming versions, keeps a newly
/// installed toolkit from being skipped in favour of whichever older one a
/// fixed list happened to name.
fn installed_cuda_toolkits() -> Vec<PathBuf> {
    #[cfg(windows)]
    let mut toolkits = versioned_cuda_toolkits_in(
        Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
        "v",
    );
    #[cfg(not(windows))]
    let mut toolkits = versioned_cuda_toolkits_in(Path::new("/usr/local"), "cuda-");
    #[cfg(not(windows))]
    toolkits.extend(["/usr/local/cuda-13", "/usr/local/cuda"].map(PathBuf::from));
    toolkits
}

/// The entries of `parent` named `{prefix}{major}.{minor}`, newest version
/// first, compared as numbers (13.10 is newer than 13.4).
fn versioned_cuda_toolkits_in(parent: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut versioned: Vec<((u32, u32), PathBuf)> = std::fs::read_dir(parent)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let (major, minor) = name.strip_prefix(prefix)?.split_once('.')?;
            Some(((major.parse().ok()?, minor.parse().ok()?), entry.path()))
        })
        .collect();
    versioned.sort_by(|a, b| b.0.cmp(&a.0));
    versioned.into_iter().map(|(_, path)| path).collect()
}
