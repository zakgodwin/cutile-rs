// Installed CUDA toolkit discovery, shared verbatim with
// `cuda-core/build.rs` and `cutile-compiler/src/cuda_tile_runtime_utils.rs`
// (build scripts cannot import each other's sources across crates, so those
// two carry hand-synced copies of `installed_cuda_toolkits`).

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
