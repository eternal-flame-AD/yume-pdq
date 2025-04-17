fn main() {
    println!("cargo:rerun-if-changed=vendor/ThreatExchange");
    println!("cargo:rerun-if-changed=official_binding.cpp");
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR")).unwrap();
    let mut builder = cc::Build::new();

    builder.inherit_rustflags(true);

    builder.flag("-Ivendor/ThreatExchange");
    builder.flag("-Wall");
    builder.flag("-Wextra");
    builder.flag("-Wpedantic");
    builder.flag("-std=c++17");
    builder.flag("-O3");
    builder.flag("-ffast-math");
    builder.flag("-funroll-loops");
    builder.flag("-ftree-vectorize");
    builder.flag("-march=native");
    builder.flag("-mtune=native");
    builder.flag("-fomit-frame-pointer");

    builder.file("official_binding.cpp");

    [
        "vendor/ThreatExchange/pdq/cpp/common",
        "vendor/ThreatExchange/pdq/cpp/downscaling",
        "vendor/ThreatExchange/pdq/cpp/hashing",
    ]
    .into_iter()
    .flat_map(|path| {
        let listing = std::fs::read_dir(path).unwrap();
        listing.filter_map(|entry| {
            let entry = entry.unwrap();
            if entry.path().is_file() {
                let path = entry.path();
                if path.to_str().unwrap().ends_with(".cpp")
                    || path.to_str().unwrap().ends_with(".c")
                {
                    Some(path)
                } else {
                    None
                }
            } else {
                None
            }
        })
    })
    .for_each(|path| {
        builder.file(path);
    });

    builder.compile("yume_pdq_official_testing_adapter");
}
