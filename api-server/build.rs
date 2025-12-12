fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let descriptor_dir = std::path::PathBuf::from(&out_dir).join("proto");
    std::fs::create_dir_all(&descriptor_dir)?;
    
    let descriptor_path = descriptor_dir.join("vector_descriptor.bin");
    
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .file_descriptor_set_path(&descriptor_path)
        .compile(
            &["proto/vector.proto"],
            &["proto"],
        )?;
    Ok(())
}

