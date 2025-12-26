use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::str::from_utf8;
use std::{env, path};

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let path = path::Path::new(&out_dir).join("release.rs");

    let mut f = File::create(path).unwrap();

    {
        let hash = match Command::new("git")
            .arg("show-ref")
            .arg("--head")
            .arg("--hash=8")
            .output()
        {
            Ok(o) => {
                if o.stdout.len() >= 8 {
                    String::from(from_utf8(&o.stdout[0..8]).unwrap())
                } else {
                    String::from("00000000")
                }
            }
            Err(_) => String::from("00000000"),
        };
        writeln!(f, "pub const GIT_SHA1: &str = \"{}\";", &hash[0..8]).unwrap();
    }

    {
        let dirty = match Command::new("git")
            .arg("diff")
            .arg("--no-ext-diff")
            .output()
        {
            Ok(o) => !o.stdout.is_empty(),
            Err(_) => true,
        };
        writeln!(
            f,
            "pub const GIT_DIRTY: bool = {};",
            if dirty { "true" } else { "false " }
        )
        .unwrap();
    }

    {
        let version = match Command::new("rustc").arg("--version").output() {
            Ok(o) => String::from(from_utf8(&o.stdout).unwrap().trim()),
            Err(_) => String::new(),
        };
        writeln!(f, "pub const RUSTC_VERSION: &str = {:?};", version).unwrap();
    }

    // Compile protobuf for gRPC when api-server feature is enabled
    #[cfg(feature = "api-server")]
    {
        let proto_path = "api-server/proto/vector.proto";
        if std::path::Path::new(proto_path).exists() {
            // Create output directory for proto descriptor
            let proto_out = path::Path::new(&out_dir).join("proto");
            std::fs::create_dir_all(&proto_out).ok();
            
            tonic_build::configure()
                .file_descriptor_set_path(proto_out.join("vector_descriptor.bin"))
                .compile(&[proto_path], &["api-server/proto"])
                .expect("Failed to compile protobuf");
        }
    }
}
