pub mod release;
#[cfg(feature = "api-server")]
pub mod api;

use std::env::args;
use std::process::exit;

use crate::release::*;
use compat::getpid;
use config::Config;
use logger::{Level, Logger};
use networking::Server;

#[cfg(not(feature = "api-server"))]
fn main() {
    run_redis_only();
}

#[cfg(feature = "api-server")]
#[tokio::main]
async fn main() {
    run_with_api().await;
}

/// Run Redis server only (no API server)
#[allow(dead_code)]
fn run_redis_only() {
    let mut config = Config::new(Logger::new(Level::Notice));
    if let Some(f) = args().nth(1) {
        if config.parsefile(f).is_err() {
            exit(1);
        }
    }

    let (port, daemonize) = (config.port, config.daemonize);
    let mut server = Server::new(config);
    {
        let mut db = server.get_mut_db();
        db.git_sha1 = GIT_SHA1;
        db.git_dirty = GIT_DIRTY;
        db.version = env!("CARGO_PKG_VERSION");
        db.rustc_version = RUSTC_VERSION;
    }

    if !daemonize {
        println!();
        println!("🚀 RedVector v{} - Redis-Compatible Vector Database", env!("CARGO_PKG_VERSION"));
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("🔴 Redis Protocol: localhost:{}", port);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("PID: {}", getpid());
        println!("Features: Redis Protocol, Vector Search (HNSW)");
        println!();
    }
    server.run();
}

/// Run Redis server with integrated REST and gRPC API servers
#[cfg(feature = "api-server")]
async fn run_with_api() {
    use std::sync::{Arc, Mutex};
    
    let mut config = Config::new(Logger::new(Level::Notice));
    if let Some(f) = args().nth(1) {
        if config.parsefile(f).is_err() {
            exit(1);
        }
    }

    let (port, daemonize) = (config.port, config.daemonize);
    
    // Create the server and get a reference to the database
    let mut server = Server::new(config);
    {
        let mut db = server.get_mut_db();
        db.git_sha1 = GIT_SHA1;
        db.git_dirty = GIT_DIRTY;
        db.version = env!("CARGO_PKG_VERSION");
        db.rustc_version = RUSTC_VERSION;
    }

    // Get database reference for API server
    // Note: We need to share the database between Redis and API servers
    // The current architecture uses Arc<Mutex<Database>> internally
    
    if !daemonize {
        println!();
        println!("🚀 RedVector v{} - Redis-Compatible Vector Database", env!("CARGO_PKG_VERSION"));
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("🔴 Redis Protocol: localhost:{}", port);
        println!("📊 REST API:       http://localhost:8888");
        println!("🔌 gRPC API:       http://localhost:50051");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("PID: {}", getpid());
        println!("Features: Redis Protocol, Vector Search (HNSW), REST API, gRPC API");
        println!();
    }
    
    // Create a new database reference for API servers
    // This creates a separate database instance for the API
    // In the future, we should share the same database
    let api_db = Arc::new(Mutex::new(database::Database::new(
        Config::new(Logger::new(Level::Notice))
    )));
    
    // Start API servers in background
    let api_config = api::ApiConfig::default();
    if let Err(e) = api::start_api_servers(api_db, api_config).await {
        eprintln!("Failed to start API servers: {}", e);
    }
    
    // Run Redis server in a blocking thread
    tokio::task::spawn_blocking(move || {
        server.run();
    }).await.expect("Redis server failed");
}
