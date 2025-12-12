// Simple gRPC test client for RedVector
// Note: This requires the generated proto code to be accessible
// For now, use grpcurl or Python client instead

fn main() {
    println!("gRPC Test Client");
    println!("================");
    println!();
    println!("To test gRPC API, use one of these methods:");
    println!();
    println!("1. Install grpcurl:");
    println!("   cargo install grpcurl");
    println!("   or download from: https://github.com/fullstorydev/grpcurl");
    println!();
    println!("2. Test with grpcurl:");
    println!("   grpcurl -plaintext localhost:50051 list");
    println!("   grpcurl -plaintext -d '{{\"collection_name\":\"test\",\"vector_size\":384,\"distance\":\"Cosine\"}}' \\");
    println!("     localhost:50051 redvector.VectorService/CreateCollection");
    println!();
    println!("3. Use Python client:");
    println!("   pip install grpcio grpcio-tools");
    println!("   # Then use the generated Python client");
    println!();
    println!("The gRPC server is running on port 50051");
}


