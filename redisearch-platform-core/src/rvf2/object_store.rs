//! Object Store: Abstraction over S3/GCS/Azure Blob/MinIO
//!
//! Provides tiered storage with local cache + object storage backing.

use async_trait::async_trait;
use bytes::Bytes;
use std::path::PathBuf;

use crate::rvf2::{Result, Rvf2Error};

/// Object store trait for remote segment storage
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get entire object
    async fn get(&self, key: &str) -> Result<Bytes>;

    /// Get byte range from object (for efficient partial reads)
    async fn get_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes>;

    /// Put object
    async fn put(&self, key: &str, data: Bytes) -> Result<()>;

    /// Delete object
    async fn delete(&self, key: &str) -> Result<()>;

    /// List objects with prefix
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Check if object exists
    async fn exists(&self, key: &str) -> Result<bool>;
}

/// Local filesystem "object store" (for development/testing)
pub struct LocalFsStore {
    base_path: PathBuf,
}

impl LocalFsStore {
    /// Create new local store
    pub fn new<P: Into<PathBuf>>(base_path: P) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    fn full_path(&self, key: &str) -> PathBuf {
        self.base_path.join(key)
    }
}

#[async_trait]
impl ObjectStore for LocalFsStore {
    async fn get(&self, key: &str) -> Result<Bytes> {
        let path = self.full_path(key);
        let data = tokio::fs::read(&path).await?;
        Ok(Bytes::from(data))
    }

    async fn get_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let path = self.full_path(key);
        let mut file = tokio::fs::File::open(&path).await?;
        file.seek(std::io::SeekFrom::Start(offset)).await?;

        let mut buffer = vec![0u8; len as usize];
        file.read_exact(&mut buffer).await?;

        Ok(Bytes::from(buffer))
    }

    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let path = self.full_path(key);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, &data).await?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let path = self.full_path(key);
        tokio::fs::remove_file(&path).await?;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let path = self.full_path(prefix);
        let mut entries = Vec::new();

        let mut read_dir = tokio::fs::read_dir(&path).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                entries.push(format!("{}/{}", prefix, name));
            }
        }

        Ok(entries)
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let path = self.full_path(key);
        Ok(tokio::fs::metadata(&path).await.is_ok())
    }
}

/// S3-compatible object store (AWS S3, MinIO, etc.)
/// 
/// Requires `aws-sdk-s3` dependency (behind feature flag)
#[cfg(feature = "s3")]
pub struct S3ObjectStore {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
}

#[cfg(feature = "s3")]
impl S3ObjectStore {
    /// Create new S3 store
    pub async fn new(bucket: &str, prefix: &str, region: &str) -> Result<Self> {
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_config::Region::new(region.to_string()))
            .load()
            .await;

        let client = aws_sdk_s3::Client::new(&config);

        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
        })
    }

    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }
}

#[cfg(feature = "s3")]
#[async_trait]
impl ObjectStore for S3ObjectStore {
    async fn get(&self, key: &str) -> Result<Bytes> {
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .send()
            .await
            .map_err(|e| Rvf2Error::ObjectStore(e.to_string()))?;

        let data = resp
            .body
            .collect()
            .await
            .map_err(|e| Rvf2Error::ObjectStore(e.to_string()))?
            .into_bytes();

        Ok(data)
    }

    async fn get_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes> {
        let range = format!("bytes={}-{}", offset, offset + len - 1);

        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .range(range)
            .send()
            .await
            .map_err(|e| Rvf2Error::ObjectStore(e.to_string()))?;

        let data = resp
            .body
            .collect()
            .await
            .map_err(|e| Rvf2Error::ObjectStore(e.to_string()))?
            .into_bytes();

        Ok(data)
    }

    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .body(data.into())
            .send()
            .await
            .map_err(|e| Rvf2Error::ObjectStore(e.to_string()))?;

        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .send()
            .await
            .map_err(|e| Rvf2Error::ObjectStore(e.to_string()))?;

        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);

        let resp = self
            .client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(&full_prefix)
            .send()
            .await
            .map_err(|e| Rvf2Error::ObjectStore(e.to_string()))?;

        let keys = resp
            .contents()
            .iter()
            .filter_map(|obj| obj.key().map(|k| k.to_string()))
            .collect();

        Ok(keys)
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let result = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(self.full_key(key))
            .send()
            .await;

        Ok(result.is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_local_fs_store() {
        let dir = tempdir().unwrap();
        let store = LocalFsStore::new(dir.path());

        // Put
        let data = Bytes::from("hello world");
        store.put("test/file.txt", data.clone()).await.unwrap();

        // Exists
        assert!(store.exists("test/file.txt").await.unwrap());
        assert!(!store.exists("nonexistent").await.unwrap());

        // Get
        let retrieved = store.get("test/file.txt").await.unwrap();
        assert_eq!(retrieved, data);

        // Get range
        let range = store.get_range("test/file.txt", 0, 5).await.unwrap();
        assert_eq!(&range[..], b"hello");

        // Delete
        store.delete("test/file.txt").await.unwrap();
        assert!(!store.exists("test/file.txt").await.unwrap());
    }
}

