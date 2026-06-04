use fastbloom::BloomFilter;

use crate::error::OperationError;

const DEFAULT_ERROR_RATE: f64 = 0.01;
const DEFAULT_CAPACITY: usize = 100;

#[derive(PartialEq, Debug, Clone)]
pub struct ValueBloom {
    filter: BloomFilter,
    error_rate: f64,
    capacity: usize,
    insertions: usize,
}

impl ValueBloom {
    pub fn new(error_rate: f64, capacity: usize) -> Result<Self, OperationError> {
        if !(error_rate > 0.0 && error_rate < 1.0) || capacity == 0 {
            return Err(OperationError::ValueError(
                "ERR invalid bloom filter parameters".to_owned(),
            ));
        }

        Ok(ValueBloom {
            filter: BloomFilter::with_false_pos(error_rate).expected_items(capacity),
            error_rate,
            capacity,
            insertions: 0,
        })
    }

    pub fn default_filter() -> Self {
        ValueBloom::new(DEFAULT_ERROR_RATE, DEFAULT_CAPACITY)
            .expect("default bloom filter parameters are valid")
    }

    pub fn add(&mut self, item: &[u8]) -> bool {
        let may_have_existed = self.filter.insert(item);
        if !may_have_existed {
            self.insertions += 1;
        }
        !may_have_existed
    }

    pub fn exists(&self, item: &[u8]) -> bool {
        self.filter.contains(item)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn error_rate(&self) -> f64 {
        self.error_rate
    }

    pub fn insertions(&self) -> usize {
        self.insertions
    }

    pub fn memory_bytes(&self) -> u64 {
        (self.filter.as_slice().len() * std::mem::size_of::<u64>()) as u64
    }

    pub fn debug_object(&self) -> String {
        format!(
            "Value at:0x0000000000 refcount:0 encoding:bloom capacity:{} error_rate:{} \
             insertions:{} bits:{} hashes:{}",
            self.capacity,
            self.error_rate,
            self.insertions,
            self.filter.num_bits(),
            self.filter.num_hashes()
        )
    }
}
