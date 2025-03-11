use rodio::Source;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// TeeSource wraps an inner f32 source and pushes every sample into a shared buffer.
/// It maintains a maximum buffer size to prevent unbounded memory growth.
pub struct TeeSource<S: Source<Item = f32>> {
    inner: S,
    pub buffer: Arc<Mutex<Vec<f32>>>,
    max_buffer_size: usize,
}

impl<S: Source<Item = f32>> TeeSource<S> {
    pub fn new(source: S, buffer: Arc<Mutex<Vec<f32>>>) -> Self {
        Self {
            inner: source,
            buffer,
            max_buffer_size: 32768, // Default max buffer size
        }
    }
}

impl<S: Source<Item = f32>> Iterator for TeeSource<S> {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(sample) = self.inner.next() {
            // Push sample into the shared buffer while maintaining max size
            let mut buffer_guard = self.buffer.lock().unwrap();
            buffer_guard.push(sample);
            
            // Trim buffer if needed
            if buffer_guard.len() > self.max_buffer_size {
                let overflow = buffer_guard.len() - self.max_buffer_size;
                buffer_guard.drain(0..overflow);
            }
            
            drop(buffer_guard); // Release lock as soon as possible
            Some(sample)
        } else {
            None
        }
    }
}

impl<S: Source<Item = f32>> Source for TeeSource<S> {
    fn current_frame_len(&self) -> Option<usize> {
        self.inner.current_frame_len()
    }
    fn channels(&self) -> u16 {
        self.inner.channels()
    }
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }
}
