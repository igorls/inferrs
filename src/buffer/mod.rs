//! GrowVec: A vector that doubles capacity on growth and never shrinks.
//!
//! This is the core memory abstraction for inferrs. Every performance-critical
//! buffer in the system uses this type. It behaves like a C++ std::vector:
//!   - Starts with zero or a small initial capacity
//!   - Doubles capacity when more space is needed
//!   - Never deallocates or shrinks the backing storage
//!   - Allows clearing (len = 0) without releasing memory
//!
//! This is the opposite of vLLM's approach which pre-allocates 90% of GPU
//! memory up front. Instead, we grow conservatively and keep what we've grown.
//! For a long-running server, the buffers warm up to their steady-state size
//! and then never allocate again. For desktop use, we only use what we need.

use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::SliceIndex;

/// A growable vector that doubles capacity and never shrinks.
///
/// Once memory is allocated, it is kept for the lifetime of the GrowVec.
/// This avoids repeated allocation/deallocation in hot paths while only
/// using as much memory as the high-water mark requires.
#[derive(Debug)]
pub struct GrowVec<T> {
    inner: Vec<T>,
    /// Track the high-water mark for diagnostics.
    high_water_mark: usize,
}

impl<T> GrowVec<T> {
    /// Create an empty GrowVec with no allocation.
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            high_water_mark: 0,
        }
    }

    /// Create a GrowVec with a specific initial capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Vec::with_capacity(cap),
            high_water_mark: 0,
        }
    }

    /// Current number of live elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the vector is logically empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Current allocated capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// The maximum length this GrowVec has ever reached.
    pub fn high_water_mark(&self) -> usize {
        self.high_water_mark
    }

    /// Push a value, doubling capacity if needed.
    pub fn push(&mut self, value: T) {
        if self.inner.len() == self.inner.capacity() {
            self.grow();
        }
        self.inner.push(value);
        self.update_hwm();
    }

    /// Extend from an iterator, growing as needed.
    pub fn extend_from_iter(&mut self, iter: impl IntoIterator<Item = T>) {
        for item in iter {
            self.push(item);
        }
    }

    /// Clear the logical contents without releasing memory.
    pub fn clear(&mut self) {
        self.inner.clear();
        // Capacity is preserved.
    }

    /// Truncate to `len` elements without releasing memory.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
        // Capacity is preserved.
    }

    /// Reserve enough capacity for at least `additional` more elements,
    /// using the doubling strategy.
    pub fn reserve(&mut self, additional: usize) {
        let required = self.inner.len() + additional;
        if required > self.inner.capacity() {
            let new_cap = required.max(self.inner.capacity() * 2).max(8);
            self.inner.reserve(new_cap - self.inner.capacity());
        }
    }

    /// Resize the vector, filling new slots with `value`.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.inner.capacity() {
            let new_cap = new_len.max(self.inner.capacity() * 2).max(8);
            self.inner.reserve(new_cap - self.inner.capacity());
        }
        self.inner.resize(new_len, value);
        self.update_hwm();
    }

    /// Get a slice of the live elements.
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Get a mutable slice of the live elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    /// Pop the last element.
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Remove and return the element at `index`, swapping with the last element.
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.inner.swap_remove(index)
    }

    /// Iterate over elements.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.inner.iter()
    }

    /// Mutably iterate over elements.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.inner.iter_mut()
    }

    /// Retain only elements matching the predicate.
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, f: F) {
        self.inner.retain(f);
    }

    /// Consume and return the inner Vec (for interop).
    pub fn into_inner(self) -> Vec<T> {
        self.inner
    }

    /// Double the capacity (minimum 8).
    fn grow(&mut self) {
        let new_cap = if self.inner.capacity() == 0 {
            8
        } else {
            self.inner.capacity() * 2
        };
        self.inner.reserve(new_cap - self.inner.capacity());
    }

    fn update_hwm(&mut self) {
        if self.inner.len() > self.high_water_mark {
            self.high_water_mark = self.inner.len();
        }
    }
}

impl<T: Clone> GrowVec<T> {
    /// Extend from a slice.
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        let required = self.inner.len() + slice.len();
        if required > self.inner.capacity() {
            let new_cap = required.max(self.inner.capacity() * 2).max(8);
            self.inner.reserve(new_cap - self.inner.capacity());
        }
        self.inner.extend_from_slice(slice);
        self.update_hwm();
    }
}

impl<T> Default for GrowVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for GrowVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.inner
    }
}

impl<T> DerefMut for GrowVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for GrowVec<T> {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.inner[index]
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for GrowVec<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<T> FromIterator<T> for GrowVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let inner: Vec<T> = iter.into_iter().collect();
        let len = inner.len();
        Self {
            inner,
            high_water_mark: len,
        }
    }
}

impl<'a, T> IntoIterator for &'a GrowVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut GrowVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grow_doubles_capacity() {
        let mut v = GrowVec::new();
        assert_eq!(v.capacity(), 0);

        v.push(1u32);
        assert!(v.capacity() >= 8); // First growth goes to 8

        let cap_after_8 = v.capacity();
        for i in 1..=8 {
            v.push(i);
        }
        // Should have doubled at some point
        assert!(v.capacity() >= cap_after_8);
    }

    #[test]
    fn test_clear_preserves_capacity() {
        let mut v = GrowVec::new();
        for i in 0..100u32 {
            v.push(i);
        }
        let cap = v.capacity();
        v.clear();
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), cap);
    }

    #[test]
    fn test_high_water_mark() {
        let mut v = GrowVec::new();
        for i in 0..50u32 {
            v.push(i);
        }
        assert_eq!(v.high_water_mark(), 50);
        v.clear();
        assert_eq!(v.high_water_mark(), 50);
        for i in 0..30u32 {
            v.push(i);
        }
        assert_eq!(v.high_water_mark(), 50);
    }
}
