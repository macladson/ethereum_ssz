use super::*;
use std::io::{self, Write};

mod impls;
mod stream_encoder;

pub use stream_encoder::SszStreamEncoder;

/// Provides SSZ encoding (serialization) via the `as_ssz_bytes(&self)` method.
///
/// See `examples/` for manual implementations or the crate root for implementations using
/// `#[derive(Encode)]`.
pub trait Encode {
    /// Returns `true` if this object has a fixed-length.
    ///
    /// I.e., there are no variable length items in this object or any of it's contained objects.
    fn is_ssz_fixed_len() -> bool;

    /// Append the encoding `self` to `buf`.
    ///
    /// Note, variable length objects need only to append their "variable length" portion, they do
    /// not need to provide their offset.
    fn ssz_append(&self, buf: &mut Vec<u8>);

    /// The number of bytes this object occupies in the fixed-length portion of the SSZ bytes.
    ///
    /// By default, this is set to `BYTES_PER_LENGTH_OFFSET` which is suitable for variable length
    /// objects, but not fixed-length objects. Fixed-length objects _must_ return a value which
    /// represents their length.
    fn ssz_fixed_len() -> usize {
        BYTES_PER_LENGTH_OFFSET
    }

    /// Returns the size (in bytes) when `self` is serialized.
    ///
    /// Returns the same value as `self.as_ssz_bytes().len()` but this method is significantly more
    /// efficient.
    fn ssz_bytes_len(&self) -> usize;

    /// Returns the full-form encoding of this object.
    ///
    /// The default implementation of this method should suffice for most cases.
    fn as_ssz_bytes(&self) -> Vec<u8> {
        let mut buf = vec![];

        self.ssz_append(&mut buf);

        buf
    }

    /// Encode this value as SSZ directly to a writer.
    ///
    /// The default implementation serializes to a `Vec<u8>` via `ssz_append` and writes it in one
    /// shot. Types with large encoded representations should override this to stream their encoding
    /// without needing to allocate.
    fn ssz_write(&self, w: &mut dyn Write) -> io::Result<()> {
        let mut buf = Vec::with_capacity(self.ssz_bytes_len());
        self.ssz_append(&mut buf);
        w.write_all(&buf)
    }
}

/// Allow for encoding an ordered series of distinct or indistinct objects as SSZ bytes.
///
/// **You must call `finalize(..)` after the final `append(..)` call** to ensure the bytes are
/// written to `buf`.
///
/// ## Example
///
/// Use `SszEncoder` to produce identical output to `foo.as_ssz_bytes()`:
///
/// ```rust
/// use ssz_derive::{Encode, Decode};
/// use ssz::{Decode, Encode, SszEncoder};
///
/// #[derive(PartialEq, Debug, Encode, Decode)]
/// struct Foo {
///     a: u64,
///     b: Vec<u16>,
/// }
///
/// fn ssz_encode_example() {
///     let foo = Foo {
///         a: 42,
///         b: vec![1, 3, 3, 7]
///     };
///
///     let mut buf: Vec<u8> = vec![];
///     let offset = <u64 as Encode>::ssz_fixed_len() + <Vec<u16> as Encode>::ssz_fixed_len();
///
///     let mut encoder = SszEncoder::container(&mut buf, offset);
///
///     encoder.append(&foo.a);
///     encoder.append(&foo.b);
///
///     encoder.finalize();
///
///     assert_eq!(foo.as_ssz_bytes(), buf);
/// }
///
/// ```
pub struct SszEncoder<'a> {
    offset: usize,
    buf: &'a mut Vec<u8>,
    variable_bytes: Vec<u8>,
}

impl<'a> SszEncoder<'a> {
    /// Instantiate a new encoder for encoding a SSZ container.
    pub fn container(buf: &'a mut Vec<u8>, num_fixed_bytes: usize) -> Self {
        buf.reserve(num_fixed_bytes);

        Self {
            offset: num_fixed_bytes,
            buf,
            variable_bytes: vec![],
        }
    }

    /// Append some `item` to the SSZ bytes.
    pub fn append<T: Encode>(&mut self, item: &T) {
        self.append_parameterized(T::is_ssz_fixed_len(), |buf| item.ssz_append(buf))
    }

    /// Uses `ssz_append` to append the encoding of some item to the SSZ bytes.
    pub fn append_parameterized<F>(&mut self, is_ssz_fixed_len: bool, ssz_append: F)
    where
        F: Fn(&mut Vec<u8>),
    {
        if is_ssz_fixed_len {
            ssz_append(self.buf);
        } else {
            self.buf
                .extend_from_slice(&encode_length(self.offset + self.variable_bytes.len()));

            ssz_append(&mut self.variable_bytes);
        }
    }

    /// Write the variable bytes to `self.bytes`.
    ///
    /// This method must be called after the final `append(..)` call when serializing
    /// variable-length items.
    pub fn finalize(&mut self) -> &mut Vec<u8> {
        self.buf.append(&mut self.variable_bytes);

        self.buf
    }
}

/// Encode `len` as a little-endian byte array of `BYTES_PER_LENGTH_OFFSET` length.
///
/// If `len` is larger than `2 ^ BYTES_PER_LENGTH_OFFSET`, a `debug_assert` is raised.
pub fn encode_length(len: usize) -> [u8; BYTES_PER_LENGTH_OFFSET] {
    // Note: it is possible for `len` to be larger than what can be encoded in
    // `BYTES_PER_LENGTH_OFFSET` bytes, triggering this debug assertion.
    //
    // These are the alternatives to using a `debug_assert` here:
    //
    // 1. Use `assert`.
    // 2. Push an error to the caller (e.g., `Option` or `Result`).
    // 3. Ignore it completely.
    //
    // I have avoided (1) because it's basically a choice between "produce invalid SSZ" or "kill
    // the entire program". I figure it may be possible for an attacker to trigger this assert and
    // take the program down -- I think producing invalid SSZ is a better option than this.
    //
    // I have avoided (2) because this error will need to be propagated upstream, making encoding a
    // function which may fail. I don't think this is ergonomic and the upsides don't outweigh the
    // downsides.
    //
    // I figure a `debug_assertion` is better than (3) as it will give us a change to detect the
    // error during testing.
    //
    // If you have a different opinion, feel free to start an issue and tag @paulhauner.
    debug_assert!(len <= MAX_LENGTH_VALUE);

    let mut bytes = [0; BYTES_PER_LENGTH_OFFSET];
    bytes.copy_from_slice(&len.to_le_bytes()[0..BYTES_PER_LENGTH_OFFSET]);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_length() {
        assert_eq!(encode_length(0), [0; 4]);

        assert_eq!(encode_length(1), [1, 0, 0, 0]);

        assert_eq!(
            encode_length(MAX_LENGTH_VALUE),
            [255; BYTES_PER_LENGTH_OFFSET]
        );
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_encode_length_above_max_debug_panics() {
        encode_length(MAX_LENGTH_VALUE + 1);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_encode_length_above_max_not_debug_does_not_panic() {
        assert_eq!(&encode_length(MAX_LENGTH_VALUE + 1)[..], &[0; 4]);
    }

    /// Compare SszStreamEncoder output to SszEncoder output for the same field sequence.
    macro_rules! assert_stream_eq {
        ($fixed_len:expr, $($item:expr),+ $(,)?) => {{
            let mut stream_out = Vec::new();
            let mut stream_encoder = SszStreamEncoder::container(&mut stream_out, $fixed_len);
            $( stream_encoder.append(&$item); )+
            stream_encoder.finalize().unwrap();

            let mut expected = Vec::new();
            let mut encoder = SszEncoder::container(&mut expected, $fixed_len);
            $( encoder.append(&$item); )+
            encoder.finalize();

            assert_eq!(stream_out, expected);
        }};
    }

    #[test]
    fn ssz_stream_encoder() {
        let a: u64 = 42;
        let b: u16 = 7;
        let c: Vec<u16> = vec![1, 2, 3];
        let d: Vec<u8> = vec![];

        // Fixed-only.
        let fixed_len = <u64 as Encode>::ssz_fixed_len() + <u16 as Encode>::ssz_fixed_len();
        assert_stream_eq!(fixed_len, a, b);

        // Mixed fixed + variable.
        let fixed_len = <u64 as Encode>::ssz_fixed_len() + <Vec<u16> as Encode>::ssz_fixed_len();
        assert_stream_eq!(fixed_len, a, c);

        // Variable-fixed-variable interleaving, including empty vec.
        let fixed_len = <Vec<u16> as Encode>::ssz_fixed_len()
            + <u64 as Encode>::ssz_fixed_len()
            + <Vec<u8> as Encode>::ssz_fixed_len();
        assert_stream_eq!(fixed_len, c, a, d);
    }

    #[test]
    fn ssz_write_matches_as_ssz_bytes() {
        fn check<T: Encode>(val: &T) {
            let mut write_buf = Vec::new();
            val.ssz_write(&mut write_buf).unwrap();
            assert_eq!(write_buf, val.as_ssz_bytes());
        }

        check(&0xDEADBEEFu32);
        check(&vec![1u8, 2, 3, 4]);
        check(&vec![vec![1u8, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn ssz_stream_encoder_parameterized() {
        let a: u64 = 42;
        let b: Vec<u16> = vec![1, 2, 3];
        let fixed_len = <u64 as Encode>::ssz_fixed_len() + <Vec<u16> as Encode>::ssz_fixed_len();

        let mut out = Vec::new();
        let mut stream_encoder = SszStreamEncoder::container(&mut out, fixed_len);
        stream_encoder.append(&a);
        let b_ref = &b;
        stream_encoder.append_parameterized(
            <Vec<u16> as Encode>::is_ssz_fixed_len(),
            b_ref.ssz_bytes_len(),
            |buf| b_ref.ssz_append(buf),
            |w| b_ref.ssz_write(w),
        );
        stream_encoder.finalize().unwrap();

        let mut expected = Vec::new();
        let mut encoder = SszEncoder::container(&mut expected, fixed_len);
        encoder.append(&a);
        encoder.append(&b);
        encoder.finalize();

        assert_eq!(out, expected);
    }
}
