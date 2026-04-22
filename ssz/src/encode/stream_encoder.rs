use super::{encode_length, Encode};
use std::io::{self, Write};

type WriteFn<'a> = Box<dyn FnOnce(&mut dyn Write) -> io::Result<()> + 'a>;

/// Streaming SSZ container encoder that writes to a [`Write`] sink.
///
/// Unlike [`super::SszEncoder`] which buffers variable-length data in a `Vec<u8>`, this encoder
/// pre-computes the total size of each variable field via [`Encode::ssz_bytes_len`], writes the
/// fixed part immediately, then streams each variable field directly to the writer via
/// [`Encode::ssz_write`].
///
/// **You must call [`finalize`](SszStreamEncoder::finalize) after the final `append` call.**
///
/// ## Example
///
/// ```rust,ignore
/// let mut encoder = SszStreamEncoder::container(&mut writer, fixed_part_size);
/// encoder.append(&self.slot);       // fixed-len: buffered
/// encoder.append(&self.validators); // variable-len: offset buffered, contents streamed
/// encoder.finalize()?;              // flushes fixed part, then streams variable fields
/// ```
pub struct SszStreamEncoder<'a> {
    writer: &'a mut dyn Write,
    /// Buffer for the fixed part: inline fixed-length fields + 4-byte offsets.
    fixed_buf: Vec<u8>,
    /// Running byte offset into the variable part (starts at fixed_part_size).
    offset: usize,
    /// Deferred variable-length field writes, executed in order during `finalize`.
    variable_fields: Vec<WriteFn<'a>>,
}

impl<'a> SszStreamEncoder<'a> {
    /// Create a new streaming encoder targeting the given writer.
    ///
    /// `num_fixed_bytes` is the total size of the fixed part (sum of `ssz_fixed_len()` for all
    /// fields, which equals the fixed-field sizes plus 4 bytes per variable-length field offset).
    pub fn container(writer: &'a mut dyn Write, num_fixed_bytes: usize) -> Self {
        Self {
            writer,
            fixed_buf: Vec::with_capacity(num_fixed_bytes),
            offset: num_fixed_bytes,
            variable_fields: Vec::new(),
        }
    }

    /// Append a field to the encoding.
    ///
    /// Fixed-length fields are written inline to the fixed part buffer. Variable-length fields
    /// have their offset recorded in the fixed part but their data is only written during
    /// [`finalize`](Self::finalize).
    pub fn append<T: Encode>(&mut self, item: &'a T) {
        if T::is_ssz_fixed_len() {
            item.ssz_append(&mut self.fixed_buf);
        } else {
            // Write the 4-byte offset pointing to where this field's data will appear in the
            // variable part.
            self.fixed_buf
                .extend_from_slice(&encode_length(self.offset));

            // Advance offset by the field's encoded size.
            self.offset += item.ssz_bytes_len();

            // Defer the actual data write.
            self.variable_fields
                .push(Box::new(move |writer| item.ssz_write(writer)));
        }
    }

    /// Append a field using custom encoding functions.
    ///
    /// This supports `#[ssz(with = module)]` fields in the derive macro. The `ssz_append` closure
    /// is used for fixed-length fields (written into the fixed buffer), while `ssz_write` is used
    /// for variable-length fields (deferred to `finalize`).
    pub fn append_parameterized(
        &mut self,
        is_ssz_fixed_len: bool,
        ssz_bytes_len: usize,
        ssz_append: impl FnOnce(&mut Vec<u8>) + 'a,
        ssz_write: impl FnOnce(&mut dyn Write) -> io::Result<()> + 'a,
    ) {
        if is_ssz_fixed_len {
            ssz_append(&mut self.fixed_buf);
        } else {
            self.fixed_buf
                .extend_from_slice(&encode_length(self.offset));
            self.offset += ssz_bytes_len;
            self.variable_fields.push(Box::new(ssz_write));
        }
    }

    /// Flush the fixed part, then stream all variable-length fields in order.
    ///
    /// This must be called after the final `append` call.
    pub fn finalize(self) -> io::Result<()> {
        debug_assert_eq!(
            self.fixed_buf.len(),
            self.fixed_buf.capacity(),
            "fixed part buffer was not fully populated"
        );
        self.writer.write_all(&self.fixed_buf)?;
        for write_fn in self.variable_fields {
            write_fn(self.writer)?;
        }
        Ok(())
    }
}
