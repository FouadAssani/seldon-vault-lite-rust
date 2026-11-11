//! Decoded raster image shared by the native media decoders.
//!
//! PNG and JPEG both produce interleaved 8-bit samples in one of three
//! channel layouts: grayscale (1), RGB (3), or RGBA (4). Keeping one type
//! avoids a lossy conversion at the decoder boundary and gives the image
//! preprocessing stage a single validated input.
//!
//! The constructor is crate-private: a decoder must validate dimensions,
//! channel count, and buffer length before it can create an image.

use crate::error::{Error, Result};

/// A decoded raster image with interleaved 8-bit channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    width: usize,
    height: usize,
    channels: u8,
    pixels: Vec<u8>,
}

impl Image {
    /// Creates an image after checking its dimensions and buffer length.
    ///
    /// # Errors
    ///
    /// Returns an internal error when the dimensions are zero, the channel
    /// count is not 1, 3, or 4, the pixel count overflows, or the buffer length
    /// does not match the declared geometry. Decoders reject hostile input
    /// before calling this constructor, so a failure here indicates a bug.
    pub(crate) fn new(width: usize, height: usize, channels: u8, pixels: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::internal("decoded image dimensions must be positive"));
        }
        if !matches!(channels, 1 | 3 | 4) {
            return Err(Error::internal(format!(
                "decoded image has unsupported channel count {channels}"
            )));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(usize::from(channels)))
            .ok_or_else(|| Error::limit_exceeded("decoded image length overflow"))?;
        if pixels.len() != expected {
            return Err(Error::internal(format!(
                "decoded image has {} bytes, expected {expected}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels,
            pixels,
        })
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of interleaved channels per pixel: 1, 3, or 4.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Interleaved pixel bytes, row-major.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Borrows the channels of one pixel, or None when out of bounds.
    pub fn pixel(&self, x: usize, y: usize) -> Option<&[u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let channels = usize::from(self.channels);
        let start = (y * self.width + x) * channels;
        Some(&self.pixels[start..start + channels])
    }
}
