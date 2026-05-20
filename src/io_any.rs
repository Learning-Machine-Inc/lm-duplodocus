//! Format-agnostic input reader for duplodocus.
//!
//! Returns the same `mj_io::FileReader` type as `mj_io::read_pathbuf`, so
//! callers can keep their existing `data.lines() -> serde_json::from_str`
//! pattern with no code changes beyond the function name. For parquet input,
//! we synthesize a `FileReader::Stream(...)` whose underlying `BufRead`
//! streams each row as a newline-delimited JSON object.
//!
//! Companion `expand_dirs_any` adds `.parquet` to the default extension set
//! that `mj_io::expand_dirs` accepts (`.jsonl`, `.jsonl.gz`, `.jsonl.zst`,
//! `.jsonl.zstd`, `.json.gz`, `.json.zst`).
//!
//! The parquet adapter assumes the data-preprocess output schema (six
//! columns: `collection_id`, `collection_meta`, `doc_id`, `doc_index`,
//! `doc_meta`, `content`). Each row becomes a JSON object keyed by those
//! column names, produced via `arrow_json::LineDelimitedWriter` — the same
//! encoder Arrow uses for its own JSON output.

use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use arrow_json::LineDelimitedWriter;
use mj_io::FileReader;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

/// Same set as `mj_io::expand_dirs`'s defaults, plus `.parquet`.
const ACCEPTED_EXTS: &[&str] = &[
    ".jsonl",
    ".jsonl.gz",
    ".jsonl.zstd",
    ".jsonl.zst",
    ".json.gz",
    ".json.zst",
    ".parquet",
];

/// Drop-in for `mj_io::read_pathbuf`: returns a `FileReader` that supports
/// `.parquet` in addition to the JSONL formats mj_io already handles.
///
/// The `stream` flag is forwarded to `mj_io` for non-parquet paths; parquet
/// always streams batches and the flag is ignored.
pub fn read_any(path: &Path, stream: bool) -> Result<FileReader> {
    if path.extension().and_then(|e| e.to_str()) == Some("parquet") {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let buf: Box<dyn BufRead> = Box::new(BufReader::new(ParquetJsonlReader::new(reader)));
        Ok(FileReader::Stream(buf))
    } else {
        mj_io::read_pathbuf(&path.to_path_buf(), stream)
    }
}

/// Drop-in for `mj_io::expand_dirs(_, None)`: recursively walks `roots`,
/// returning files whose names end with any of the supported extensions
/// (JSONL family + `.parquet`).
pub fn expand_dirs_any(roots: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    mj_io::expand_dirs(roots, Some(ACCEPTED_EXTS))
}

/// Like `mj_io::get_output_filename`, but normalizes the output filename to
/// always end in `.jsonl.zst` regardless of the input extension.
///
/// duplodocus always writes JSON-serialized rows via `write_line`, so the
/// output is always JSONL in spirit. Forcing `.jsonl.zst` ensures
/// `mj_io::create_writer` picks the zstd encoder branch and that downstream
/// tools can rely on a consistent output format — instead of inheriting
/// whatever extension the input happened to use (e.g. `.parquet`).
pub fn get_output_filename_jsonl_zst(
    input_path: &Path,
    config_input_dir: &Path,
    config_output_dir: &Path,
) -> Result<PathBuf> {
    let rel = input_path
        .strip_prefix(config_input_dir)
        .with_context(|| {
            format!(
                "{} is not under {}",
                input_path.display(),
                config_input_dir.display(),
            )
        })?;
    let rel_str = rel.to_string_lossy();
    // Strip the longest matching input extension; fall back to the bare name.
    const INPUT_EXTS: &[&str] = &[
        ".jsonl.zst",
        ".jsonl.zstd",
        ".jsonl.gz",
        ".json.zst",
        ".json.gz",
        ".jsonl",
        ".parquet",
    ];
    let stem: &str = INPUT_EXTS
        .iter()
        .find_map(|ext| rel_str.strip_suffix(ext))
        .unwrap_or(rel_str.as_ref());
    Ok(config_output_dir.join(format!("{stem}.jsonl.zst")))
}

/// Streams a parquet file's rows as newline-delimited JSON via `Read`.
///
/// Pulls one `RecordBatch` at a time from the parquet reader, encodes it to
/// JSONL bytes with `arrow_json::LineDelimitedWriter`, and serves those bytes
/// to the wrapping `BufReader`. Memory footprint is bounded by one batch
/// worth of encoded JSON (parquet defaults to row groups of ~1M rows, but
/// the batch iterator yields smaller chunks).
pub struct ParquetJsonlReader {
    batches: ParquetRecordBatchReader,
    buf: Cursor<Vec<u8>>,
}

impl ParquetJsonlReader {
    pub fn new(batches: ParquetRecordBatchReader) -> Self {
        Self {
            batches,
            buf: Cursor::new(Vec::new()),
        }
    }
}

impl Read for ParquetJsonlReader {
    fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let n = self.buf.read(dst)?;
            if n > 0 {
                return Ok(n);
            }
            // Current batch drained; pull the next one.
            match self.batches.next() {
                Some(Ok(batch)) => {
                    let mut bytes = Vec::with_capacity(batch.num_rows() * 256);
                    {
                        let mut w = LineDelimitedWriter::new(&mut bytes);
                        w.write(&batch).map_err(std::io::Error::other)?;
                        w.finish().map_err(std::io::Error::other)?;
                    }
                    self.buf = Cursor::new(bytes);
                }
                Some(Err(e)) => return Err(std::io::Error::other(e)),
                None => return Ok(0), // EOF
            }
        }
    }
}
