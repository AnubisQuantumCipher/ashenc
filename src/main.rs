use std::{
    fs::File,
    io::{self, Read, Write, BufReader, BufWriter},
    path::PathBuf,
};
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use getrandom::getrandom;
use zeroize::Zeroize;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use aes_gcm::{
    Aes256Gcm,
    aead::{
        Aead, AeadCore, KeyInit, Payload,
        generic_array::GenericArray
    },
};
use aes_kw::{wrap::Aes256Kw, unwrap::Aes256KwUnwrap};
use pbkdf2::pbkdf2_hmac;
use hmac::Hmac;
use sha2::Sha256;

const MAGIC: &[u8; 4] = b"ASH1";
const ALG_AES256_GCM: u8 = 1;
const KW_AES_KW: u8 = 1;

#[derive(Parser)]
#[command(name = "ashenc", version)]
struct Cli { #[command(subcommand)] cmd: Cmd }

#[derive(Subcommand)]
enum Cmd {
    /// Generate a random 256-bit key (base64)
    Keygen,
    /// Encrypt a file
    Encrypt {
        #[arg(short, long)] input: PathBuf,
        #[arg(short, long)] output: PathBuf,
        /// Read passphrase from stdin (recommended on a-Shell/WASI)
        #[arg(long)] pass_stdin: bool,
        /// Raw KEK (base64, 32 bytes) to bypass PBKDF2
        #[arg(long)] kek_b64: Option<String>,
        /// PBKDF2 iterations (default ~600k)
        #[arg(long, default_value_t = 600_000)] iter: u32,
        /// Optional AAD string bound to ciphertext
        #[arg(long)] aad: Option<String>,
        /// Streaming chunk size (bytes), default 1 MiB
        #[arg(long, default_value_t = 1_048_576)] chunk: u32,
    },
    /// Decrypt a file
    Decrypt {
        #[arg(short, long)] input: PathBuf,
        #[arg(short, long)] output: PathBuf,
        #[arg(long)] pass_stdin: bool,
        #[arg(long)] kek_b64: Option<String>,
    },
}

fn read_passphrase_from_stdin() -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf)?;
    while buf.last().is_some_and(|b| *b == b'\n' || *b == b'\r') { buf.pop(); }
    Ok(buf)
}

fn gen_random(len: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; len];
    getrandom(&mut v)?;
    Ok(v)
}

#[derive(Debug)]
struct Header {
    version: u8,
    alg: u8,
    flags: u32,
    iv: [u8; 12],      // base nonce (last 4 bytes become a BE counter)
    salt: Vec<u8>,     // PBKDF2 salt (empty if kek_b64 used)
    iter: u32,         // PBKDF2 iters (0 if kek_b64 used)
    kw_alg: u8,        // 1 = AES-KW/KWP
    wrapped: Vec<u8>,  // wrapped DEK
    chunk: u32,        // chunk size for streaming
    aad: Vec<u8>,      // external AAD (optional)
}

impl Header {
    fn write_to(&self, mut w: impl Write) -> Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&[self.version])?;
        w.write_all(&[self.alg])?;
        w.write_u32::<LittleEndian>(self.flags)?;
        w.write_all(&[self.iv.len() as u8])?;
        w.write_all(&[self.salt.len() as u8])?;
        w.write_u32::<LittleEndian>(self.iter)?;
        w.write_all(&[self.kw_alg])?;
        w.write_u16::<LittleEndian>(self.wrapped.len() as u16)?;
        w.write_u32::<LittleEndian>(self.chunk)?;
        w.write_u16::<LittleEndian>(self.aad.len() as u16)?;
        w.write_all(&self.salt)?;
        w.write_all(&self.iv)?;
        w.write_all(&self.wrapped)?;
        if !self.aad.is_empty() { w.write_all(&self.aad)?; }
        Ok(())
    }

    fn read_from(mut r: impl Read) -> Result<Self> {
        let mut m = [0u8;4]; r.read_exact(&mut m)?; if &m != MAGIC { return Err(anyhow!("bad magic")); }
        let mut b = [0u8;1];
        r.read_exact(&mut b)?; let version = b[0];
        r.read_exact(&mut b)?; let alg = b[0];
        let flags = r.read_u32::<LittleEndian>()?;
        r.read_exact(&mut b)?; let iv_len = b[0] as usize;
        r.read_exact(&mut b)?; let salt_len = b[0] as usize;
        let iter = r.read_u32::<LittleEndian>()?;
        r.read_exact(&mut b)?; let kw_alg = b[0];
        let wrapped_len = r.read_u16::<LittleEndian>()? as usize;
        let chunk = r.read_u32::<LittleEndian>()?;
        let aad_len = r.read_u16::<LittleEndian>()? as usize;
        if iv_len != 12 { return Err(anyhow!("bad IV length")); }
        let mut salt = vec![0u8; salt_len]; r.read_exact(&mut salt)?;
        let mut iv = [0u8;12]; r.read_exact(&mut iv)?;
        let mut wrapped = vec![0u8; wrapped_len]; r.read_exact(&mut wrapped)?;
        let mut aad = vec![0u8; aad_len]; if aad_len>0 { r.read_exact(&mut aad)?; }
        Ok(Header{ version, alg, flags, iv, salt, iter, kw_alg, wrapped, chunk, aad })
    }
}

fn derive_kek(pass: &[u8], salt: &[u8], iter: u32) -> [u8;32] {
    let mut out = [0u8;32];
    pbkdf2_hmac::<Hmac<Sha256>>(pass, salt, iter, &mut out);
    out
}

type NonceBytes = GenericArray<u8, <Aes256Gcm as AeadCore>::NonceSize>;

fn nonce_from(base: &[u8;12], counter: u32) -> NonceBytes {
    let mut n = [0u8;12];
    n.copy_from_slice(base);
    n[8..12].copy_from_slice(&counter.to_be_bytes());
    GenericArray::clone_from_slice(&n)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Keygen => {
            let key = gen_random(32)?;
            println!("{}", STANDARD.encode(key));
        }
        Cmd::Encrypt { input, output, pass_stdin, kek_b64, iter, aad, chunk } => {
            let aad_bytes = aad.unwrap_or_default().into_bytes();
            // KEK setup (password or raw KEK)
            let (kek, salt, iter_eff) = if let Some(b64) = kek_b64 {
                let k = STANDARD.decode(b64)?;
                if k.len()!=32 { return Err(anyhow!("KEK must be 32 bytes (base64)")); }
                let mut a = [0u8;32]; a.copy_from_slice(&k);
                (a, Vec::new(), 0)
            } else {
                if !pass_stdin { return Err(anyhow!("Use --pass-stdin (no interactive prompt in WASI).")); }
                let pass = read_passphrase_from_stdin()?;
                let salt = gen_random(16)?;
                let kek = derive_kek(&pass, &salt, iter);
                let mut pass_mut = pass; pass_mut.zeroize();
                (kek, salt, iter)
            };
            // Per-file DEK and header
            let dek = gen_random(32)?;
            let iv_rand = gen_random(12)?; let mut iv = [0u8;12]; iv.copy_from_slice(&iv_rand);
            let wrapped = Aes256Kw::wrap_with_padding(&kek, &dek).map_err(|_| anyhow!("AES-KW wrap failed"))?;
            let header = Header {
                version: 1, alg: ALG_AES256_GCM, flags: 1, // bit0: streaming
                iv, salt, iter: iter_eff, kw_alg: KW_AES_KW, wrapped,
                chunk, aad: aad_bytes.clone()
            };
            // Write header
            let fin = File::open(&input).with_context(|| format!("open input {}", input.display()))?;
            let mut reader = BufReader::new(fin);
            let fout = File::create(&output).with_context(|| format!("create output {}", output.display()))?;
            let mut writer = BufWriter::new(fout);
            header.write_to(&mut writer)?;
            // AEAD context with DEK
            let aead = Aes256Gcm::new_from_slice(&dek).map_err(|_| anyhow!("bad DEK size"))?;
            // Stream encrypt: counter = 1,2,3...
            let mut buf = vec![0u8; chunk as usize];
            let mut counter: u32 = 1;
            loop {
                let n = reader.read(&mut buf)?;
                if n == 0 { break; }
                let nonce = nonce_from(&header.iv, counter);
                let ct = aead.encrypt(&nonce, Payload { msg: &buf[..n], aad: &header.aad })
                    .map_err(|_| anyhow!("encrypt failed"))?;
                writer.write_all(&ct)?;
                counter = counter.wrapping_add(1);
            }
            writer.flush()?;
            // Zeroize secrets
            let mut dek_mut = dek; dek_mut.zeroize();
            let mut kek_mut = kek; kek_mut.zeroize();
        }
        Cmd::Decrypt { input, output, pass_stdin, kek_b64 } => {
            // Read header
            let fin = File::open(&input).with_context(|| format!("open input {}", input.display()))?;
            let mut reader = BufReader::new(fin);
            let header = Header::read_from(&mut reader)?;
            if header.version != 1 { return Err(anyhow!("unsupported version")); }
            if header.alg != ALG_AES256_GCM { return Err(anyhow!("unsupported algorithm")); }
            // KEK
            let kek = if let Some(b64) = kek_b64 {
                let k = STANDARD.decode(b64)?;
                if k.len()!=32 { return Err(anyhow!("KEK must be 32 bytes")); }
                let mut a = [0u8;32]; a.copy_from_slice(&k); a
            } else {
                if !pass_stdin { return Err(anyhow!("Use --pass-stdin or --kek-b64")); }
                let pass = read_passphrase_from_stdin()?;
                let kek = derive_kek(&pass, &header.salt, header.iter);
                let mut pass_mut = pass; pass_mut.zeroize();
                kek
            };
            // Unwrap DEK and set up AEAD
            let dek = Aes256KwUnwrap::unwrap_with_padding(&kek, &header.wrapped)
                .map_err(|_| anyhow!("AES-KW unwrap failed"))?;
            let aead = Aes256Gcm::new_from_slice(&dek).map_err(|_| anyhow!("bad DEK size"))?;
            let mut dek_mut = dek; dek_mut.zeroize();
            let mut kek_mut = kek; kek_mut.zeroize();
            let fout = File::create(&output).with_context(|| format!("create output {}", output.display()))?;
            let mut writer = BufWriter::new(fout);
            // Each ciphertext segment is (chunk bytes of PT) + 16-byte tag.
            let seg_ct_max = header.chunk.max(1) as usize + 16;
            let mut counter: u32 = 1;
            loop {
                // fill up to seg_ct_max or hit EOF for the final (short) segment
                let mut seg = vec![0u8; seg_ct_max];
                let mut have = 0usize;
                while have < seg.len() {
                    let m = reader.read(&mut seg[have..])?;
                    if m == 0 { break; }
                    have += m;
                    if have == seg.len() { break; }
                }
                if have == 0 { break; } // no more data at all
                let nonce = nonce_from(&header.iv, counter);
                let pt = aead.decrypt(&nonce, Payload { msg: &seg[..have], aad: &header.aad })
                    .map_err(|_| anyhow!("tag check failed (wrong passphrase or file corrupted)"))?;
                writer.write_all(&pt)?;
                counter = counter.wrapping_add(1);
                if have < seg.len() { break; } // that was the last (short) segment
            }
            writer.flush()?;
        }
    }
    Ok(())
}

