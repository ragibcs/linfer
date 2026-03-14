pub const MAGIC: &[u8; 6] = b"LINFER";

#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub version: u32,
    pub flags: u64,
}

impl Header {
    pub fn encode(self) -> [u8; 18] {
        let mut out = [0_u8; 18];
        out[..6].copy_from_slice(MAGIC);
        out[6..10].copy_from_slice(&self.version.to_le_bytes());
        out[10..18].copy_from_slice(&self.flags.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 18 || &bytes[..6] != MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[6..10].try_into().ok()?);
        let flags = u64::from_le_bytes(bytes[10..18].try_into().ok()?);
        Some(Self { version, flags })
    }
}
