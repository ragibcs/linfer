use half::f16;

#[derive(Debug, Clone)]
pub struct Q4Block {
    pub d: f16,
    pub qs: [u8; 16],
}

pub fn quantize_q4_0(src: &[f32]) -> Vec<Q4Block> {
    let mut blocks = Vec::new();

    for chunk in src.chunks(32) {
        let mut padded = [0.0_f32; 32];
        padded[..chunk.len()].copy_from_slice(chunk);

        let absmax = padded
            .iter()
            .fold(0.0_f32, |acc, &x| if x.abs() > acc { x.abs() } else { acc });
        let scale = if absmax == 0.0 { 1.0 } else { absmax / 7.0 };

        let mut qs = [0_u8; 16];
        for i in 0..16 {
            let a = quantize_to_4bit(padded[i * 2], scale);
            let b = quantize_to_4bit(padded[i * 2 + 1], scale);
            qs[i] = (a & 0x0F) | ((b & 0x0F) << 4);
        }

        blocks.push(Q4Block {
            d: f16::from_f32(scale),
            qs,
        });
    }

    blocks
}

fn quantize_to_4bit(v: f32, scale: f32) -> u8 {
    let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
    ((q + 8) as u8) & 0x0F
}
