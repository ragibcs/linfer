#[derive(Debug, Clone)]
pub struct Q8Row {
    pub scale: f32,
    pub data: Vec<i8>,
}

pub fn quantize_q8_0(src: &[f32], row_size: usize) -> Vec<Q8Row> {
    let mut rows = Vec::new();
    for row in src.chunks(row_size) {
        let absmax = row
            .iter()
            .fold(0.0_f32, |acc, &x| if x.abs() > acc { x.abs() } else { acc });
        let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
        let data = row
            .iter()
            .map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        rows.push(Q8Row { scale, data });
    }
    rows
}
