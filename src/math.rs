pub fn hash11(mut p: f32) -> f32 {
    p = (p * 0.1031).fract();
    p *= p + 33.33;
    p *= p + p;
    p.fract()
}

pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1. - t) + b * t
}

pub fn smoothstep(x: f32, edge0: f32, edge1: f32) -> f32 {
    let x = ((x - edge0) / (edge1 - edge0)).clamp(0., 1.);
    x * x * (3. - 2. * x)
}

// https://www.shadertoy.com/view/md2GWW
pub fn step_noise(x: f32, n: f32) -> f32 {
    let i = x.floor();
    let s = 0.1;
    let u = smoothstep(0.5 - s, 0.5 + s, x.fract());
    mix((hash11(i) * n).floor(), (hash11(i + 1.) * n).floor(), u) // from 0. to n - 1.
}
