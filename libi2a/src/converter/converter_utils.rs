use imageproc::image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

pub fn get_intensity(pixel: Rgba<u8>) -> f64 {
    //Magic numbers
    const LINEAR_POWER: f64 = 2.2_f64;
    const CR: f64 = 0.2126_f64;
    const CG: f64 = 0.7152_f64;
    const CB: f64 = 0.0722_f64;
    const CH: f64 = 1.0_f64;
    const CS: f64 = 1.0_f64;
    const CV: f64 = 1.2_f64;
    
    //Convert to HSV
    let (h, s, v) = rgb_to_hsv((pixel.0[0], pixel.0[1], pixel.0[2]));
    
    //Scale HSV
    let (sh, ss, sv) = (CH * h, CS * s, CV * v);
    
    //Convert back to rgb
    let (r, g, b) = hsv_to_rgb((sh, ss, sv));
    
    //'Linearize' channels
    let r_lin = (r as f64 / 255_f64).powf(LINEAR_POWER);
    let g_lin = (g as f64 / 255_f64).powf(LINEAR_POWER);
    let b_lin = (b as f64 / 255_f64).powf(LINEAR_POWER);
    
    //Intensity as linear combination
    CR * r_lin + CG * g_lin + CB * b_lin
}

pub fn get_color(rgb: (u8, u8, u8)) -> u32 {
    ((rgb.0 as u32) << 16) + ((rgb.1 as u32) << 8) + rgb.2 as u32
}

pub fn gaussian(w: usize) -> Vec<f64> {
    let mut g = vec![0_f64; w * w];
    
    let stddev = (w as f64).sqrt().log2() / 1.5_f64;
    
    let center = (w as f64 - 1_f64) / 2_f64;
    
    let mut sum = 0_f64;
    
    //Generate circular gaussian matrix
    for i in 0..w {
        for j in 0..w {
            let value = 1_f64 / (2_f64 * std::f64::consts::PI * stddev.powf(2_f64))
            * std::f64::consts::E.powf(((i as f64 - center).powf(2_f64) + (j as f64 - center).powf(2_f64))
            / (-2_f64 * stddev.powf(2_f64)));
            
            sum += value;
            g[(w * j) + i] = value;
        }
    }
    
    //Normalize to sum 1
    for item in &mut g {
        *item /= sum;
    }
    
    g
}

pub fn rgb_to_hsv(rgb: (u8, u8, u8)) -> (f64, f64, f64) {
    let (r, g, b) = (rgb.0 as f64 / 255_f64, rgb.1 as f64 / 255_f64, rgb.2 as f64 / 255_f64);
    let xp_min = f64::min(r, f64::min(g, b));
    let xp_max = f64::max(r, f64::max(g, b));
    let xp_delta = xp_max - xp_min;
    
    //Value
    let v = xp_max;
    
    //Saturation
    let s = if xp_max == 0_f64 {
        0_f64
    }
    else {
        xp_delta / xp_max
    };
    
    //No chroma
    if xp_delta == 0_f64 {
        return (0_f64, s, v);
    }
    
    //Set chroma data
    let mut h: f64 = 0_f64;
    
    //Hue
    if r == xp_max {
        h = (g - b) / xp_delta;
    }
    else if g == xp_max {
        h = 2_f64 + (b - r) / xp_delta;
    }
    else if b == xp_max {
        h = 4_f64 + (r - g) / xp_delta;
    }
    
    h *= 60_f64;
    
    if h < 0_f64 {
        h += 360_f64;
    }
    
    h /= 360_f64;
    
    (h, s, v)
}

pub fn hsv_to_rgb(hsv: (f64, f64, f64)) -> (u8, u8, u8) {
    let (h, s, v) = hsv;
    
    let r;
    let g;
    let b;
    
    //If saturation is 0, no chroma
    if hsv.1 == 0_f64 {
        r = v;
        g = v;
        b = v;
    }
    else {
        let h = h * 6_f64;
        let i = h.floor() as u32;
        
        let f = h - (i as f64);
        let p = v * (1_f64 - s);
        let q = v * (1_f64 - s * f);
        let t = v * (1_f64 - s * (1_f64 - f));
        
        match i {
            0 => {
                r = v;
                g = t;
                b = p;
            }
            1 => {
                r = q;
                g = v;
                b = p;
            }
            2 => {
                r = p;
                g = v;
                b = t;
            }
            3 => {
                r = p;
                g = q;
                b = v;
            }
            4 => {
                r = t;
                g = p;
                b = v;
            }
            _ => {
                r = v;
                g = p;
                b = q;
            }
        }
    }
    
    ((r * 255_f64).round() as u8, (g * 255_f64).round() as u8, (b * 255_f64).round() as u8)
}

pub fn apply_edges(image: &DynamicImage, edge_color: &Rgba<u8>) -> DynamicImage {
    const EDGE_MIN: f32 = 15_f32;
    const EDGE_MAX: f32 = 35_f32;
    
    //Get greyscale image and apply edge detection
    let greyscale = image.to_luma8();
    let edges = imageproc::edges::canny(&greyscale, EDGE_MIN, EDGE_MAX);
    
    //Make edges thicker
    let dilated = imageproc::morphology::dilate(&edges, imageproc::distance_transform::Norm::L2, 1);
    
    //Convert to rgba, with given foreground color for edges, and everything else transparent
    let edges_rgba: RgbaImage = imageproc::map::map_colors(&dilated, |pixel| {
        if pixel[0] > 0 {
            *edge_color
        }
        else {
            Rgba([0, 0, 0, 0])
        }
    });
    
    //Composite
    let mut composite = image.to_rgba8();
    
    for (i, j, pixel) in edges_rgba.enumerate_pixels() {
        let Rgba([er, eg, eb, ea]) = pixel;
        
        //There's an edge here
        if *ea > 0 {
            let Rgba([ir, ig, ib, ia]) = image.get_pixel(i, j);
            
            //Composite edge with image pixel by taking a weighted average
            let (r, g, b, a) = (
                ((0.4_f64 * (ir as f64 / 255_f64) + 0.6_f64 * (*er as f64 / 255_f64)) * 255_f64) as u8,
                ((0.4_f64 * (ig as f64 / 255_f64) + 0.6_f64 * (*eg as f64 / 255_f64)) * 255_f64) as u8,
                ((0.4_f64 * (ib as f64 / 255_f64) + 0.6_f64 * (*eb as f64 / 255_f64)) * 255_f64) as u8,
                ((0.4_f64 * (ia as f64 / 255_f64) + 0.6_f64 * (*ea as f64 / 255_f64)) * 255_f64) as u8
            );
            
            composite.put_pixel(i, j, Rgba([r, g, b, a]))
        }
    }
    
    DynamicImage::ImageRgba8(composite)
}