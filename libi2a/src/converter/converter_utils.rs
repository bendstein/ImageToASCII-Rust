use imageproc::image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

pub fn get_intensity(pixel: Rgba<u8>) -> f32 {
    //Magic numbers
    const LINEAR_POWER: f32 = 2.2_f32;
    const CR: f32 = 0.2126_f32;
    const CG: f32 = 0.7152_f32;
    const CB: f32 = 0.0722_f32;
    const CH: f32 = 1.0_f32;
    const CS: f32 = 1.0_f32;
    const CV: f32 = 1.2_f32;
    
    //Convert to HSV
    let (h, s, v) = rgb_to_hsv((pixel.0[0], pixel.0[1], pixel.0[2]));
    
    //Scale HSV
    let (sh, ss, sv) = (CH * h, CS * s, CV * v);
    
    //Convert back to rgb
    let (r, g, b) = hsv_to_rgb((sh, ss, sv));
    
    //'Linearize' channels
    let r_lin = (r as f32 / 255_f32).powf(LINEAR_POWER);
    let g_lin = (g as f32 / 255_f32).powf(LINEAR_POWER);
    let b_lin = (b as f32 / 255_f32).powf(LINEAR_POWER);
    
    //Intensity as linear combination
    CR * r_lin + CG * g_lin + CB * b_lin
}

pub fn get_color(rgb: (u8, u8, u8)) -> u32 {
    ((rgb.0 as u32) << 16) + ((rgb.1 as u32) << 8) + rgb.2 as u32
}

pub fn gaussian(w: usize) -> Vec<f32> {
    let mut g = vec![0_f32; w * w];
    
    let stddev = (w as f32).sqrt().log2() / 1.5_f32;
    
    let center = (w as f32 - 1_f32) / 2_f32;
    
    let mut sum = 0_f32;
    
    //Generate circular gaussian matrix
    for i in 0..w {
        for j in 0..w {
            let value = 1_f32 / (2_f32 * std::f32::consts::PI * stddev.powf(2_f32))
            * std::f32::consts::E.powf(((i as f32 - center).powf(2_f32) + (j as f32 - center).powf(2_f32))
            / (-2_f32 * stddev.powf(2_f32)));
            
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

pub fn rgb_to_hsv(rgb: (u8, u8, u8)) -> (f32, f32, f32) {
    let (r, g, b) = (rgb.0 as f32 / 255_f32, rgb.1 as f32 / 255_f32, rgb.2 as f32 / 255_f32);
    let xp_min = f32::min(r, f32::min(g, b));
    let xp_max = f32::max(r, f32::max(g, b));
    let xp_delta = xp_max - xp_min;
    
    //Value
    let v = xp_max;
    
    //Saturation
    let s = if xp_max == 0_f32 {
        0_f32
    }
    else {
        xp_delta / xp_max
    };
    
    //No chroma
    if xp_delta == 0_f32 {
        return (0_f32, s, v);
    }
    
    //Set chroma data
    let mut h: f32 = 0_f32;
    
    //Hue
    if r == xp_max {
        h = (g - b) / xp_delta;
    }
    else if g == xp_max {
        h = 2_f32 + (b - r) / xp_delta;
    }
    else if b == xp_max {
        h = 4_f32 + (r - g) / xp_delta;
    }
    
    h *= 60_f32;
    
    if h < 0_f32 {
        h += 360_f32;
    }
    
    h /= 360_f32;
    
    (h, s, v)
}

pub fn hsv_to_rgb(hsv: (f32, f32, f32)) -> (u8, u8, u8) {
    let (h, s, v) = hsv;
    
    let r;
    let g;
    let b;
    
    //If saturation is 0, no chroma
    if hsv.1 == 0_f32 {
        r = v;
        g = v;
        b = v;
    }
    else {
        let h = h * 6_f32;
        let i = h.floor() as u32;
        
        let f = h - (i as f32);
        let p = v * (1_f32 - s);
        let q = v * (1_f32 - s * f);
        let t = v * (1_f32 - s * (1_f32 - f));
        
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
    
    ((r * 255_f32).round() as u8, (g * 255_f32).round() as u8, (b * 255_f32).round() as u8)
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
                ((0.4_f32 * (ir as f32 / 255_f32) + 0.6_f32 * (*er as f32 / 255_f32)) * 255_f32) as u8,
                ((0.4_f32 * (ig as f32 / 255_f32) + 0.6_f32 * (*eg as f32 / 255_f32)) * 255_f32) as u8,
                ((0.4_f32 * (ib as f32 / 255_f32) + 0.6_f32 * (*eb as f32 / 255_f32)) * 255_f32) as u8,
                ((0.4_f32 * (ia as f32 / 255_f32) + 0.6_f32 * (*ea as f32 / 255_f32)) * 255_f32) as u8
            );
            
            composite.put_pixel(i, j, Rgba([r, g, b, a]))
        }
    }
    
    DynamicImage::ImageRgba8(composite)
}