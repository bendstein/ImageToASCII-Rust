use std::collections::HashMap;

use imageproc::image::DynamicImage;
use imageproc::image::GenericImageView;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

pub mod converter_utils {
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
}

pub struct SSIMConverter {
    tile_size: u32,
    subdivide: u32,
    glyph_images: HashMap<char, DynamicImage>
}

impl SSIMConverter {
    pub fn new(tile_size: u32, subdivide: u32, glyph_images: HashMap<char, DynamicImage>) -> Self {
        Self {
            tile_size,
            subdivide,
            glyph_images
        }
    }

    pub fn convert<'b, 'a: 'b>(&'a self, image: &'b DynamicImage) -> SSIMConverterIter<'b> {
        SSIMConverterIter {
            parent: self,
            image_iter: ImageTileIter {
                image,
                tile_size: self.tile_size,
                offset: 0
            }
        }
    }

    fn calculate_ssim(&self, glyph_image: &DynamicImage, tile: &DynamicImage) -> f32 {
        //Get intensities of pixels in each tile
        let glyph_intensities: Vec<f32> = glyph_image.pixels().map(|p| converter_utils::get_intensity(p.2)).collect();
        let tile_intensities: Vec<f32> = tile.pixels().map(|p| converter_utils::get_intensity(p.2)).collect();

        let gaussian = converter_utils::gaussian(usize::max(glyph_intensities.len(), tile_intensities.len()));

        //Get gaussian-weighted mean intensity of each image (luminance)
        let glyph_mean: f32 = glyph_intensities.iter()
            .enumerate()
            .map(|(ndx, i)| gaussian[ndx] * i)
            .sum();

        let tile_mean: f32 = tile_intensities.iter()
            .enumerate()
            .map(|(ndx, i)| gaussian[ndx] * i)
            .sum();

        //Get gaussian-weighted standard deviation of the intensities of each image (contrast)
        let glyph_stddev: f32 = f32::sqrt(glyph_intensities.iter()
            .enumerate()
            .map(|(ndx, i)| gaussian[ndx] * (i - glyph_mean).powf(2_f32))
            .sum());

        let tile_stddev: f32 = f32::sqrt(tile_intensities.iter()
            .enumerate()
            .map(|(ndx, i)| gaussian[ndx] * (i - tile_mean).powf(2_f32))
            .sum());

        //Get gaussian-weighted covariance (structure) of the intensities of each tile
        let mut covar = 0_f32;

        for i in 0..usize::min(glyph_intensities.len(), tile_intensities.len()) {
            covar += gaussian[i] * (glyph_intensities[i] - glyph_mean) * (tile_intensities[i] - tile_mean);
        }

        //Constants to prevent instability around 0
        let k1 = (0.001_f32 * 255_f32).powf(2_f32);
        let k2 = (0.003_f32 * 255_f32).powf(2_f32);

        //SSIM component weights
        const W1: f32 = 1.1_f32;
        const W2: f32 = 0.4_f32;
        const W3: f32 = 1.75_f32;

        ((W1 * 2_f32 * glyph_mean * tile_mean + k1) * (W3 * 2_f32 * covar + k2))
        / ((W1 * (glyph_mean.powf(2_f32) + tile_mean.powf(2_f32)) + k1) * (W2 * (glyph_stddev.powf(2_f32) + tile_stddev.powf(2_f32)) + k2))
    }

    fn calculate(&self, glyph: char, tile: &DynamicImage) -> f32 {
        let glyph_image = self.glyph_images.get(&glyph)
            .unwrap_or_else(|| panic!("Glyph {glyph} has no image."));

        //Subdivide each image and compare them
        if self.subdivide > 0 {
            let (gw, gh) = glyph_image.dimensions();
            let (tw, th) = tile.dimensions();
    
            let gmd = u32::max(gw, gh);
            let tmd = u32::max(tw, th);
    
            //Get tile dims
            let glyph_window_size = f32::ceil(gmd as f32 / 2_f32.powf(self.subdivide as f32)) as u32;
            let tile_window_size = f32::ceil(tmd as f32 / 2_f32.powf(self.subdivide as f32)) as u32;
    
            let glyph_windows_per_row = f32::ceil(gmd as f32 / glyph_window_size as f32) as u32;
            
            let gaussian = converter_utils::gaussian((glyph_windows_per_row * glyph_windows_per_row) as usize);

            //Iterate over tiles
            let mut glyph_iter = ImageTileIter {
                image: glyph_image,
                tile_size: glyph_window_size,
                offset: 0
            };
    
            let mut tile_iter = ImageTileIter {
                image: tile,
                tile_size: tile_window_size,
                offset: 0
            };
    
            //Aggregate SSIMs to average
            let mut ssim_sum = 0_f32;

            let mut current_glyph_window: Option<(u32, u32, DynamicImage)> = glyph_iter.next();
            let mut current_tile_window: Option<(u32, u32, DynamicImage)> = tile_iter.next();
    
            loop {    
                //Done            
                if current_glyph_window.is_none() && current_tile_window.is_none() {
                    break;
                }

                //Handle mismatched lengths as having 0 similarity
                if current_glyph_window.is_none() || current_tile_window.is_none() {
                    if current_glyph_window.is_some() {
                        current_glyph_window = glyph_iter.next();
                    }
    
                    if current_tile_window.is_some() {
                        current_tile_window = tile_iter.next();
                    }
    
                    continue;
                }

                let (glyph_row, glyph_col, current_glyph) = current_glyph_window.as_ref().unwrap();
                let (tile_row, tile_col, current_tile) = current_tile_window.as_ref().unwrap();

                //Tile is ahead of glyph, advance to catch up (consider 0 similarity for mismatched dims)
                if glyph_row < tile_row || glyph_col < tile_col {
                    current_glyph_window = glyph_iter.next();
                    continue;
                }
                //Glyph is ahead of tile, advance to catch up (consider 0 similarity for mismatched dims)
                else if tile_row < glyph_row || tile_col < glyph_col {
                    current_tile_window = tile_iter.next();
                    continue;
                }

                //Tile indices match; compare and take weighted avg
                let ssim = self.calculate_ssim(current_glyph, current_tile);
                ssim_sum += gaussian[(glyph_col + (glyph_row * glyph_windows_per_row)) as usize] * ssim;

                //Advance iterators
                current_glyph_window = glyph_iter.next();
                current_tile_window = tile_iter.next();
            }    

            ssim_sum
        }
        //Not subdividing; compare images as is
        else {
            self.calculate_ssim(glyph_image, tile)
        }
    }

    fn best_match(&self, tile: &DynamicImage) -> char {
        let glyphs: Vec<char> = self.glyph_images.keys().copied().collect();

        glyphs.par_iter()
            .map(|glyph| (glyph, self.calculate(*glyph, tile)))
            .reduce_with(|a, b| if a.1 > b.1 || (a.1 == b.1 && a.0 < b.0) { a } else { b })
            .map_or(' ', |(glyph, _)| *glyph)
    }
}

struct ImageTileIter<'a> {
    image: &'a DynamicImage,
    tile_size: u32,
    offset: u32
}

impl<'a> Iterator for ImageTileIter<'a> {
    type Item = (u32, u32, DynamicImage);

    fn next(&mut self) -> Option<Self::Item> {
        //Calculate top left corner of tile
        let (w, h) = imageproc::image::GenericImageView::dimensions(self.image);

        let windows_per_row = f32::ceil(w as f32 / self.tile_size as f32) as u32;

        let row = self.offset / windows_per_row;
        let col = self.offset % windows_per_row;

        let x = col * self.tile_size;
        let y = row * self.tile_size;

        //If past bottom of image, stop
        if y >= h {
            return None
        }

        //Increment offset
        self.offset += 1;

        //Get tile_size x tile_size window, but don't go past bounds of image
        let tile_width = u32::min(w - x, self.tile_size);
        let tile_height = u32::min(h - y, self.tile_size);

        let window = DynamicImage::ImageRgba8(self.image.view(x, y, tile_width, tile_height).to_image());

        Some((row, col, window))
    }
}

pub struct SSIMConverterIter<'a> {
    parent: &'a SSIMConverter,
    image_iter: ImageTileIter<'a>
}

impl<'a> Iterator for SSIMConverterIter<'a> {
    type Item = (String, Option<u32>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((row, col, next)) = self.image_iter.next() {
            //First column of new row, prepend newline to result
            let newline = row > 0 && col == 0;

            //Get average color
            let count = next.pixels().count();
            let mut h = 0_f32;
            let mut s = 0_f32;
            let mut v = 0_f32;

            //Sum up HSV components
            for (_, _, pixel) in next.pixels() {
                let hsv = converter_utils::rgb_to_hsv((pixel.0[0], pixel.0[1], pixel.0[2]));
                h += hsv.0;
                s += hsv.1;
                v += hsv.2;
            }

            //Take average of each and convert back to RGB
            let rgb = converter_utils::hsv_to_rgb((
                h / count as f32,
                s / count as f32,
                v / count as f32
            ));

            //Convert to u32
            let color = converter_utils::get_color((rgb.0, rgb.1, rgb.2));

            //Yield best matching glyph
            Some((format!("{}{}", if newline { "\r\n" } else { "" }, self.parent.best_match(&next)), Some(color)))
        }
        else {
            None
        }
    }
}