use std::collections::HashMap;

use imageproc::image::DynamicImage;
use imageproc::image::GenericImageView;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use super::{converter_utils, Converter, ImageTileIter};

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

    pub fn preprocess<'a>(&'a self, image: &'a DynamicImage) -> SSIMPreprocessIterator<'a> {
        SSIMPreprocessIterator {
            parent: self,
            image_iter: ImageTileIter {
                image,
                tile_size: self.tile_size,
                offset: 0
            }
        }
    }
}

impl Converter for SSIMConverter {
    fn best_match(&self, tile: &DynamicImage) -> char {
        let glyphs: Vec<char> = self.glyph_images.keys().copied().collect();

        glyphs.par_iter()
            .map(|glyph| (glyph, self.calculate(*glyph, tile)))
            .reduce_with(|a, b| if a.1 > b.1 || (a.1 == b.1 && a.0 < b.0) { a } else { b })
            .map_or(' ', |(glyph, _)| *glyph)
    }

    fn get_tile_size(&self) -> u32 {
        self.tile_size
    }
}

pub struct SSIMPreprocessIterator<'a> {
    parent: &'a SSIMConverter,
    image_iter: ImageTileIter<'a>
}

impl<'a> Iterator for SSIMPreprocessIterator<'a> {
    type Item = (String, Option<u32>, Vec<f32>, Vec<f32>);

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

            //Calculate SSIM for all glyphs
            let glyphs: Vec<char> = self.parent.glyph_images.keys().copied().collect();

            let mut ssims: Vec<(char, f32)> = glyphs.par_iter()
                .map(|glyph| (*glyph, self.parent.calculate(*glyph, &next)))
                .collect();

            //Make sure sorted by character value
            ssims.sort_by(|a, b| a.0.cmp(&b.0));

            //Get best match
            let best_match = ssims.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(&(' ', 0_f32));

            //Yield best matching glyph, intensities, and all SSIMs
            Some((format!("{}{}", if newline { "\r\n" } else { "" }, best_match.0), Some(color),
                next.pixels().map(|(_, _, pixel)| converter_utils::get_intensity(pixel)).collect(),
                ssims.iter().map(|(_, ssim)| *ssim).collect()))
        }
        else {
            None
        }
    }
}