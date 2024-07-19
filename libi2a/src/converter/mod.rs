use imageproc::image::DynamicImage;
use imageproc::image::GenericImageView;

pub mod converter_utils;
pub mod ssim;
pub mod model;

pub trait Converter {
    fn best_match(&self, tile: &DynamicImage) -> char;

    fn get_tile_size(&self) -> u32;
}

impl dyn Converter {
    pub fn convert<'b, 'a: 'b>(&'a self, image: &'b DynamicImage) -> ConverterIter<'b> {
        ConverterIter {
            parent: self,
            image_iter: ImageTileIter {
                image,
                tile_size: self.get_tile_size(),
                offset: 0
            }
        }
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

pub struct ConverterIter<'a> {
    parent: &'a dyn Converter,
    image_iter: ImageTileIter<'a>
}

impl<'a> Iterator for ConverterIter<'a> {
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