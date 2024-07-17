use ab_glyph::{Font, ScaleFont};
use ab_glyph::{FontRef, PxScale};
use imageproc::image::DynamicImage;
use imageproc::image::ImageBuffer;
use imageproc::image::Rgba;

pub fn generate_glyph(glyph: char, tile_size: u32, font_face: &FontRef, foreground: Rgba<u8>, background: Rgba<u8>) -> DynamicImage {
    //Create blank canvas for text
    let mut img = ImageBuffer::from_pixel(tile_size, tile_size, background);

    //Write text
    if !glyph.is_whitespace() {
        //Scale font to be 2/3s of the tile
        let scale = PxScale::from(tile_size as f32 * 2_f32 / 3_f32);
        let scaled_font = font_face.as_scaled(scale);

        //Get the bounds of the glyph
        let mut font_glyph = scaled_font.glyph_id(glyph).with_scale(scale);
        font_glyph.position = ab_glyph::point(0_f32, 0_f32);

        let glyph_outline = scaled_font.outline_glyph(font_glyph).unwrap();
        let bounds = glyph_outline.px_bounds();
        let w = bounds.width();
        let h = bounds.height();

        let x = (tile_size as f32 - w) / 2_f32 - bounds.min.x;
        let y = (tile_size as f32 - h) / 2_f32 - bounds.max.y;

        imageproc::drawing::draw_text_mut(&mut img, 
            foreground, 
            x.round() as i32,
            y.round() as i32, 
            scale, 
            font_face, 
            &glyph.to_string());
    }

    DynamicImage::ImageRgba8(img)
}

