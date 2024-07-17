use std::collections::HashMap;

use colored::Colorize;
use imageproc::image::DynamicImage;
use imageproc::image::Rgba;
use libi2a::converter::SSIMConverter;

const ARG_PATH: &str = "--path";
const FONT_PATH: &str = "assets/CascadiaMono.ttf";
const TILE_SIZE_MIN: u32 = 1;
const TILE_SIZE_DFT: u32 = 8;
const SUBDIVIDE_MIN: u32 = 0;
const SUBDIVIDE_DFT: u32 = 2;

fn main() -> Result<(), String> {    
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        _ = std::env::set_current_dir(exe);
    }

    let args = utils::command_line::parse_args(std::env::args(), Some(String::from(ARG_PATH)))?;
    
    let path = match args.get(ARG_PATH) {
        Some(Some(p)) => Ok(p.clone()),
        _ => Err(String::from("Image path is required."))
    }?;
    
    let tile_size = u32::max(TILE_SIZE_MIN, match args.get("--tile-size") {
        Some(Some(p)) => p.parse::<u32>().map_err(|err| format!("Invalid tile size. {err}")),
        _ => Ok(TILE_SIZE_DFT)
    }?);
    
    let subdivide = u32::max(SUBDIVIDE_MIN, match args.get("--subdivide") {
        Some(Some(p)) => p.parse::<u32>().map_err(|err| format!("Invalid subdivision. {err}")),
        _ => Ok(SUBDIVIDE_DFT)
    }?);
    
    let invert = args.contains_key("--invert");
    
    let use_edges = !args.contains_key("--no-edges");
    
    let use_color = !args.contains_key("--no-color");
    
    //Read image from path
    let img = imageproc::image::io::Reader::open(path.clone())
        .map_err(|err| format!("Failed to open image {path}. {err}"))?
        .with_guessed_format()
        .map_err(|err| format!("Failed to read image {path}. {err}"))?
        .decode()
        .map_err(|err| format!("Failed to decode image {path}. {err}"))?;
    
    //Load reference font face
    let font_bytes = std::fs::read(FONT_PATH)
    .map_err(|err| format!("Failed to read font. {err}"))?;
    
    let font = ab_glyph::FontRef::try_from_slice(&font_bytes)
    .map_err(|err| format!("Failed to load font. {err}"))?;
    
    //Generate images for all glyphs (include whitespace only if not using color)
    let glyphs: Vec<char> = (0 as char..u8::MAX as char)
    .filter(|g| !g.is_control() && (!use_color || !g.is_whitespace()) && g.is_ascii())
    .collect();
    
    let mut glyph_images: HashMap<char, DynamicImage> = HashMap::new();
    
    let white = Rgba([255, 255, 255, 255]);
    let black = Rgba([0, 0, 0, 255]);
    
    for glyph in glyphs {
        let glyph_image = libi2a::glyphs::generate_glyph(glyph, 
            tile_size,
            &font,
            if invert { white } else { black },
            if invert { black } else { white });

        glyph_images.insert(glyph, glyph_image);
    }
        
    let to_convert = if use_edges {
        let edge_color = if invert { white } else { black };
        
        libi2a::converter::converter_utils::apply_edges(&img, &edge_color)
    }
    else {
        img
    };
    
    let converter = SSIMConverter::new(tile_size, subdivide, glyph_images);
    
    if use_color {
        _ = colored::control::set_virtual_terminal(true);
    }

    //Convert and print image glyphs
    for (glyph, maybe_color) in converter.convert(&to_convert) {
        if let Some(color) = maybe_color {
            if use_color {
                let r = ((color >> 16) & 255) as u8;
                let g = ((color >> 8) & 255) as u8;
                let b = (color & 255) as u8;
                
                let color = colored::Color::TrueColor { r, g, b };
                
                let colored = colored::ColoredString::from(glyph.to_string()).color(color);
                print!("{colored}");
            }
            else {
                print!("{glyph}");   
            }
        }
        else {
            print!("{glyph}");
        }
    }
    
    Ok(())
}