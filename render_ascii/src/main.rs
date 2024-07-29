use std::collections::HashMap;

use colored::Colorize;
use imageproc::image::DynamicImage;
use imageproc::image::Rgba;
use libi2a::converter::model::ModelConverter;
use libi2a::converter::ssim::SSIMConverter;
use libi2a::converter::Converter;

const ARG_PATH: &str = "--path";
const CFG_PATH: &str = "config.toml";
const METHOD_SSIM: &str = "ssim";
const METHOD_MODEL: &str = "model";

fn main() -> Result<(), String> {    
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        _ = std::env::set_current_dir(exe);
    }

    let args = utils::command_line::parse_args(std::env::args(), Some(String::from(ARG_PATH)))?;
    
    let config = utils::config::read_config(CFG_PATH)?;

    let path = match args.get(ARG_PATH) {
        Some(Some(p)) => Ok(p.clone()),
        _ => Err(String::from("Image path is required."))
    }?;

    let invert = args.contains_key("--invert");
    
    let use_edges = !args.contains_key("--no-edges");
    
    let use_color = !args.contains_key("--no-color");
    
    let white = Rgba([255, 255, 255, 255]);
    let black = Rgba([0, 0, 0, 255]);

    //Read image from path
    let img = imageproc::image::io::Reader::open(path.clone())
        .map_err(|err| format!("Failed to open image {path}. {err}"))?
        .with_guessed_format()
        .map_err(|err| format!("Failed to read image {path}. {err}"))?
        .decode()
        .map_err(|err| format!("Failed to decode image {path}. {err}"))?;
    
    //Apply edges if enabled
    let to_convert = if use_edges {
        let edge_color = if invert { white } else { black };
        
        libi2a::converter::converter_utils::apply_edges(&img, &edge_color)
    }
    else {
        img
    };

    let converter: Box<dyn Converter> = match config.general.render_method.as_str(){
        METHOD_MODEL => {
            let model = libi2a::converter::model::Model::load_from_file(&config.general.model)?;

            Ok(Box::new(ModelConverter::new(model)) as Box<dyn Converter>)
        },
        METHOD_SSIM => {
            //Load reference font face
            let font_bytes = std::fs::read(config.ssim.font)
                .map_err(|err| format!("Failed to read font. {err}"))?;

            let font = ab_glyph::FontRef::try_from_slice(&font_bytes)
            .map_err(|err| format!("Failed to load font. {err}"))?;

            //Generate images for all glyphs (include whitespace only if not using color)
            let glyphs: Vec<char> = config.general.glyphs
                .iter()
                .filter(|c| !use_color || !c.is_whitespace())
                .copied()
                .collect();

            let mut glyph_images: HashMap<char, DynamicImage> = HashMap::new();

            for glyph in &glyphs {
                let glyph_image = libi2a::glyphs::generate_glyph(*glyph, 
                    config.ssim.tile_size,
                    &font,
                    if invert { white } else { black },
                    if invert { black } else { white });

                glyph_images.insert(*glyph, glyph_image);
            }

            Ok(Box::new(SSIMConverter::new(config.ssim.tile_size, config.ssim.subdivide, glyphs, glyph_images)) as Box<dyn Converter>)
        },
        _ => Err(format!("Invalid method {}", config.general.render_method))
    }?;
    
    if use_color {
        _ = colored::control::set_virtual_terminal(true);
    }

    //Convert and print image glyphs
    for (glyph, maybe_color) in (*converter).convert(&to_convert) {
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