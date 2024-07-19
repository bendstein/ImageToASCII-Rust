pub mod command_line {
    use std::collections::HashMap;

    const PREFIX: &str = "--";
    const PROGRAM: &str = "--program";

    pub fn parse_args(args: std::env::Args, implicit_first_key: Option<String>) -> Result<HashMap<String, Option<String>>, String> {
        let mut map = HashMap::new();

        let mut enumerator = args.enumerate().peekable();

        while let Some((ndx, key)) = enumerator.next() {
            //0th arg is the program path
            if ndx == 0 {
                map.insert(String::from(PROGRAM), Some(key));
                continue;
            }

            //If first argument has implicit key, use this as value
            if ndx == 1 && !key.starts_with(PREFIX) {
                if let Some(ref first_key) = implicit_first_key {
                    map.insert(first_key.clone(), Some(key));
                    continue;
                }
            }

            //Check if valid arg
            if !key.starts_with(PREFIX) {
                return Err(format!("Invalid argument {key}"));
            }

            //Try get value
            if let Some((_, value)) = enumerator.peek() {
                //Don't use next key as value
                if value.starts_with(PREFIX) {
                    map.insert(key, None);
                }
                else {
                    map.insert(key, Some(value.clone()));
                    _ = enumerator.next();
                }
            }
            else {
                map.insert(key, None);
            }
        }

        Ok(map)
    }
}

pub mod glyph {
    pub fn get_glyphs(include_ws: bool, max: u8) -> Vec<char> {
        let mut glyphs = Vec::new();

        for i in 0..max {
            let c = i as char;

            if (c.is_whitespace() && !include_ws) || c.is_control() {
                continue;
            }

            glyphs.push(c);
        }

        glyphs
    }
}

pub mod data {
    use std::io::{BufWriter, Write};

    type Data = (Vec<f32>, Vec<f32>);

    pub fn read_training_data(path: &str) -> Result<Vec<Data>, String> {
        Ok(std::fs::read_to_string(path)
            .map_err(|err| format!("Failed to read training data. {err}"))?
            .lines()
            .flat_map(|line| {
                let parts: Vec<&str> = line.split(';').collect();

                if parts.len() != 2 {
                    return Err(format!("Invalid training data line {line}"));
                }

                let features: Vec<f32> = parts[0].split(',')
                    .map(|p| p.parse::<f32>().unwrap_or(0_f32))
                    .collect();

                let outputs: Vec<f32> = parts[1].split(',')
                    .map(|p| p.parse::<f32>().unwrap_or(0_f32))
                    .collect();

                Ok((features, outputs))
            })
            .collect())
    }

    pub fn write_training_data(path: &str, data: impl Iterator<Item = Data>) -> Result<(), String> {
        //Format training data
        let data = data.map(|(features, outputs)| {
            let features: String = features.iter()
                .map(|f| f.to_string())
                .collect::<Vec<String>>()
                .join(",");

            let outputs: String = outputs.iter()
                .map(|o| o.to_string())
                .collect::<Vec<String>>()
                .join(",");

            format!("{features};{outputs}")
        });

        //Create or open file in append mode
        let file = if std::path::Path::new(path).exists() {
            std::fs::OpenOptions::new()
                .append(true)
                .open(path)
                .map_err(|err| format!("Failed to open training data. {err}"))?
        }
        else {
            //If directory doesn't exist, create it
            let maybe_dir = std::path::Path::new(path).parent();

            if let Some(dir) = maybe_dir {
                std::fs::create_dir_all(dir)
                    .map_err(|err| format!("Failed to create training data directory. {err}"))?;
            }

            //Create file
            std::fs::File::create(path)
                .map_err(|err| format!("Failed to create training data. {err}"))?
        };
        
        //Write data to file
        let mut writer = BufWriter::new(file);

        for line in data {
            writer.write_all(line.as_bytes())
                .map_err(|err| format!("Failed to write training data. {err}"))?;
            writer.write(b"\n")
                .map_err(|err| format!("Failed to write training data. {err}"))?;
        }

        Ok(())
    }
}