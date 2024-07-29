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

pub mod config {
    use serde::Deserialize;
    
    #[derive(Deserialize)]
    pub struct Config {
        pub general: GeneralConfig,
        pub ssim: SSIMConfig,
        pub training: TrainingConfig,
    }

    #[derive(Deserialize)]
    pub struct GeneralConfig {
        pub render_method: String,
        pub model: String,
        pub glyphs: Vec<char>
    }

    #[derive(Deserialize)]
    pub struct SSIMConfig {
        pub subdivide: u32,
        pub font: String,
        pub tile_size: u32,
    }

    #[derive(Deserialize)]
    pub struct TrainingConfig {
        pub hidden_layers: u32,
        pub hidden_neurons: u32,
        pub alpha: f32,
        pub l2: f32,
        pub learning_rate: f32,
        pub adam_beta1: f32,
        pub adam_beta2: f32,
        pub batch_size: u32
    }

    pub fn read_config(path: &str) -> Result<Config, String> {
        let config = std::fs::read_to_string(path)
            .map_err(|err| format!("Failed to read config. {err}"))?;

        toml::from_str(&config)
            .map_err(|err| format!("Failed to parse config. {err}"))
    }
}

pub mod vmath {
    //Implement a function that performs checked addition on 2 DMatrix
    pub fn checked_add(a: &nalgebra::DMatrix<f32>, b: &nalgebra::DMatrix<f32>) -> Result<nalgebra::DMatrix<f32>, String> {
        if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
            return Err(format!("Matrix dimensions do not match. ({}x{} vs {}x{})", a.nrows(), a.ncols(), b.nrows(), b.ncols()));
        }

        Ok(a + b)
    }

    //Implement a function that performs checked subtraction on 2 DMatrix
    pub fn checked_sub(a: &nalgebra::DMatrix<f32>, b: &nalgebra::DMatrix<f32>) -> Result<nalgebra::DMatrix<f32>, String> {
        if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
            return Err(format!("Matrix dimensions do not match. ({}x{} vs {}x{})", a.nrows(), a.ncols(), b.nrows(), b.ncols()));
        }

        Ok(a - b)
    }

    //Implement a function that performs checked multiplication on 2 DMatrix
    pub fn checked_mul(a: &nalgebra::DMatrix<f32>, b: &nalgebra::DMatrix<f32>) -> Result<nalgebra::DMatrix<f32>, String> {
        if a.ncols() != b.nrows() {
            return Err(format!("Matrix dimensions do not match. ({}x{} vs {}x{})", a.nrows(), a.ncols(), b.nrows(), b.ncols()));
        }

        Ok(a * b)
    }

    //Implement a function that performs checked multiplication on DMatrix and DVector
    pub fn checked_mul_mv(a: &nalgebra::DMatrix<f32>, b: &nalgebra::DVector<f32>) -> Result<nalgebra::DVector<f32>, String> {
        if a.ncols() != b.len() {
            return Err(format!("Matrix and vector dimensions do not match. ({}x{} vs {}x{})", a.nrows(), a.ncols(), b.len(), 1));
        }

        Ok(a * b)
    }

    //Implement a function that performs checked add on 2 DVector
    pub fn checked_add_v(a: &nalgebra::DVector<f32>, b: &nalgebra::DVector<f32>) -> Result<nalgebra::DVector<f32>, String> {
        if a.len() != b.len() {
            return Err(format!("Vector dimensions do not match. ({}x{} vs {}x{})", a.len(), 1, b.len(), 1));
        }

        Ok(a + b)
    }

    //Implement a function that performs checked mul on 2 DVector
    pub fn checked_component_mul_v(a: &nalgebra::DVector<f32>, b: &nalgebra::DVector<f32>) -> Result<nalgebra::DVector<f32>, String> {
        if a.len() != b.len() {
            return Err(format!("Vector dimensions do not match. ({}x{} vs {}x{})", a.len(), 1, b.len(), 1));
        }

        Ok(a.component_mul(b))
    }

    //Implement a function that performs checked sub on 2 DVector
    pub fn checked_sub_v(a: &nalgebra::DVector<f32>, b: &nalgebra::DVector<f32>) -> Result<nalgebra::DVector<f32>, String> {
        if a.len() != b.len() {
            return Err(format!("Vector dimensions do not match. ({}x{} vs {}x{})", a.len(), 1, b.len(), 1));
        }

        Ok(a - b)
    }

    //Implement a function that performs checked dot on 2 DVector
    pub fn checked_dot_v(a: &nalgebra::DVector<f32>, b: &nalgebra::DVector<f32>) -> Result<f32, String> {
        if a.len() != b.len() {
            return Err(format!("Vector dimensions do not match. ({}x{} vs {}x{})", a.len(), 1, b.len(), 1));
        }

        Ok(a.dot(b))
    }

    //Implement a function that performs checked multiplication on DVector and DMatrix
    pub fn checked_mul_vm(a: &nalgebra::DVector<f32>, b: &nalgebra::DMatrix<f32>) -> Result<nalgebra::DMatrix<f32>, String> {
        if b.nrows() != 1 {
            return Err(format!("Vector and matrix dimensions do not match. ({}x{} vs {}x{})", 1, a.len(), b.nrows(), b.ncols()));
        }

        Ok(a * b)
    }

    //Implement a function that performs checked component division on 2 DMatrix
    pub fn checked_component_div(a: &nalgebra::DMatrix<f32>, b: &nalgebra::DMatrix<f32>) -> Result<nalgebra::DMatrix<f32>, String> {
        if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
            return Err(format!("Matrix dimensions do not match. ({}x{} vs {}x{})", a.nrows(), a.ncols(), b.nrows(), b.ncols()));
        }

        Ok(a.component_div(b))
    }

    //Implement a function that performs checked component division on 2 DVector
    pub fn checked_component_div_v(a: &nalgebra::DVector<f32>, b: &nalgebra::DVector<f32>) -> Result<nalgebra::DVector<f32>, String> {
        if a.len() != b.len() {
            return Err(format!("Vector dimensions do not match. ({}x{} vs {}x{})", a.len(), 1, b.len(), 1));
        }

        Ok(a.component_div(b))
    }

    pub fn median(a: &[f32]) -> f32 {
        let mut data = a.to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = data.len() / 2;

        if data.len() % 2 == 0 {
            (data[mid - 1] + data[mid]) / 2_f32
        }
        else {
            data[mid]
        }
    }
}