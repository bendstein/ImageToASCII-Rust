use std::io::BufRead;

use libi2a::converter::model::{Model, ModelConverter, ModelInitParams, ModelTrainingExample, ModelTrainingParams};

const CFG_PATH: &str = "config.toml";

fn main() -> Result<(), String> {    
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        _ = std::env::set_current_dir(exe);
    }

    let config = utils::config::read_config(CFG_PATH)?;

    let args = utils::command_line::parse_args(std::env::args(), None)?;
    
    let path = config.general.model.clone();

    let source = match args.get("--source") {
        Some(Some(p)) => Ok(p.clone()),
        _ => Err(String::from("Source path is required."))
    }?;

    //Create model directory if not exists
    let maybe_dir = std::path::Path::new(&path).parent();

    if let Some(dir) = maybe_dir {
        if !dir.exists() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("Failed to create model directory: {}", e))?;
        }
    }

    //Load model from path if it exists, otherwise init
    let model_initial = if std::path::Path::new(&path).exists() {
        Model::load_from_file(&path)?
    } else {
        //Get list of glyphs
        let glyphs: Vec<char> = config.general.glyphs;
        
        Model::init_from_params(ModelInitParams {
            glyphs,
            feature_count: config.ssim.tile_size * config.ssim.tile_size,
            hidden_layer_count: config.training.hidden_layers,
            hidden_layer_neuron_count: config.training.hidden_neurons,
            alpha: config.training.alpha
        })
    };

    fn log(msg: String, is_error: bool) {
        let full_message = format!("{} {} {msg}",
            " ",
            if is_error { "ERROR | " } else { "INFO  | " });

        if is_error {
            eprintln!("{full_message}");
        } else {
            println!("{full_message}");
        }
    }

    let save_model = |model: &Model, epoch: u32| -> Result<(), String> {        
        //Save model checkpoint
        if epoch > 0 && epoch % config.training.checkpoint_period == 0 {
            log(String::from("Saving model checkpoint."), false);

            model.save_to_file(format!("{path}.{epoch}.checkpoint").as_str())
                .map_err(|e| format!("Failed to save model checkpoint: {e}"))?;
        }

        //Save model
        log(String::from("Saving model."), false);

        model.save_to_file(&path)
            .map_err(|e| format!("Failed to save model: {e}"))
    };

    let reader = std::io::BufReader::new(std::fs::File::open(source)
        .map_err(|err| format!("Failed to open source file. {err}"))?);

    let mut training_data = reader.lines()
        .flat_map(|line| {
            if let Err(e) = line {
                return Err(format!("Failed to read line. {e}"));
            }

            let line = line.unwrap();

            let parts: Vec<&str> = line.split(';').collect();

            if parts.len() != 2 {
                return Err(format!("Invalid training data line {line}"));
            }

            let features: Vec<f64> = parts[0].split(',')
                .map(|p| p.parse::<f64>().unwrap_or(0_f64))
                .collect();

            let outputs: Vec<f64> = parts[1].split(',')
                .map(|p| p.parse::<f64>().unwrap_or(0_f64))
                .collect();

            if features.len() != model_initial.feature_count() {
                return Err(format!("Invalid feature count in line {line}"));
            }

            if outputs.len() != model_initial.output_count() {
                return Err(format!("Invalid output count in line {line}"));
            }

            Ok((features, outputs))
        })
        .map(|(features, outputs)| ModelTrainingExample::new(features.clone(), outputs.clone()));

    let training_params = ModelTrainingParams {
        learning_rate: config.training.learning_rate,
        batch_size: config.training.batch_size,
        lambda: config.training.l2,
        adam: (config.training.adam_beta1, config.training.adam_beta2)
    };

    //Train model
    let trained_model = ModelConverter::train_model(&model_initial, &training_params, &mut training_data, &log, &save_model)?;

    //Save trained model
    save_model(&trained_model, 0)?;

    Ok(())
}

