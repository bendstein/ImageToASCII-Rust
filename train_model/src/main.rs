use libi2a::converter::model::{Model, ModelConverter, ModelInitParams, ModelTrainingExample, ModelTrainingParams};

const ARG_PATH: &str = "--path";
const TILE_SIZE_MIN: u32 = 1;
const TILE_SIZE_DFT: u32 = 8;
const HIDDEN_LAYER_DFT: u32 = 1;
const HIDDEN_LAYER_MIN: u32 = 0;
const HIDDEN_NODES_DFT: u32 = 16;
const HIDDEN_NODES_MIN: u32 = 1;
const ALPHA_DFT: f32 = 0_f32;
const ALPHA_MIN: f32 = 0_f32;
const LEARNING_RATE_MIN: f32 = 0e-8_f32;
const LEARNING_RATE_DFT: f32 = 0.01_f32;
const BATCH_SIZE_DFT: u32 = 1;
const BATCH_SIZE_MIN: u32 = 1;
const LAMBDA_DFT: f32 = 0_f32;
const LAMBDA_MIN: f32 = 0_f32;
const ADAM_BETA1_DFT: f32 = 0.99_f32;
const ADAM_BETA1_MIN: f32 = 0e-8_f32;
const ADAM_BETA2_DFT: f32 = 0.999_f32;
const ADAM_BETA2_MIN: f32 = 0e-8_f32;

fn main() -> Result<(), String> {    
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        _ = std::env::set_current_dir(exe);
    }

    let args = utils::command_line::parse_args(std::env::args(), Some(String::from(ARG_PATH)))?;
    
    let path = match args.get(ARG_PATH) {
        Some(Some(p)) => Ok(p.clone()),
        _ => Err(String::from("Model path is required."))
    }?;

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
        let glyphs: Vec<char> = utils::glyph::get_glyphs(true, 255_u8)
            .iter()
            .filter(|c| c.is_ascii())
            .copied()
            .collect();
        
        //Read params from args or use defaults
        let tile_size = u32::max(TILE_SIZE_MIN, match args.get("--tile-size") {
            Some(Some(p)) => p.parse::<u32>().map_err(|err| format!("Invalid tile size. {err}")),
            _ => Ok(TILE_SIZE_DFT)
        }?);

        let hidden_layers = u32::max(HIDDEN_LAYER_MIN, match args.get("--hidden-layers") {
            Some(Some(p)) => p.parse::<u32>().map_err(|err| format!("Invalid number of hidden layers. {err}")),
            _ => Ok(HIDDEN_LAYER_DFT)
        }?);

        let hidden_neurons = u32::max(HIDDEN_NODES_MIN, match args.get("--hidden-neurons") {
            Some(Some(p)) => p.parse::<u32>().map_err(|err| format!("Invalid number of hidden neurons. {err}")),
            _ => Ok(HIDDEN_NODES_DFT)
        }?);

        let alpha = f32::max(ALPHA_MIN, match args.get("--alpha") {
            Some(Some(p)) => p.parse::<f32>().map_err(|err| format!("Invalid leaky ReLU alpha. {err}")),
            _ => Ok(ALPHA_DFT)
        }?);

        Model::init_from_params(ModelInitParams {
            glyphs,
            feature_count: tile_size * tile_size,
            hidden_layer_count: hidden_layers,
            hidden_layer_neuron_count: hidden_neurons,
            alpha
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

    let save_model = |model: &Model| -> Result<(), String> {
        log(String::from("Saving model."), false);
        
        model.save_to_file(&path)
            .map_err(|e| format!("Failed to save model: {e}"))
    };

    let training_data_bound = utils::data::read_training_data(&source)?;
    let mut training_data = training_data_bound
        .iter()
        .map(|(features, outputs)| ModelTrainingExample::new(features.clone(), outputs.clone()));

    let learning_rate = f32::max(LEARNING_RATE_MIN, match args.get("--learning-rate") {
        Some(Some(p)) => p.parse::<f32>().map_err(|err| format!("Invalid learning rate. {err}")),
        _ => Ok(LEARNING_RATE_DFT)
    }?);

    let lambda = f32::max(LAMBDA_MIN, match args.get("--l2") {
        Some(Some(p)) => p.parse::<f32>().map_err(|err| format!("Invalid L2 Regularization coefficient. {err}")),
        _ => Ok(LAMBDA_DFT)
    }?);

    let adam_beta1 = f32::max(ADAM_BETA1_MIN, match args.get("--adam-1") {
        Some(Some(p)) => p.parse::<f32>().map_err(|err| format!("Invalid adam beta 1. {err}")),
        _ => Ok(ADAM_BETA1_DFT)
    }?);

    let adam_beta2 = f32::max(ADAM_BETA2_MIN, match args.get("--adam-2") {
        Some(Some(p)) => p.parse::<f32>().map_err(|err| format!("Invalid adam beta 1. {err}")),
        _ => Ok(ADAM_BETA2_DFT)
    }?);

    let batch_size = u32::max(BATCH_SIZE_MIN, match args.get("--batch-size") {
        Some(Some(p)) => p.parse::<u32>().map_err(|err| format!("Invalid batch size. {err}")),
        _ => Ok(BATCH_SIZE_DFT)
    }?);

    let training_params = ModelTrainingParams {
        learning_rate,
        batch_size,
        lambda,
        adam: (adam_beta1, adam_beta2)
    };

    //Train model
    let trained_model = ModelConverter::train_model(&model_initial, &training_params, &mut training_data, &log, &save_model)?;

    //Save trained model
    save_model(&trained_model)?;

    Ok(())
}

