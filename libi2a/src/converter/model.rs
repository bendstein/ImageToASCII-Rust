use std::mem::size_of;

use imageproc::image::GenericImageView;
use nalgebra::{DMatrix, DVector};
use rand_distr::Distribution;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use utils::vmath;

use super::{converter_utils, Converter};

type BackpropData = (Vec<DMatrix<f64>>, Vec<DVector<f64>>);

#[derive(Clone)]
pub struct Model {
    glyphs: Vec<char>,
    biases: Vec<DVector<f64>>,
    weights: Vec<DMatrix<f64>>,
    alpha: f64
}

#[derive(Clone)]
pub struct ModelInitParams {
    pub feature_count: u32,
    pub hidden_layer_count: u32,
    pub hidden_layer_neuron_count: u32,
    pub alpha: f64,
    pub glyphs: Vec<char>
}

#[derive(Clone)]
pub struct ModelConverter {
    model: Model
}

#[derive(Clone, Debug)]
pub struct ModelTrainingExample {
    pub intensities: DVector<f64>,
    pub ssims: DVector<f64>
}

pub struct ModelTrainingParams {
    pub learning_rate: f64,
    pub batch_size: u32,
    pub lambda: f64,
    pub adam: (f64, f64)
}

pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub moment1: Vec<DMatrix<f64>>,
    pub moment2: Vec<DMatrix<f64>>,
    pub bias1: Vec<DVector<f64>>,
    pub bias2: Vec<DVector<f64>>
}

impl ModelConverter {
    const ADAM_EPS: f64 = 1e-8_f64;

    pub fn new(model: Model) -> Self {
        Self {
            model
        }
    }

    pub fn train_model(model_initial: &Model, params: &ModelTrainingParams, examples: &mut dyn Iterator<Item=ModelTrainingExample>,
        log: &dyn Fn(String, bool), on_epoch_complete: &dyn Fn(&Model) -> Result<(), String>) -> Result<Model, String> {
        if model_initial.glyphs.is_empty() {
            return Err("Model has no glyphs".to_string());
        }

        if model_initial.weights.is_empty() || model_initial.biases.is_empty() {
            return Err("Model has no weights or biases".to_string());
        }

        if model_initial.weights.len() != model_initial.biases.len() {
            return Err("Model weights and biases count mismatch".to_string());
        }

        let mut epoch = 0;
        let mut model = model_initial.clone();
        let mut adam = Adam::new(params.learning_rate, params.adam.0, params.adam.1, model_initial);

        loop {
            //Get next batch of training examples
            let batch = examples.take(params.batch_size as usize)
                .collect::<Vec<ModelTrainingExample>>();

            //Stop when no more examples
            if batch.is_empty() {
                break;
            }

            //Calculate class weights as being inversely proportional to frequency and confidence
            let batch_ssim_freq = batch.iter().map(|ex| ex.normalized_ssims())
                .try_fold(DVector::zeros(model.glyphs.len()), |acc, ssim| vmath::checked_add_v(&acc, &ssim))
                .map_err(|e| format!("Failed to calculate class weights: {e}"))?;

            // //Z-score normalize
            // let batch_ssim_mean = batch_ssim_freq.mean();
            // let batch_ssim_stddev = batch_ssim_freq.map(|s| (s - batch_ssim_mean).powi(2)).sum().sqrt();

            // let batch_ssim_freq_normalized = if batch_ssim_stddev == 0_f64 {
            //     batch_ssim_freq
            // }
            // else {
            //     batch_ssim_freq.map(|s| (s - batch_ssim_mean) / batch_ssim_stddev)
            // };

            //Scale to be in the range 0-1
            let batch_ssim_freq_min = batch_ssim_freq.min();

            let batch_ssim_freq_translated = if batch_ssim_freq_min < 0_f64 {
                batch_ssim_freq.map(|s| s - batch_ssim_freq_min)
            }
            else {
                batch_ssim_freq
            };

            let batch_ssim_freq_max = batch_ssim_freq_translated.max();

            let batch_ssim_freq_scaled = if batch_ssim_freq_max <= 1_f64 {
                batch_ssim_freq_translated
            }
            else {
                batch_ssim_freq_translated.map(|s| s / batch_ssim_freq_max)
            };

            // let batch_ssim_freq_scaled: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>> = batch_ssim_freq
            //     .map(|s| (s - batch_ssim_freq_min) / (if batch_ssim_freq_max == 0_f64 { 1_f64 } else { batch_ssim_freq_max }));

            //Calculate class weights using complement of logistic function
            const K: f64 = 8_f64;
            const C: f64 = 0.5_f64;

            let logistic = |s: f64| 1_f64 / (1_f64 + (-K * (s - C)).exp());

            let class_weights = batch_ssim_freq_scaled
                .map(|s| 1_f64 - logistic(s) + logistic(0_f64));

            //Propagate forwards
            let activations = batch.par_iter()
                .map(|example| {
                    let activations = Self::propagate_forwards(&model, example.clone())?;

                    //Get last layer
                    let output = activations.last()
                        .ok_or("Failed to get output layer")?;

                    //Compare the activations to the expected value
                    let expected = example.normalized_ssims();

                    //Calculate total loss using (weighted) cross-entropy loss
                    let weighted_output = vmath::checked_component_mul_v(
                        &output.map(|a| a.ln()),
                        &class_weights)
                        .map_err(|e| format!("Failed to calculate weighted output for cross-entropy loss: {e}"))?;
                    let loss = -vmath::checked_dot_v(&expected, &weighted_output)
                        .map_err(|e| format!("Failed to calculate dot product for cross-entropy loss: {e}"))?;

                    //Calculate component loss using binary cross-entropy loss
                    let ln_output = output.map(|a| a.ln());
                    let ln_output_comp = output.map(|a| (1_f64 - a).ln());
                    let expected_comp = expected.map(|e| 1_f64 - e);
                    let cel_a = vmath::checked_component_mul_v(&expected, &ln_output)
                        .map_err(|e| format!("Failed to calculate component loss part A: {e}"))?;
                    let cel_b = vmath::checked_component_mul_v(&expected_comp, &ln_output_comp)
                        .map_err(|e| format!("Failed to calculate component loss part B: {e}"))?;
                    let cel = vmath::checked_add_v(&cel_a, &cel_b)
                        .map_err(|e| format!("Failed to calculate component loss: {e}"))?;
                    let component_loss = -vmath::checked_component_mul_v(&cel, &class_weights)
                        .map_err(|e| format!("Failed to calculate weighted component loss: {e}"))?;

                    //Return activations, total loss, and pointwise loss
                    Ok((activations, loss, component_loss))
                })
                .collect::<Result<Vec<(Vec<DVector<f64>>, f64, DVector<f64>)>, String>>()
                .map_err(|e| format!("Failed to propagate forwards. {e}"))?;

            //Get average batch loss
            let avg_loss = activations.iter()
                .map(|(_, loss, _)| loss).sum::<f64>() 
                / params.batch_size as f64;

            log(format!("Epoch {epoch} - Average loss: {avg_loss}"), false);

            //Get average pointwise loss across each batch (unused)
            let _avg_pointwise_loss = activations.iter()
                .map(|(_, _, pointwise)| pointwise)
                .fold(DVector::zeros(model.glyphs.len()), |acc, pointwise| acc + pointwise)
                .map(|p| p / params.batch_size as f64);

            //Propagate backwards
            let gradients = batch.par_iter()
                .zip(activations)
                .map(|(example, (activations, _, _))| 
                    Self::propagate_backwards(&model, example.clone(), activations, class_weights.clone()))
                .collect::<Result<Vec<(Vec<DMatrix<f64>>, Vec<DVector<f64>>)>, String>>()
                .map_err(|e| format!("Failed to propagate backwards. {e}"))?;

            //Average weights and biases across all trials
            let first_trial_gradients = gradients.first()
                .ok_or("Backpropagation resulted in no gradients.")?;
            let mut average_weight_gradients = Vec::<DMatrix<f64>>::new();
            let mut average_bias_gradients = Vec::<DVector<f64>>::new();

            for l in 0..first_trial_gradients.0.len() {
                let mut weight = DMatrix::zeros(first_trial_gradients.0[l].nrows(), first_trial_gradients.0[l].ncols());
                let mut bias = DVector::zeros(first_trial_gradients.1[l].nrows());

                for (gweight, gbias) in gradients.iter() {
                    let wsum = vmath::checked_add(&weight, &gweight[l])
                        .map_err(|e| format!("Failed to add weight gradients: {e}"))?;
                    let bsum = vmath::checked_add_v(&bias, &gbias[l])
                        .map_err(|e| format!("Failed to add bias gradients: {e}"))?;
                    weight = wsum;
                    bias = bsum;
                }

                average_weight_gradients.push(weight / params.batch_size as f64);
                average_bias_gradients.push(bias / params.batch_size as f64);
            }

            //Update weights using Adam
            model = Self::update_adam_weights(&mut adam, &model, (average_weight_gradients, average_bias_gradients), params.lambda, epoch)
                .map_err(|e| format!("Failed to update weights. {e}"))?;

            //Increment epoch counter
            epoch += 1;

            //Save model after each epoch
            on_epoch_complete(&model)
                .map_err(|e| format!("Failed to perform epoch-complete callback. {e}"))?;
        }

        Ok(model)
    }

    fn propagate_forwards(model: &Model, example: ModelTrainingExample) -> Result<Vec<DVector<f64>>, String> {
        let neurons = example.normalized_intensities();
        let mut activations = vec![neurons];

        for l in 0..model.weights.len() {
            let weights = &model.weights[l];
            let biases = &model.biases[l];

            //Take product of weights and neurons, and add bias to get activation inputs
            let product = vmath::checked_mul_mv(weights, &activations[l])
                .map_err(|e| format!("Failed to multiply weights and activations: {e}"))?;
            let weighted_sum = vmath::checked_add_v(&product, biases)
                .map_err(|e| format!("Failed to add bias to weighted sum: {e}"))?;

            if l == model.weights.len() - 1 {
                //Output layer - Softmax
                activations.push(Self::normalized_softmax(weighted_sum));
            }
            else {
                //Hidden layer - Leaky ReLU
                activations.push(weighted_sum.map(|a| if a < 0_f64 { model.alpha * a } else { a }));
            }
        }

        Ok(activations)
    }

    fn propagate_backwards(model: &Model, example: ModelTrainingExample,
        activations: Vec<DVector<f64>>, class_weights: DVector<f64>) -> Result<BackpropData, String> {
        let expected = example.normalized_ssims();
        let mut all_weight_gradients = Vec::<DMatrix<f64>>::new();
        let mut all_bias_gradients = Vec::<DVector<f64>>::new();
        let mut all_errors = Vec::<DVector<f64>>::new();
        
        //Init vectors
        for l in 0..model.weights.len() {
            all_weight_gradients.push(DMatrix::zeros(model.weights[l].nrows(), model.weights[l].ncols()));
            all_bias_gradients.push(DVector::zeros(model.biases[l].nrows()));
            all_errors.push(DVector::zeros(model.weights[l].ncols()));
        }

        for l in (0..model.weights.len()).rev() {
            let neurons = &activations[l + 1];

            let error_term = if l == model.weights.len() - 1 {
                //Gradient of cross-entropy loss wrt softmax activation simplifies to predicted - expected
                //Multiply by class weights
                let diff = vmath::checked_sub_v(neurons, &expected)
                    .map_err(|e| format!("Failed to calculate gradient for softmax: {e}"))?;
                let weighted = vmath::checked_component_mul_v(&diff, &class_weights)
                    .map_err(|e| format!("Failed to calculate weighted gradient for softmax: {e}"))?;

                Result::<DVector<f64>, String>::Ok(weighted)
            }
            else {
                //Get weights and error terms for next layer
                let next_weights = &model.weights[l + 1];
                let next_errors = &all_errors[l + 1];

                //Using chain rule, error term is the transpose of the next layer's weights times its error term,
                //times the derivative of Leaky ReLU
                let transposed = next_weights.transpose();
                let product = vmath::checked_mul_mv(&transposed, next_errors)
                    .map_err(|e| format!("Failed to multiply weights and error terms for leaky ReLU gradient: {e}"))?;
                let error_term = vmath::checked_component_mul_v(&product, &neurons.map(|a| if a < 0_f64 { model.alpha } else { 1_f64 }))?;
                Ok(error_term)
            }?;

            //Grad of loss wrt bias is just error
            all_bias_gradients[l] = error_term.clone();

            //Grad of loss wrt weights is error * prev activations transpose (chain rule)
            let activations = &activations[l];
            let mut activations_transpose: DMatrix<f64> = DMatrix::zeros(1, activations.len());
            for i in 0..activations.len() {
                activations_transpose[(0, i)] = activations[i];
            }

            all_weight_gradients[l] = vmath::checked_mul_vm(&error_term, &activations_transpose)
                .map_err(|e| format!("Failed to multiply error term and activations for weight gradient: {e}"))?;

            all_errors[l] = error_term;
        }

        Ok((all_weight_gradients, all_bias_gradients))
    }

    fn update_adam_weights(adam: &mut Adam, model: &Model, gradients: (Vec<DMatrix<f64>>, Vec<DVector<f64>>),
        l2_coeff: f64, epoch: u32) -> Result<Model, String> {
        let mut model_clone = model.clone();
        let (gweight, gbias) = gradients;

        for l in 0..model.weights.len() {
            //Update adam first moment
            adam.moment1[l] = vmath::checked_add(&(&adam.moment1[l] * adam.beta1), &(&gweight[l] * (1_f64 - adam.beta1)))
                .map_err(|e| format!("Failed to update Adam first moment: {e}"))?;
            adam.bias1[l] = vmath::checked_add_v(&(&adam.bias1[l] * adam.beta1), &(&gbias[l] * (1_f64 - adam.beta1)))
                .map_err(|e| format!("Failed to update Adam first bias: {e}"))?;

            //Update second moment
            adam.moment2[l] = vmath::checked_add(&(&adam.moment2[l] * adam.beta2), &(&gweight[l].map(|x| x * x) * (1_f64 - adam.beta2)))
                .map_err(|e| format!("Failed to update Adam second moment: {e}"))?;
            adam.bias2[l] = vmath::checked_add_v(&(&adam.bias2[l] * adam.beta2), &(&gbias[l].map(|x| x * x) * (1_f64 - adam.beta2)))
                .map_err(|e| format!("Failed to update Adam second bias: {e}"))?;

            //Compute bias-corrected first moment
            let moment1_corrected = &adam.moment1[l] / (1_f64 - adam.beta1.powi(epoch as i32 + 1));
            let bias1_corrected = &adam.bias1[l] / (1_f64 - adam.beta1.powi(epoch as i32 + 1));

            //Compute bias-corrected second moment
            let moment2_corrected = &adam.moment2[l] / (1_f64 - adam.beta2.powi(epoch as i32 + 1));
            let bias2_corrected = &adam.bias2[l] / (1_f64 - adam.beta2.powi(epoch as i32 + 1));

            //Compute learning rates (L2 regularized for weight learning rate)
            let learning_rate_unregularized = vmath::checked_component_div(&moment1_corrected, &moment2_corrected.map(|x| x.sqrt() + Self::ADAM_EPS))
                .map_err(|e| format!("Failed to calculate learning rate: {e}"))?;
            let learning_rate = vmath::checked_add(&learning_rate_unregularized, &(&model.weights[l] * l2_coeff))
                .map_err(|e| format!("Failed to add L2 regularization to learning rate: {e}"))?;
            let bias_learning_rate = vmath::checked_component_div_v(&bias1_corrected, &bias2_corrected.map(|x| x.sqrt() + Self::ADAM_EPS))
                .map_err(|e| format!("Failed to calculate bias learning rate: {e}"))?;

            //Update weights and biases
            let updated_weights = vmath::checked_sub(&model_clone.weights[l], &(adam.learning_rate * learning_rate))
                .map_err(|e| format!("Failed to update weights: {e}"))?;
            model_clone.weights[l] = updated_weights;

            let updated_biases = vmath::checked_sub_v(&model_clone.biases[l], &(adam.learning_rate * bias_learning_rate))
                .map_err(|e| format!("Failed to update biases: {e}"))?;
            model_clone.biases[l] = updated_biases;
        }

        Ok(model_clone)
    }

    fn normalized_softmax(input: DVector<f64>) -> DVector<f64> {
        let max = input.max();
        let exp: Vec<f64> = input.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        let result = exp.iter().map(|x| x / sum).collect();
        DVector::from_vec(result)    
    }
}

impl Converter for ModelConverter {
    fn best_match(&self, tile: &imageproc::image::DynamicImage) -> char {
        //Get intensities of pixels in each tile
        let intensities = tile.pixels().map(|p| converter_utils::get_intensity(p.2)).collect();
        let example = ModelTrainingExample::new(intensities, vec![0_f64; self.model.glyphs.len()]);
        
        //Send tile through the neural net
        let maybe_activations: Result<Vec<DVector<f64>>, String> = Self::propagate_forwards(&self.model, example);

        if maybe_activations.is_err() {
            return ' ';
        }

        let activations = maybe_activations.unwrap();

        if activations.is_empty() {
            return ' ';
        }
        
        //Get output layer
        let output = activations.last().unwrap();

        //Get index of max value
        let mut max_index = 0_usize;
        let mut max_value = 0_f64;

        for (i, &value) in output.iter().enumerate() {
            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        //Get one-hot encoded glyph
        self.model.glyphs[max_index] 
    }
    
    fn get_tile_size(&self) -> u32 {
        (self.model.feature_count() as f64)
            .sqrt()
            .round() as u32
    }
}

impl Model {
    pub fn init_from_params(params: ModelInitParams) -> Model {
        let (weights, biases) = Model::init_weights_he_normal(&params);

        Model {
            glyphs: params.glyphs,
            alpha: params.alpha,
            weights,
            biases
        }
    }

    pub fn load_from_file(file_path: &str) -> Result<Model, String> {
        let bytes = std::fs::read(file_path)
            .map_err(|e| format!("Failed to load model. {e}"))?;

        Model::try_from(bytes)
    }

    pub fn save_to_file(&self, file_path: &str) -> Result<(), String> {
        let bytes: Vec<u8> = self.into();

        //Create directory if not exists
        let maybe_dir = std::path::Path::new(file_path).parent();

        if let Some(dir) = maybe_dir {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("Failed to create model directory. {e}"))?;
        }

        std::fs::write(file_path, bytes)
            .map_err(|e| format!("Failed to save model. {e}"))
            .map(|_| ())
    }

    pub fn feature_count(&self) -> usize {
        self.weights
            .first()
            .map(|w| w.ncols() as f64)
            .unwrap_or(0_f64)
            as usize
    }

    pub fn output_count(&self) -> usize {
        self.glyphs.len()
    }

    fn init_weights_he_normal(params: &ModelInitParams) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>) {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut rng = rand::thread_rng();

        let layers = params.hidden_layer_count + 1;

        for i in 0..layers {
            let activations_count = if i == layers - 1 { params.glyphs.len() } else { params.hidden_layer_neuron_count as usize };
            let neuron_count = if i == 0 { params.feature_count } else { params.hidden_layer_neuron_count } as usize;

            let mut weight = DMatrix::zeros(activations_count, neuron_count);
            let bias = DVector::zeros(activations_count);

            let fan_in = neuron_count as f64;
            let std_dev = (2_f64 / fan_in).sqrt();
            let norm = rand_distr::Normal::new(0_f64, std_dev).unwrap();

            //Init weights
            for i in 0..activations_count {
                for j in 0..neuron_count {
                    weight[(i, j)] = norm.sample(&mut rng);
                }
            }

            weights.push(weight);
            biases.push(bias);
        }

        (weights, biases)
    }
}

impl TryFrom<&[u8]> for Model {
    type Error = String;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Model::try_from(value.to_vec())
    }
}

impl TryFrom<Vec<u8>> for Model {
    type Error = String;

    fn try_from(value: Vec<u8>) -> Result<Self, Self::Error> {
        let mut glyphs: Vec<char> = Vec::new();
        let mut biases: Vec<DVector<f64>> = Vec::new();
        let mut weights: Vec<DMatrix<f64>> = Vec::new();

        let mut offset = 0_usize;

        //Read number of glyphs
        let glyph_count = usize::from_le_bytes(value[offset..offset + size_of::<usize>()].try_into()
            .map_err(|e| format!("Failed to read number of glyphs. {e}"))?);
        offset += size_of::<usize>();

        //Read glyphs
        for _ in 0..glyph_count {
            //Read glyph len
            let glyph_len = value[offset];
            offset += 1;

            //Read glyph
            let glyph = std::str::from_utf8(&value[offset..offset + glyph_len as usize])
                .map_err(|e| format!("Failed to read glyph from utf8. {e}"))?
                .chars().next()
                .ok_or_else(|| "Failed to read glyph".to_string())?;
            offset += glyph_len as usize;

            glyphs.push(glyph);
        }       

        //Read alpha
        let alpha = f64::from_le_bytes(value[offset..offset + size_of::<f64>()].try_into()
            .map_err(|e| format!("Failed to read alpha. {e}"))?);
        offset += size_of::<f64>();

        //Read layer count
        let layer_count = usize::from_le_bytes(value[offset..offset + size_of::<usize>()].try_into()
            .map_err(|e| format!("Failed to read layer count. {e}"))?);
        offset += size_of::<usize>();

        for _ in 0..layer_count {
            let rows = usize::from_le_bytes(value[offset..offset + size_of::<usize>()].try_into()
                .map_err(|e| format!("Failed to read layer row count. {e}"))?);
            offset += size_of::<usize>();

            let cols = usize::from_le_bytes(value[offset..offset + size_of::<usize>()].try_into()
                .map_err(|e| format!("Failed to read layer column count. {e}"))?);
            offset += size_of::<usize>();

            let mut weight = DMatrix::zeros(rows, cols);
            let mut bias = DVector::zeros(rows);

            //Read weights
            for i in 0..rows {
                for j in 0..cols {
                    weight[(i, j)] = f64::from_le_bytes(value[offset..offset + size_of::<f64>()].try_into()
                        .map_err(|e| format!("Failed to read weight. {e}"))?);
                    offset += size_of::<f64>();
                }
            }

            //Read biases
            for i in 0..rows {
                bias[i] = f64::from_le_bytes(value[offset..offset + size_of::<f64>()].try_into()
                    .map_err(|e| format!("Failed to read bias. {e}"))?);
                offset += size_of::<f64>();
            }

            weights.push(weight);
            biases.push(bias);
        }

        Ok(Self {
            glyphs,
            biases,
            weights,
            alpha
        })
    }
}

impl From<&Model> for Vec<u8> {
    fn from(value: &Model) -> Self {
        let mut result: Vec<u8> = Vec::new();

        //Write number of glyphs
        result.extend_from_slice(&value.glyphs.len().to_le_bytes());

        //Write glyphs
        for glyph in value.glyphs.iter() {
            let mut buffer = [0; size_of::<char>()];
            let glyph_bytes = glyph.encode_utf8(&mut buffer);
            let glyph_size = glyph_bytes.len() as u8;

            //Write glyph len
            result.push(glyph_size);

            //Write glyph
            result.extend(glyph_bytes.as_bytes());
        }

        //Write alpha
        result.extend_from_slice(&value.alpha.to_le_bytes());

        //Write layer count
        result.extend_from_slice(&value.biases.len().to_le_bytes());

        for l in 0..value.biases.len() {
            let weights = &value.weights[l];
            let biases = &value.biases[l];

            //Write dimensions of layer
            result.extend_from_slice(&weights.nrows().to_le_bytes());
            result.extend_from_slice(&weights.ncols().to_le_bytes());

            //Write weights
            for i in 0..weights.nrows() {
                for j in 0..weights.ncols() {
                    result.extend_from_slice(&weights[(i, j)].to_le_bytes());
                }
            }

            //Write biases
            for i in 0..biases.nrows() {
                result.extend_from_slice(&biases[i].to_le_bytes());
            }
        }

        result
    }
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, model: &Model) -> Adam {
        let mut moment1 = Vec::new();
        let mut moment2 = Vec::new();
        let mut bias1 = Vec::new();
        let mut bias2 = Vec::new();

        for i in 0..model.biases.len() {
            let weight = &model.weights[i];
            let bias = &model.biases[i];

            let mut m1 = DMatrix::zeros(weight.nrows(), weight.ncols());
            let mut m2 = DMatrix::zeros(weight.nrows(), weight.ncols());
            let mut b1 = DVector::zeros(bias.nrows());
            let mut b2 = DVector::zeros(bias.nrows());

            for i in 0..weight.nrows() {
                for j in 0..weight.ncols() {
                    m1[(i, j)] = 0_f64;
                    m2[(i, j)] = 0_f64;
                }
            }

            for i in 0..bias.nrows() {
                b1[i] = 0_f64;
                b2[i] = 0_f64;
            }

            moment1.push(m1);
            moment2.push(m2);
            bias1.push(b1);
            bias2.push(b2);
        }

        Adam {
            learning_rate,
            beta1,
            beta2,
            moment1,
            moment2,
            bias1,
            bias2
        }
    }
}

impl ModelTrainingExample {
    pub fn new(features: Vec<f64>, expected: Vec<f64>) -> ModelTrainingExample {
        ModelTrainingExample {
            intensities: DVector::from_vec(features),
            ssims: DVector::from_vec(expected)
        }
    }

    pub fn normalized_intensities(&self) -> DVector<f64> {
        // //Z-score normalization
        // let mean = self.intensities.mean();
        // let stddev = self.intensities.map(|i| (i - mean).powi(2)).sum().sqrt();

        // let normalized = if stddev == 0_f64 {
        //     self.intensities.clone()
        // }
        // else {
        //     self.intensities.map(|i| (i - mean) / stddev)
        // };

        // //MAD normalization
        // let median = vmath::median(self.intensities.data.as_vec());
        // let mad_vector = &self.intensities.map(|i| (i - median).abs());
        // let mad = vmath::median(mad_vector.data.as_vec());

        // let normalized = if mad == 0_f64 {
        //     self.intensities.clone()
        // }
        // else {
        //     self.intensities.map(|i| (i - median) / mad)
        // };

        let normalized = self.intensities.clone();

        //Scale to be in the range 0-1
        let min = normalized.min();
        
        let translated = if min < 0_f64 {
            normalized.map(|i| i - min)
        }
        else {
            normalized
        };

        let max = translated.max();

        if max <= 1_f64 {
            translated
        }
        else {
            translated.map(|i| i / max)
        }
    }

    pub fn normalized_ssims(&self) -> DVector<f64> {
        const COERCE_TO_ZERO: f64 = 1e-9_f64;
        const STD_DEV: f64 = 0.15_f64;

        //Normalize ssims, gaussian weighted by their ratio to the max
        let max = self.ssims.max();

        if max == 0_f64 {
            return self.ssims.clone();
        }

        //Let gaussian weights be the ssim mapped to the normal curve
        let weights = DVector::from_iterator(self.ssims.nrows(), self.ssims.iter()
        .map(|&s| {
            let exponent = -(s / max - 1.0).powi(2) / (2.0 * STD_DEV * STD_DEV);
            let numerator = exponent.exp();
            let denominator = (2.0 * std::f64::consts::PI).sqrt() * STD_DEV;
            numerator / denominator
        }));

        //Penalize SSIMs
        let penalized = self.ssims.component_mul(&weights)
            .map(|p| if p <= COERCE_TO_ZERO { 0_f64 } else { p });

        //Normalize
        let sum = penalized.sum();

        if sum == 0_f64 {
            penalized.clone()
        }
        else {
            penalized / sum
        }

    }
}