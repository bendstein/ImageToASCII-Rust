use std::mem::size_of;

use imageproc::image::GenericImageView;
use nalgebra::{DMatrix, DVector};
use rand_distr::Distribution;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use super::{converter_utils, Converter};

#[derive(Clone)]
pub struct Model {
    glyphs: Vec<char>,
    biases: Vec<DVector<f32>>,
    weights: Vec<DMatrix<f32>>,
    alpha: f32
}

#[derive(Clone)]
pub struct ModelInitParams {
    pub feature_count: u32,
    pub hidden_layer_count: u32,
    pub hidden_layer_neuron_count: u32,
    pub alpha: f32,
    pub glyphs: Vec<char>
}

#[derive(Clone)]
pub struct ModelConverter {
    model: Model
}

#[derive(Clone, Debug)]
pub struct ModelTrainingExample {
    pub intensities: DVector<f32>,
    pub ssims: DVector<f32>
}

pub struct ModelTrainingParams {
    pub learning_rate: f32,
    pub batch_size: u32,
    pub lambda: f32,
    pub adam: (f32, f32)
}

pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub moment1: Vec<DMatrix<f32>>,
    pub moment2: Vec<DMatrix<f32>>,
    pub bias1: Vec<DVector<f32>>,
    pub bias2: Vec<DVector<f32>>
}

impl ModelConverter {
    const ADAM_EPS: f32 = 1e-8_f32;
    
    pub fn new(model: Model) -> Self {
        Self {
            model
        }
    }

    pub fn train_model(model_initial: &Model, params: &ModelTrainingParams, examples: &mut dyn Iterator<Item=ModelTrainingExample>,
        log: &dyn Fn(String, bool), on_epoch_complete: &dyn Fn(&Model) -> Result<(), String>) -> Result<Model, String> {

        let err = |msg: &str| -> Result<Model, String> {
            log(msg.to_string(), true);
            Err(msg.to_string())
        };

        if model_initial.glyphs.is_empty() {
            return err("Model has no glyphs");
        }

        if model_initial.weights.is_empty() || model_initial.biases.is_empty() {
            return err("Model has no weights or biases");
        }

        if model_initial.weights.len() != model_initial.biases.len() {
            return err("Model weights and biases count mismatch");
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
                .fold(DVector::zeros(model.glyphs.len()), |acc, ssim| acc + ssim);

            //Z-score normalize
            let batch_ssim_mean = batch_ssim_freq.mean();
            let batch_ssim_stddev = batch_ssim_freq.map(|s| (s - batch_ssim_mean).powi(2)).sum().sqrt();

            let batch_ssim_freq = if batch_ssim_stddev == 0_f32 {
                batch_ssim_freq.clone()
            }
            else {
                batch_ssim_freq.map(|s| (s - batch_ssim_mean) / batch_ssim_stddev)
            };

            //Scale to be in the range 0-1
            let batch_ssim_freq_min = batch_ssim_freq.min();
            let batch_ssim_freq_max = batch_ssim_freq.max();
            let batch_ssim_freq = batch_ssim_freq
                .map(|s| (s - batch_ssim_freq_min) / (if batch_ssim_freq_max == 0_f32 { 1_f32 } else { batch_ssim_freq_max }));

            //Calculate class weights using complement of logistic function
            const K: f32 = 8_f32;
            const C: f32 = 0.5_f32;

            let logit = |s: f32| 1_f32 / (1_f32 + (-K * (s - C)).exp());

            let class_weights = batch_ssim_freq
                .map(|s| 1_f32 - logit(s) + logit(0_f32));

            //Propagate forwards
            let activations = batch.par_iter()
                .map(|example| {
                    let activations = Self::propagate_forwards(&model, example.clone());

                    //Get last layer
                    let output = activations.last().unwrap();

                    //Compare the activations to the expected value
                    let expected = example.normalized_ssims();

                    //Calculate total loss using (weighted) cross-entropy loss
                    let loss = -expected.dot(&output.map(|a| a.ln()).component_mul(&class_weights));

                    //Calculate component loss using binary cross-entropy loss
                    let component_loss = (expected.component_mul(&output.map(|a| a.ln()))
                        + expected.map(|e| 1_f32 - e).component_mul(&output.map(|a| 1_f32 - a).map(|a| a.ln())))
                        .component_mul(&class_weights);

                    //Return activations, total loss, and pointwise loss
                    (activations, loss, component_loss)
                })
                .collect::<Vec<(Vec<DVector<f32>>, f32, DVector<f32>)>>();

            //Get average batch loss
            let avg_loss = activations.iter()
                .map(|(_, loss, _)| loss).sum::<f32>() 
                / params.batch_size as f32;

            log(format!("Epoch {epoch} - Average loss: {avg_loss}"), false);

            // //Get average pointwise loss across each batch
            // let avg_pointwise_loss = activations.iter()
            //     .map(|(_, _, pointwise)| pointwise)
            //     .fold(DVector::zeros(model.glyphs.len()), |acc, pointwise| acc + pointwise)
            //     .map(|p| p / params.batch_size as f32);

            //Propagate backwards
            let gradients = batch.par_iter()
                .zip(activations)
                .map(|(example, (activations, _, _))| 
                    Self::propagate_backwards(&model, example.clone(), activations, class_weights.clone()))
                .collect::<Vec<(Vec<DMatrix<f32>>, Vec<DVector<f32>>)>>();

            //Average weights and biases across all trials
            let first_trial_gradients = gradients.first().unwrap();
            let mut average_weight_gradients = Vec::<DMatrix<f32>>::new();
            let mut average_bias_gradients = Vec::<DVector<f32>>::new();

            for l in 0..first_trial_gradients.0.len() {
                let mut weight = DMatrix::zeros(first_trial_gradients.0[l].nrows(), first_trial_gradients.0[l].ncols());
                let mut bias = DVector::zeros(first_trial_gradients.1[l].nrows());

                for (gweight, gbias) in gradients.iter() {
                    weight += &gweight[l];
                    bias += &gbias[l];
                }

                average_weight_gradients.push(weight / params.batch_size as f32);
                average_bias_gradients.push(bias / params.batch_size as f32);
            }

            //Update weights using Adam
            model = Self::update_adam_weights(&mut adam, &model, (average_weight_gradients, average_bias_gradients), params.lambda, epoch);

            //Increment epoch counter
            epoch += 1;

            //Save model after each epoch
            on_epoch_complete(&model)?;
        }

        Ok(model)
    }

    fn propagate_forwards(model: &Model, example: ModelTrainingExample) -> Vec<DVector<f32>> {
        let neurons = example.normalized_intensities();
        let mut activations = vec![neurons];

        for l in 0..model.weights.len() {
            let weights = &model.weights[l];
            let biases = &model.biases[l];

            //Take dot product of weights and neurons, and add bias to get activation inputs
            let weighted_sum = weights * &activations[l] + biases;

            if l == model.weights.len() - 1 {
                //Output layer - Softmax
                activations.push(Self::normalized_softmax(weighted_sum));
            }
            else {
                //Hidden layer - Leaky ReLU
                activations.push(weighted_sum.map(|a| if a < 0_f32 { model.alpha * a } else { a }));
            }
        }

        activations
    }

    fn propagate_backwards(model: &Model, example: ModelTrainingExample, 
        activations: Vec<DVector<f32>>, class_weights: DVector<f32>) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        let expected = example.normalized_ssims();
        let mut all_weight_gradients = Vec::<DMatrix<f32>>::new();
        let mut all_bias_gradients = Vec::<DVector<f32>>::new();
        let mut all_errors = Vec::<DVector<f32>>::new();
        
        //Init vectors
        for l in 0..model.weights.len() {
            all_weight_gradients.push(DMatrix::zeros(model.weights[l].nrows(), model.weights[l].ncols()));
            all_bias_gradients.push(DVector::zeros(model.biases[l].nrows()));
            all_errors.push(DVector::zeros(model.weights[l].ncols()));
        }

        for l in (0..model.weights.len()).rev() {
            let neurons = &activations[l];
            
            let error_term = if l == model.weights.len() - 1 {
                //Gradient of cross-entropy loss wrt softmax activation simplifies to predicted - expected
                //Multiply by class weights
                (neurons - &expected).component_mul(&class_weights)
            }
            else {
                //Get weights and error terms for next layer
                let next_weights = &model.weights[l + 1];
                let next_errors = all_errors.last().unwrap();

                //Using chain rule, error term is the transpose of the next layer's weights times it's error term,
                //times the derivative of Leaky ReLU
                (next_weights.transpose() * next_errors)
                    .component_mul(&neurons.map(|a| if a < 0_f32 { model.alpha } else { 1_f32 }))
            };

            //Grad of loss wrt bias is just error
            all_bias_gradients[l] = error_term.clone();

            //Grad of loss wrt weights is error * prev activations transpose (chain rule)
            all_weight_gradients[l] = error_term.clone() * activations[l].transpose();

            all_errors[l] = error_term;
        }

        (all_weight_gradients, all_bias_gradients)
    }

    fn update_adam_weights(adam: &mut Adam, model: &Model, gradients: (Vec<DMatrix<f32>>, Vec<DVector<f32>>),
        l2_coeff: f32, epoch: u32) -> Model {
        let mut model_clone = model.clone();
        let (gweight, gbias) = gradients;

        for l in 0..model.weights.len() {
            //Update adam first moment
            adam.moment1[l] = &adam.moment1[l] * adam.beta1 + &gweight[l] * (1_f32 - adam.beta1);
            adam.bias1[l] = &adam.bias1[l] * adam.beta1 + &gbias[l] * (1_f32 - adam.beta1);

            //Update second moment
            adam.moment2[l] = &adam.moment2[l] * adam.beta2 + &gweight[l].map(|x| x * x) * (1_f32 - adam.beta2);
            adam.bias2[l] = &adam.bias2[l] * adam.beta2 + &gbias[l].map(|x| x * x) * (1_f32 - adam.beta2);

            //Compute bias-corrected first moment
            let moment1_corrected = &adam.moment1[l] / (1_f32 - adam.beta1.powi(epoch as i32 + 1));
            let bias1_corrected = &adam.bias1[l] / (1_f32 - adam.beta1.powi(epoch as i32 + 1));

            //Compute bias-corrected second moment
            let moment2_corrected = &adam.moment2[l] / (1_f32 - adam.beta2.powi(epoch as i32 + 1));
            let bias2_corrected = &adam.bias2[l] / (1_f32 - adam.beta2.powi(epoch as i32 + 1));

            //Compute learning rates (L2 regularized for weight learning rate)
            let learning_rate = moment1_corrected.component_div(&(moment2_corrected.map(|x| x.sqrt() + Self::ADAM_EPS)))
                + (l2_coeff * &model.weights[l]);
            let bias_learning_rate = bias1_corrected.component_div(&(bias2_corrected.map(|x| x.sqrt() + Self::ADAM_EPS)));

            //Update weights and biases
            model_clone.weights[l] -= adam.learning_rate * learning_rate;
            model_clone.biases[l] -= adam.learning_rate * bias_learning_rate;
        }

        model_clone
    }

    fn normalized_softmax(input: DVector<f32>) -> DVector<f32> {
        let max = input.max();
        let exp: Vec<f32> = input.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let result = exp.iter().map(|x| x / sum).collect();
        DVector::from_vec(result)    
    }
}

impl Converter for ModelConverter {
    fn best_match(&self, tile: &imageproc::image::DynamicImage) -> char {
        //Get intensities of pixels in each tile
        let intensities = tile.pixels().map(|p| converter_utils::get_intensity(p.2)).collect();
        let example = ModelTrainingExample::new(intensities, vec![0_f32; self.model.glyphs.len()]);
        
        //Send tile through the neural net
        let activations: Vec<DVector<f32>> = Self::propagate_forwards(&self.model, example);
        
        if activations.is_empty() {
            return ' ';
        }
        
        //Get output layer
        let output = activations.last().unwrap();

        //Get index of max value
        let mut max_index = 0_usize;
        let mut max_value = 0_f32;

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
        self.model.weights
            .first()
            .map(|w| w.nrows() as f32)
            .unwrap_or(0_f32)
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

    fn init_weights_he_normal(params: &ModelInitParams) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut rng = rand::thread_rng();

        for i in 0..(params.hidden_layer_count + 1) {
            let activations_count = if i == 0 { params.glyphs.len() } else { params.hidden_layer_neuron_count as usize };
            let neuron_count = if i == 0 { params.feature_count } else { params.hidden_layer_neuron_count } as usize;

            let mut weight = DMatrix::zeros(activations_count, neuron_count);
            let bias = DVector::zeros(activations_count);

            let fan_in = neuron_count as f32;
            let std_dev = (2_f32 / fan_in).sqrt();
            let norm = rand_distr::Normal::new(0_f32, std_dev).unwrap();

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
        let mut biases: Vec<DVector<f32>> = Vec::new();
        let mut weights: Vec<DMatrix<f32>> = Vec::new();

        let mut offset = 0_usize;

        //Read number of glyphs
        let glyph_count = usize::from_le_bytes(value[offset..offset + std::mem::size_of::<usize>()].try_into()
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
        let alpha = f32::from_le_bytes(value[offset..offset + std::mem::size_of::<f32>()].try_into()
            .map_err(|e| format!("Failed to read alpha. {e}"))?);
        offset += size_of::<f32>();

        //Read layer count
        let layer_count = value[offset];
        offset += 1;

        for _ in 0..layer_count {
            let rows = value[offset];
            offset += 1;

            let cols = value[offset];
            offset += 1;

            let mut weight = DMatrix::zeros(rows as usize, cols as usize);
            let mut bias = DVector::zeros(rows as usize);

            //Read weights
            for i in 0..rows {
                for j in 0..cols {
                    weight[(i as usize, j as usize)] = f32::from_le_bytes(value[offset..offset + size_of::<f32>()].try_into()
                        .map_err(|e| format!("Failed to read weight. {e}"))?);
                    offset += size_of::<f32>();
                }
            }

            //Read biases
            for i in 0..rows {
                bias[i as usize] = f32::from_le_bytes(value[offset..offset + size_of::<f32>()].try_into()
                    .map_err(|e| format!("Failed to read bias. {e}"))?);
                offset += size_of::<f32>();
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
        result.push(value.biases.len() as u8);

        for l in 0..value.biases.len() {
            let weights = &value.weights[l];
            let biases = &value.biases[l];

            //Write dimensions of layer
            result.push(weights.nrows() as u8);
            result.push(weights.ncols() as u8);

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
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, model: &Model) -> Adam {
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
                    m1[(i, j)] = 0_f32;
                    m2[(i, j)] = 0_f32;
                }
            }

            for i in 0..bias.nrows() {
                b1[i] = 0_f32;
                b2[i] = 0_f32;
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
    pub fn new(features: Vec<f32>, expected: Vec<f32>) -> ModelTrainingExample {
        ModelTrainingExample {
            intensities: DVector::from_vec(features),
            ssims: DVector::from_vec(expected)
        }
    }

    pub fn normalized_intensities(&self) -> DVector<f32> {
        //Z-score normalization
        let mean = self.intensities.mean();
        let stddev = self.intensities.map(|i| (i - mean).powi(2)).sum().sqrt();

        if stddev == 0_f32 {
            self.intensities.clone()
        }
        else {
            self.intensities.map(|i| (i - mean) / stddev)
        }
    }

    pub fn normalized_ssims(&self) -> DVector<f32> {
        const COERCE_TO_ZERO: f32 = 1e-9_f32;
        const STD_DEV: f32 = 0.15_f32;

        //Normalize ssims, gaussian weighted by their percentage of the max
        let max = self.ssims.max();

        if max == 0_f32 {
            return self.ssims.clone();
        }

        //Let gaussian weights be the ssim mapped to the normal curve
        let weights = DVector::from_iterator(self.ssims.nrows(), self.ssims.iter()
        .map(|&s| {
            let exponent = -((s / max - 1.0).powi(2)) / (2.0 * STD_DEV * STD_DEV);
            let numerator = (exponent).exp();
            let denominator = (2.0 * std::f32::consts::PI).sqrt() * STD_DEV;
            numerator / denominator
        }));

        //Penalize SSIMs
        let penalized = self.ssims.component_mul(&weights)
            .map(|p| if p <= COERCE_TO_ZERO { 0_f32 } else { p });

        //Normalize
        let sum = penalized.sum();

        if sum == 0_f32 {
            penalized.clone()
        }
        else {
            penalized / sum
        }

    }
}