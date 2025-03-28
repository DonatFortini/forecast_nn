use crate::back_propagation::NetworkExt;
use crate::dataset_loader::{SimplifiedWeatherDataPoint, prepare_inputs, prepare_outputs};
use crate::layer::Layer;
use crate::neural_network::NeuralNetwork;
use crate::neuron::Neuron;
use rand::Rng;

/// A struct representing a binary classification trainer.
///
/// This trainer is used to configure and execute the training process
/// for binary classification models. It includes parameters for learning
/// rate, number of epochs, and batch size.
///
/// # Fields
///
/// * `learning_rate` - The step size used for updating model parameters during training, maximally 1.0 and minimally 0.0.
/// * `epochs` - The number of complete passes through the training dataset.
/// * `batch_size` - The number of training samples used in one forward/backward pass.
pub struct BinaryTrainer {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
}

impl BinaryTrainer {
    pub fn new(learning_rate: f32, epochs: usize, batch_size: usize) -> Self {
        BinaryTrainer {
            learning_rate,
            epochs,
            batch_size,
        }
    }

    pub fn create_weather_network(
        &self,
        input_size: usize,
        hidden_sizes: &[usize],
    ) -> NeuralNetwork {
        let mut rng = rand::rng();
        let mut layers = Vec::new();
        let mut prev_layer_size = input_size;

        for (layer_idx, &layer_size) in hidden_sizes.iter().enumerate() {
            let mut neurons = Vec::new();

            for i in 0..layer_size {
                let mut weights = Vec::new();
                let weight_scale = (6.0 / (prev_layer_size + layer_size) as f32).sqrt();

                for _ in 0..prev_layer_size {
                    weights.push(rng.random_range(-weight_scale..weight_scale));
                }

                let neuron = Neuron::new(
                    i as u32,
                    format!("Caché{}_{}", layer_idx + 1, i),
                    "relu".to_string(),
                    rng.random_range(-0.1..0.1),
                    weights,
                );

                neurons.push(neuron);
            }

            layers.push(Layer::new(
                layer_idx as u32,
                format!("Caché{}", layer_idx + 1),
                neurons,
            ));
            prev_layer_size = layer_size;
        }

        let mut output_weights = Vec::new();
        let weight_scale = (6.0 / (prev_layer_size + 1) as f32).sqrt();

        for _ in 0..prev_layer_size {
            output_weights.push(rng.random_range(-weight_scale..weight_scale));
        }

        let output_neuron = Neuron::new(
            0,
            "Sortie".to_string(),
            "sigmoid".to_string(),
            rng.random_range(-0.1..0.1),
            output_weights,
        );

        layers.push(Layer::new(
            hidden_sizes.len() as u32,
            "Sortie".to_string(),
            vec![output_neuron],
        ));

        NeuralNetwork::new(layers)
    }

    pub fn train(
        &self,
        network: &mut NeuralNetwork,
        training_data: &[SimplifiedWeatherDataPoint],
        validation_data: &[SimplifiedWeatherDataPoint],
    ) -> f32 {
        let train_inputs = prepare_inputs(training_data);
        let train_outputs = prepare_outputs(training_data);

        let valid_inputs = prepare_inputs(validation_data);
        let valid_outputs = prepare_outputs(validation_data);

        let mut best_validation_accuracy = 0.0;
        let mut patience_counter = 0;
        let patience = 20;

        println!(
            "Début de l'entraînement avec un taux d'apprentissage de : {}",
            self.learning_rate
        );
        println!(
            "Données d'entraînement : {} échantillons, Données de validation : {} échantillons",
            train_inputs.len(),
            valid_inputs.len()
        );

        let precipitation_count = training_data.iter().filter(|d| d.output).count();
        let clear_count = training_data.len() - precipitation_count;
        println!(
            "Distribution des classes dans les données d'entraînement : Précipitation : {}, Clair : {}",
            precipitation_count, clear_count
        );

        for epoch in 0..self.epochs {
            let mut indices: Vec<usize> = (0..train_inputs.len()).collect();
            indices.shuffle(&mut rand::rng());

            let mut total_loss = 0.0;

            for batch_start in (0..indices.len()).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(indices.len());
                let batch_indices = &indices[batch_start..batch_end];

                for &idx in batch_indices {
                    let input = &train_inputs[idx];
                    let target = &train_outputs[idx];

                    let loss = network.backward(input, target, self.learning_rate);
                    total_loss += loss;
                }
            }

            let avg_loss = total_loss / train_inputs.len() as f32;

            let training_accuracy = self.evaluate_binary(network, &train_inputs, &train_outputs);
            let validation_accuracy = self.evaluate_binary(network, &valid_inputs, &valid_outputs);

            if epoch % 10 == 0 || epoch == self.epochs - 1 {
                println!(
                    "Époque {}/{} : Perte = {:.4}, Précision entraînement = {:.2}%, Précision validation = {:.2}%",
                    epoch + 1,
                    self.epochs,
                    avg_loss,
                    training_accuracy * 100.0,
                    validation_accuracy * 100.0
                );
            }

            if validation_accuracy > best_validation_accuracy {
                best_validation_accuracy = validation_accuracy;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience {
                    println!(
                        "Arrêt anticipé déclenché ! Pas d'amélioration pendant {} époques.",
                        patience
                    );
                    break;
                }
            }
        }

        best_validation_accuracy
    }

    fn evaluate_binary(
        &self,
        network: &NeuralNetwork,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> f32 {
        let mut correct = 0;
        let threshold = 0.5;

        for (i, input) in inputs.iter().enumerate() {
            let outputs = network.activate(input);
            let prediction = outputs.last().unwrap()[0];
            let target = targets[i][0];

            let predicted_class = if prediction >= threshold { 1.0 } else { 0.0 };

            if predicted_class == target {
                correct += 1;
            }
        }

        correct as f32 / inputs.len() as f32
    }
}

trait VecExt<T> {
    fn shuffle(&mut self, rng: &mut rand::rngs::ThreadRng);
}

impl<T> VecExt<T> for Vec<T> {
    fn shuffle(&mut self, rng: &mut rand::rngs::ThreadRng) {
        for i in (1..self.len()).rev() {
            let j = rng.random_range(0..=i);
            self.swap(i, j);
        }
    }
}
