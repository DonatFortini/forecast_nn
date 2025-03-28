use crate::layer::Layer;
use crate::neural_network::NeuralNetwork;
use crate::neuron::Neuron;

pub trait NeuronExt {
    fn calculate_gradient(&self, input: &[f32], target: f32, output: f32) -> f32;
    fn update_weights(&mut self, inputs: &[f32], gradient: f32, learning_rate: f32);
    fn calculate_derivative(&self, value: f32) -> f32;
}

impl NeuronExt for Neuron {
    fn calculate_gradient(&self, _input: &[f32], target: f32, output: f32) -> f32 {
        // For output neurons: gradient = (target - output) * derivative(output)
        let derivative = self.calculate_derivative(output);
        (target - output) * derivative
    }

    fn update_weights(&mut self, inputs: &[f32], gradient: f32, learning_rate: f32) {
        for (i, input) in inputs.iter().enumerate() {
            if i < self.weights.len() {
                self.weights[i] += learning_rate * gradient * input;
            }
        }
        self.bias += learning_rate * gradient;
    }

    fn calculate_derivative(&self, value: f32) -> f32 {
        match self.activation_function.as_str() {
            "sigmoid" => {
                // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
                value * (1.0 - value)
            }
            "relu" => {
                // Derivative of ReLU: 1 if x > 0, 0 otherwise
                if value > 0.0 { 1.0 } else { 0.0 }
            }
            // Default to linear derivative
            _ => 1.0,
        }
    }
}

pub trait LayerExt {
    fn forward_with_cache(&self, inputs: &[f32]) -> (Vec<f32>, Vec<f32>);
    fn backward(&mut self, inputs: &[f32], gradients: &[f32], learning_rate: f32) -> Vec<f32>;
}

impl LayerExt for Layer {
    fn forward_with_cache(&self, inputs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut outputs = Vec::with_capacity(self.neurons.len());
        let mut pre_activations = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons {
            let weighted_sum: f32 = inputs
                .iter()
                .zip(&neuron.weights)
                .map(|(x, w)| x * w)
                .sum::<f32>()
                + neuron.bias;

            pre_activations.push(weighted_sum);
            outputs.push(neuron.apply_activation_function(weighted_sum));
        }

        (outputs, pre_activations)
    }

    fn backward(&mut self, inputs: &[f32], gradients: &[f32], learning_rate: f32) -> Vec<f32> {
        let mut prev_layer_gradients = vec![0.0; inputs.len()];
        for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
            let gradient = gradients[neuron_idx];

            neuron.update_weights(inputs, gradient, learning_rate);

            for (input_idx, &weight) in neuron.weights.iter().enumerate() {
                if input_idx < prev_layer_gradients.len() {
                    prev_layer_gradients[input_idx] += gradient * weight;
                }
            }
        }

        prev_layer_gradients
    }
}

pub trait NetworkExt {
    fn forward_with_cache(&self, inputs: &[f32]) -> Vec<Vec<f32>>;
    fn backward(&mut self, inputs: &[f32], targets: &[f32], learning_rate: f32) -> f32;
}

impl NetworkExt for NeuralNetwork {
    fn forward_with_cache(&self, inputs: &[f32]) -> Vec<Vec<f32>> {
        let mut layer_outputs = Vec::new();
        let mut current_inputs = inputs.to_vec();
        layer_outputs.push(current_inputs.clone());

        for layer in &self.layers {
            let (layer_output, _) = layer.forward_with_cache(&current_inputs);
            layer_outputs.push(layer_output.clone());
            current_inputs = layer_output;
        }

        layer_outputs
    }

    fn backward(&mut self, inputs: &[f32], targets: &[f32], learning_rate: f32) -> f32 {
        let layer_outputs = self.forward_with_cache(inputs);

        let network_output = layer_outputs.last().unwrap();
        let mut loss = 0.0;
        for (output, target) in network_output.iter().zip(targets) {
            loss += 0.5 * (target - output).powi(2);
        }

        let mut next_gradients = Vec::with_capacity(network_output.len());

        for (i, (&output, &target)) in network_output.iter().zip(targets).enumerate() {
            let output_neuron = &self.layers.last().unwrap().neurons[i];
            let deriv = output_neuron.calculate_derivative(output);
            next_gradients.push((target - output) * deriv);
        }

        for layer_idx in (0..self.layers.len()).rev() {
            let layer_inputs = if layer_idx == 0 {
                inputs.to_vec()
            } else {
                layer_outputs[layer_idx].clone()
            };

            next_gradients =
                self.layers[layer_idx].backward(&layer_inputs, &next_gradients, learning_rate);
        }

        loss
    }
}
