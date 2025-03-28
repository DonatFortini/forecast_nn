use serde::{Deserialize, Serialize};

/// Represents a neuron in a neural network.
///
/// ## Fields
/// - `id`: A unique identifier for the neuron.
/// - `name`: The name of the neuron.
/// - `activation_function`: The activation function used by the neuron (e.g., "sigmoid", "relu").
/// - `bias`: The bias value added to the weighted sum of inputs.
/// - `weights`: The weights associated with the inputs to the neuron.
///
/// ## Methods
///
/// ### `new`
///  ``` rust
///     let neuron = Neuron::new(1, "Neuron1".to_string(), "sigmoid".to_string(), 0.5, vec![0.2, 0.3]);
/// ```   
///
/// -------------------------------------
///
/// ### `activate`
/// Computes the output of the neuron given a set of inputs.
///
/// #### Parameters:
/// - `inputs`: A slice of input values to the neuron.
/// ``` rust
/// let neuron = Neuron::new(1, "Neuron1".to_string(), "sigmoid".to_string(), 0.5, vec![0.2, 0.3]);
/// let inputs = vec![1.0, 2.0];
/// let output = neuron.activate(&inputs);
/// println!("Output: {}", output);
/// ```
/// #### Returns:
/// The output of the neuron after applying the activation function.
///
/// -----------------------------------
///
/// ### `apply_activation_function`
/// Applies the specified activation function to a given value.
///
/// #### Parameters:
/// - `value`: The input value to the activation function.
///
/// ``` rust
/// let neuron = Neuron::new(1, "Neuron1".to_string(), "sigmoid".to_string(), 0.5, vec![0.2, 0.3]);
/// let value = 0.5;
/// let activated_value = neuron.apply_activation_function(value);
/// println!("Activated Value: {}", activated_value);
/// ```
/// #### Returns:
/// The result of applying the activation function. Defaults to linear if the activation function is unknown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neuron {
    pub id: u32,
    pub name: String,
    pub activation_function: String,
    pub bias: f32,
    pub weights: Vec<f32>,
}

impl Neuron {
    pub fn new(
        id: u32,
        name: String,
        activation_function: String,
        bias: f32,
        weights: Vec<f32>,
    ) -> Self {
        Neuron {
            id,
            name,
            activation_function,
            bias,
            weights,
        }
    }

    pub fn activate(&self, inputs: &[f32]) -> f32 {
        let weighted_sum: f32 = inputs.iter().zip(&self.weights).map(|(x, w)| x * w).sum();
        self.apply_activation_function(weighted_sum + self.bias)
    }

    pub fn apply_activation_function(&self, value: f32) -> f32 {
        match self.activation_function.as_str() {
            "sigmoid" => 1.0 / (1.0 + (-value).exp()),
            "relu" => value.max(0.0),
            _ => value, // Default to linear if unknown
        }
    }
}
