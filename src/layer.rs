use crate::neuron::Neuron;
use serde::{Deserialize, Serialize};

/// Represents a layer in a neural network.
///
/// ## Fields
/// - `id`: A unique identifier for the layer.
/// - `name`: The name of the layer.
/// - `neurons`: A vector of neurons that belong to this layer.
///
/// ## Methods
///
/// ### `new`
/// Creates a new layer with the specified ID, name, and neurons.
///
/// #### Parameters:
/// - `id`: A unique identifier for the layer.
/// - `name`: The name of the layer.
/// - `neurons`: A vector of neurons to initialize the layer with.
///
/// ```rust
/// let layer = Layer::new(1, "Hidden Layer".to_string(), vec![]);
/// ```
///
/// -------------------------------------
///
/// ### `add_neuron`
/// Adds a neuron to the layer.
///
/// #### Parameters:
/// - `neuron`: The neuron to add to the layer.
///
/// ```rust
/// layer.add_neuron(neuron);
/// ```
///
/// -------------------------------------
///
/// ### `remove_neuron`
/// Removes a neuron from the layer by its ID.
///
/// #### Parameters:
/// - `neuron_id`: The ID of the neuron to remove.
///
/// ```rust
/// layer.remove_neuron(1);
/// ```
///
/// -------------------------------------
///
/// ### `get_neuron`
/// Retrieves a reference to a neuron by its ID.
///
/// #### Parameters:
/// - `neuron_id`: The ID of the neuron to retrieve.
///
/// ```rust
/// if let Some(neuron) = layer.get_neuron(1) {
///     println!("Neuron found: {:?}", neuron);
/// }
/// ```
/// #### Returns:
/// An `Option` containing a reference to the neuron, or `None` if not found.
///
/// -------------------------------------
///
/// ### `get_neuron_mut`
/// Retrieves a mutable reference to a neuron by its ID.
///
/// #### Parameters:
/// - `neuron_id`: The ID of the neuron to retrieve.
///
/// ```rust
/// if let Some(neuron) = layer.get_neuron_mut(1) {
///     neuron.bias = 0.5;
/// }
/// ```
/// #### Returns:
/// An `Option` containing a mutable reference to the neuron, or `None` if not found.
///
/// -------------------------------------
///
/// ### `activate`
/// Activates all neurons in the layer with the given inputs.
///
/// #### Parameters:
/// - `inputs`: A slice of input values to the neurons.
///
/// ```rust
/// let outputs = layer.activate(&[1.0, 2.0]);
/// println!("Outputs: {:?}", outputs);
/// ```
/// #### Returns:
/// A vector of outputs from all neurons in the layer.
///
/// -------------------------------------
///
/// ### `get_neuron_count`
/// Retrieves the number of neurons in the layer.
///
/// ```rust
/// let count = layer.get_neuron_count();
/// println!("Neuron count: {}", count);
/// ```
/// #### Returns:
/// The number of neurons in the layer.
///
/// -------------------------------------
///
/// ### `get_neuron_ids`
/// Retrieves the IDs of all neurons in the layer.
///
/// ```rust
/// let ids = layer.get_neuron_ids();
/// println!("Neuron IDs: {:?}", ids);
/// ```
/// #### Returns:
/// A vector of neuron IDs.
///
/// -------------------------------------
///
/// ### `get_neuron_names`
/// Retrieves the names of all neurons in the layer.
///
/// ```rust
/// let names = layer.get_neuron_names();
/// println!("Neuron names: {:?}", names);
/// ```
/// #### Returns:
/// A vector of neuron names.
///
/// -------------------------------------
///
/// ### `get_neuron_activation_functions`
/// Retrieves the activation functions of all neurons in the layer.
///
/// ```rust
/// let functions = layer.get_neuron_activation_functions();
/// println!("Activation functions: {:?}", functions);
/// ```
/// #### Returns:
/// A vector of activation function names.
///
/// -------------------------------------
///
/// ### `get_neuron_biases`
/// Retrieves the biases of all neurons in the layer.
///
/// ```rust
/// let biases = layer.get_neuron_biases();
/// println!("Neuron biases: {:?}", biases);
/// ```
/// #### Returns:
/// A vector of neuron biases.
///
/// -------------------------------------
///
/// ### `get_neuron_weights`
/// Retrieves the weights of all neurons in the layer.
///
/// ```rust
/// let weights = layer.get_neuron_weights();
/// println!("Neuron weights: {:?}", weights);
/// ```
/// #### Returns:
/// A vector of weight vectors for each neuron.
///
/// -------------------------------------
///
/// ### `set_neuron_weights`
/// Sets the weights of a specific neuron by its ID.
///
/// #### Parameters:
/// - `neuron_id`: The ID of the neuron to update.
/// - `weights`: The new weights to set.
///
/// ```rust
/// layer.set_neuron_weights(1, vec![0.1, 0.2]);
/// ```
///
/// -------------------------------------
///
/// ### `set_neuron_bias`
/// Sets the bias of a specific neuron by its ID.
///
/// #### Parameters:
/// - `neuron_id`: The ID of the neuron to update.
/// - `bias`: The new bias to set.
///
/// ```rust
/// layer.set_neuron_bias(1, 0.5);
/// ```
///
/// -------------------------------------
///
/// ### `set_neuron_activation_function`
/// Sets the activation function of a specific neuron by its ID.
///
/// #### Parameters:
/// - `neuron_id`: The ID of the neuron to update.
/// - `activation_function`: The new activation function to set.
///
/// ```rust
/// layer.set_neuron_activation_function(1, "relu".to_string());
/// ```
///
/// -------------------------------------
///
/// ### `set_neuron_name`
/// Sets the name of a specific neuron by its ID.
///
/// #### Parameters:
/// - `neuron_id`: The ID of the neuron to update.
/// - `name`: The new name to set.
///
/// ```rust
/// layer.set_neuron_name(1, "Updated Neuron".to_string());
/// ```
///
/// -------------------------------------
///
/// ### `set_neuron_id`
/// Updates the ID of a specific neuron.
///
/// #### Parameters:
/// - `old_id`: The current ID of the neuron.
/// - `new_id`: The new ID to assign to the neuron.
///
/// ```rust
/// layer.set_neuron_id(1, 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub id: u32,
    pub name: String,
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(id: u32, name: String, neurons: Vec<Neuron>) -> Self {
        Layer { id, name, neurons }
    }

    pub fn add_neuron(&mut self, neuron: Neuron) {
        self.neurons.push(neuron);
    }

    pub fn remove_neuron(&mut self, neuron_id: u32) {
        self.neurons.retain(|neuron| neuron.id != neuron_id);
    }

    pub fn get_neuron(&self, neuron_id: u32) -> Option<&Neuron> {
        self.neurons.iter().find(|&neuron| neuron.id == neuron_id)
    }

    pub fn get_neuron_mut(&mut self, neuron_id: u32) -> Option<&mut Neuron> {
        self.neurons
            .iter_mut()
            .find(|neuron| neuron.id == neuron_id)
    }

    pub fn activate(&self, inputs: &[f32]) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.activate(inputs))
            .collect()
    }

    pub fn get_neuron_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn get_neuron_ids(&self) -> Vec<u32> {
        self.neurons.iter().map(|neuron| neuron.id).collect()
    }

    pub fn get_neuron_names(&self) -> Vec<String> {
        self.neurons
            .iter()
            .map(|neuron| neuron.name.clone())
            .collect()
    }

    pub fn get_neuron_activation_functions(&self) -> Vec<String> {
        self.neurons
            .iter()
            .map(|neuron| neuron.activation_function.clone())
            .collect()
    }

    pub fn get_neuron_biases(&self) -> Vec<f32> {
        self.neurons.iter().map(|neuron| neuron.bias).collect()
    }

    pub fn get_neuron_weights(&self) -> Vec<Vec<f32>> {
        self.neurons
            .iter()
            .map(|neuron| neuron.weights.clone())
            .collect()
    }

    pub fn set_neuron_weights(&mut self, neuron_id: u32, weights: Vec<f32>) {
        if let Some(neuron) = self.neurons.iter_mut().find(|n| n.id == neuron_id) {
            neuron.weights = weights;
        }
    }

    pub fn set_neuron_bias(&mut self, neuron_id: u32, bias: f32) {
        if let Some(neuron) = self.neurons.iter_mut().find(|n| n.id == neuron_id) {
            neuron.bias = bias;
        }
    }

    pub fn set_neuron_activation_function(&mut self, neuron_id: u32, activation_function: String) {
        if let Some(neuron) = self.neurons.iter_mut().find(|n| n.id == neuron_id) {
            neuron.activation_function = activation_function;
        }
    }

    pub fn set_neuron_name(&mut self, neuron_id: u32, name: String) {
        if let Some(neuron) = self.neurons.iter_mut().find(|n| n.id == neuron_id) {
            neuron.name = name;
        }
    }

    pub fn set_neuron_id(&mut self, old_id: u32, new_id: u32) {
        if let Some(neuron) = self.neurons.iter_mut().find(|n| n.id == old_id) {
            neuron.id = new_id;
        }
    }
}
