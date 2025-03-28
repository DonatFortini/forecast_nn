use crate::layer::Layer;
use serde::{Deserialize, Serialize};

/// Represents a neural network composed of multiple layers.
///
/// ## Fields
/// - `layers`: A vector of `Layer` objects that make up the neural network.
///
/// ## Methods
///
/// ### `new`
/// Creates a new neural network with the specified layers.
///
/// #### Parameters:
/// - `layers`: A vector of `Layer` objects to initialize the neural network with.
///
/// ```rust
/// let network = NeuralNetwork::new(vec![layer1, layer2]);
/// ```
///
/// -------------------------------------
///
/// ### `add_layer`
/// Adds a new layer to the neural network.
///
/// #### Parameters:
/// - `layer`: The `Layer` object to add to the network.
///
/// ```rust
/// network.add_layer(new_layer);
/// ```
///
/// -------------------------------------
///
/// ### `remove_layer`
/// Removes a layer from the neural network by its ID.
///
/// #### Parameters:
/// - `layer_id`: The ID of the layer to remove.
///
/// ```rust
/// network.remove_layer(1);
/// ```
///
/// -------------------------------------
///
/// ### `get_layer`
/// Retrieves a reference to a layer by its ID.
///
/// #### Parameters:
/// - `layer_id`: The ID of the layer to retrieve.
///
/// ```rust
/// if let Some(layer) = network.get_layer(1) {
///     println!("Layer found: {:?}", layer);
/// }
/// ```
/// #### Returns:
/// An `Option` containing a reference to the layer, or `None` if not found.
///
/// -------------------------------------
///
/// ### `get_layer_mut`
/// Retrieves a mutable reference to a layer by its ID.
///
/// #### Parameters:
/// - `layer_id`: The ID of the layer to retrieve.
///
/// ```rust
/// if let Some(layer) = network.get_layer_mut(1) {
///     layer.name = "Updated Layer".to_string();
/// }
/// ```
/// #### Returns:
/// An `Option` containing a mutable reference to the layer, or `None` if not found.
///
/// -------------------------------------
///
/// ### `activate`
/// Activates the neural network by propagating the inputs through all layers.
///
/// #### Parameters:
/// - `inputs`: A slice of input values to feed into the network.
///
/// ```rust
/// let outputs = network.activate(&[1.0, 2.0]);
/// println!("Outputs: {:?}", outputs);
/// ```
/// #### Returns:
/// A vector of vectors, where each inner vector contains the outputs of a layer.
///
/// -------------------------------------
///
/// ### `get_layer_count`
/// Retrieves the number of layers in the neural network.
///
/// ```rust
/// let count = network.get_layer_count();
/// println!("Layer count: {}", count);
/// ```
/// #### Returns:
/// The number of layers in the network.
///
/// -------------------------------------
///
/// ### `get_layer_ids`
/// Retrieves the IDs of all layers in the neural network.
///
/// ```rust
/// let ids = network.get_layer_ids();
/// println!("Layer IDs: {:?}", ids);
/// ```
/// #### Returns:
/// A vector of layer IDs.
///
/// -------------------------------------
///
/// ### `get_layer_names`
/// Retrieves the names of all layers in the neural network.
///
/// ```rust
/// let names = network.get_layer_names();
/// println!("Layer names: {:?}", names);
/// ```
/// #### Returns:
/// A vector of layer names.
///
/// -------------------------------------
///
/// ### `get_layer_neuron_details`
/// Extracts specific details about neurons in all layers using a custom extractor function.
///
/// #### Parameters:
/// - `extractor`: A closure or function that takes a reference to a `Layer` and returns the desired detail.
///
/// ```rust
/// let neuron_counts = network.get_layer_neuron_details(|layer| layer.neurons.len());
/// println!("Neuron counts: {:?}", neuron_counts);
/// ```
/// #### Returns:
/// A vector of extracted details for each layer.
///
/// -------------------------------------
///
/// ### `set_layer_property`
/// Updates a property of a specific layer by its ID using a custom setter function.
///
/// #### Parameters:
/// - `layer_id`: The ID of the layer to update.
/// - `setter`: A closure or function that modifies the layer.
///
/// ```rust
/// network.set_layer_property(1, |layer| layer.name = "Updated Layer".to_string());
/// ```
///
/// -------------------------------------
///
/// ### `set_layer_neuron_property`
/// Updates a property of a specific neuron in a specific layer using a custom setter function.
///
/// #### Parameters:
/// - `layer_id`: The ID of the layer containing the neuron.
/// - `neuron_id`: The ID of the neuron to update.
/// - `setter`: A closure or function that modifies the neuron.
///
/// ```rust
/// network.set_layer_neuron_property(1, 2, |layer, neuron_id| {
///     if let Some(neuron) = layer.get_neuron_mut(neuron_id) {
///         neuron.bias = 0.5;
///     }
/// });
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        NeuralNetwork { layers }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn remove_layer(&mut self, layer_id: u32) {
        self.layers.retain(|layer| layer.id != layer_id);
    }

    pub fn get_layer(&self, layer_id: u32) -> Option<&Layer> {
        self.layers.iter().find(|&layer| layer.id == layer_id)
    }

    pub fn get_layer_mut(&mut self, layer_id: u32) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|layer| layer.id == layer_id)
    }

    pub fn activate(&self, inputs: &[f32]) -> Vec<Vec<f32>> {
        let mut outputs = Vec::new();
        let mut current_inputs = inputs.to_vec();

        for layer in &self.layers {
            let layer_output = layer.activate(&current_inputs);
            outputs.push(layer_output.clone());
            current_inputs = layer_output;
        }

        outputs
    }

    pub fn get_layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn get_layer_ids(&self) -> Vec<u32> {
        self.layers.iter().map(|layer| layer.id).collect()
    }

    pub fn get_layer_names(&self) -> Vec<String> {
        self.layers.iter().map(|layer| layer.name.clone()).collect()
    }

    pub fn get_layer_neuron_details<T, F>(&self, extractor: F) -> Vec<T>
    where
        F: Fn(&Layer) -> T,
    {
        self.layers.iter().map(extractor).collect()
    }

    pub fn set_layer_property<F>(&mut self, layer_id: u32, setter: F)
    where
        F: FnOnce(&mut Layer),
    {
        if let Some(layer) = self.get_layer_mut(layer_id) {
            setter(layer);
        }
    }

    pub fn set_layer_neuron_property<F>(&mut self, layer_id: u32, neuron_id: u32, setter: F)
    where
        F: FnOnce(&mut Layer, u32),
    {
        if let Some(layer) = self.get_layer_mut(layer_id) {
            setter(layer, neuron_id);
        }
    }
}
