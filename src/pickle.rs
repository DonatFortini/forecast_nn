use crate::neural_network::NeuralNetwork;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct SavedModel {
    pub network: NeuralNetwork,
    pub normalization_params: [f32; 8],
}

pub fn save_model<P: AsRef<Path>>(
    network: &NeuralNetwork,
    normalization_params: &[f32; 8],
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let saved_model = SavedModel {
        network: network.clone(),
        normalization_params: *normalization_params,
    };

    let serialized = serde_json::to_string_pretty(&saved_model)?;

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    file.write_all(serialized.as_bytes())?;

    Ok(())
}

pub fn load_model<P: AsRef<Path>>(
    path: P,
) -> Result<(NeuralNetwork, [f32; 8]), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let saved_model: SavedModel = serde_json::from_str(&contents)?;

    Ok((saved_model.network, saved_model.normalization_params))
}
