#[cfg(test)]
mod tests {

    use forecast_nn::dataset_loader::{self, SimplifiedWeatherDataPoint, WeatherInput};
    use forecast_nn::layer::Layer;
    use forecast_nn::neural_network::NeuralNetwork;
    use forecast_nn::neuron::Neuron;
    use forecast_nn::pickle;
    use std::path::Path;

    #[test]
    fn test_pickle() {
        let neuron1 = Neuron::new(
            1,
            "Test1".to_string(),
            "relu".to_string(),
            0.5,
            vec![0.1, 0.2],
        );
        let neuron2 = Neuron::new(
            2,
            "Test2".to_string(),
            "sigmoid".to_string(),
            0.3,
            vec![0.4, 0.5],
        );

        let layer = Layer::new(1, "TestLayer".to_string(), vec![neuron1, neuron2]);
        let network = NeuralNetwork::new(vec![layer]);

        let norm_params = [0.0, 100.0, 1000.0, 1030.0, 0.0, 1500.0, 0.0, 100.0];
        let test_path = Path::new("test_model.json");
        let save_result = pickle::save_model(&network, &norm_params, test_path);
        assert!(
            save_result.is_ok(),
            "Échec de la sauvegarde du modèle : {:?}",
            save_result.err()
        );

        let load_result = pickle::load_model(test_path);
        assert!(
            load_result.is_ok(),
            "Échec du chargement du modèle : {:?}",
            load_result.err()
        );

        std::fs::remove_file(test_path).unwrap_or(());
    }

    #[test]
    fn test_model_prediction() {
        let model_path = Path::new("weather_model.json");
        if !model_path.exists() {
            println!("Fichier modèle introuvable, test de prédiction ignoré");
            return;
        }

        let load_result = pickle::load_model(model_path);
        assert!(
            load_result.is_ok(),
            "Échec du chargement du modèle : {:?}",
            load_result.err()
        );

        let (network, norm_params) = load_result.unwrap();
        let test_cases = [
            (
                WeatherInput {
                    temp: 30.0,
                    pressure: 1008.0,
                    altitude: 50.0,
                    humidity: 85.0,
                },
                Some(true),
            ),
            (
                WeatherInput {
                    temp: 5.0,
                    pressure: 1025.0,
                    altitude: 1000.0,
                    humidity: 30.0,
                },
                Some(false),
            ),
            (
                WeatherInput {
                    temp: 20.0,
                    pressure: 1015.0,
                    altitude: 300.0,
                    humidity: 60.0,
                },
                None,
            ),
        ];

        for (i, (input, expected)) in test_cases.iter().enumerate() {
            let normalized_input = dataset_loader::normalize_with_params(input, &norm_params);

            let input_vector = [
                normalized_input.temp,
                normalized_input.pressure,
                normalized_input.altitude,
                normalized_input.humidity,
            ];

            let outputs = network.activate(&input_vector);
            let prediction = outputs.last().unwrap()[0];
            let binary_prediction = prediction >= 0.5;

            println!(
                "Cas de test {}: Entrée={:?}, Prédiction brute={:.4}, Binaire={}",
                i + 1,
                input,
                prediction,
                binary_prediction
            );

            match expected {
                Some(expected_result) => {
                    assert_eq!(
                        binary_prediction,
                        *expected_result,
                        "Cas de test {} échoué : attendu {}, obtenu {}",
                        i + 1,
                        expected_result,
                        binary_prediction
                    );
                }
                None => {
                    println!("Cas de test {}: La prédiction est incertaine", i + 1);
                }
            }
        }
    }

    #[test]
    fn test_binary_classification() {
        let test_data = vec![
            SimplifiedWeatherDataPoint {
                input: WeatherInput {
                    temp: 0.8,
                    pressure: 0.3,
                    altitude: 0.2,
                    humidity: 0.9,
                },
                output: true,
            },
            SimplifiedWeatherDataPoint {
                input: WeatherInput {
                    temp: 0.2,
                    pressure: 0.8,
                    altitude: 0.7,
                    humidity: 0.1,
                },
                output: false,
            },
        ];

        let inputs = dataset_loader::prepare_inputs(&test_data);
        assert_eq!(inputs.len(), 2, "Attendu 2 vecteurs d'entrée");
        assert_eq!(inputs[0].len(), 4, "Attendu 4 caractéristiques par entrée");

        let outputs = dataset_loader::prepare_outputs(&test_data);
        assert_eq!(outputs.len(), 2, "Attendu 2 vecteurs de sortie");
        assert_eq!(
            outputs[0][0], 1.0,
            "Le premier échantillon devrait être de classe positive"
        );
        assert_eq!(
            outputs[1][0], 0.0,
            "Le deuxième échantillon devrait être de classe négative"
        );
    }
}
