use std::path::Path;

use forecast_nn::{dataset_loader, pickle, trainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Réseau de neurones pour la prévision météorologique (Classification binaire)");

    let train_data_path = Path::new("weather-train-dataset.json");
    let test_data_path = Path::new("weather-test-dataset.json");

    println!(
        "Chargement des données d'entraînement depuis {:?}",
        train_data_path
    );
    let train_data = dataset_loader::load_dataset(train_data_path)?;
    println!("Chargé {} exemples d'entraînement", train_data.len());

    println!("Chargement des données de test depuis {:?}", test_data_path);
    let test_data = dataset_loader::load_dataset(test_data_path)?;
    println!("Chargé {} exemples de test", test_data.len());

    println!("Conversion en classification binaire (précipitations vs. ciel dégagé)");
    let binary_train_data = dataset_loader::simplify_forecasts(&train_data);
    let binary_test_data = dataset_loader::simplify_forecasts(&test_data);

    println!("Normalisation des données");
    let (normalized_train, normalization_params) =
        dataset_loader::normalize_inputs(&binary_train_data);

    let (normalized_test, _) = dataset_loader::normalize_inputs(&binary_test_data);

    let train_precipitation = normalized_train.iter().filter(|d| d.output).count();
    let train_clear = normalized_train.len() - train_precipitation;
    println!(
        "Données d'entraînement : {} précipitations, {} ciel dégagé",
        train_precipitation, train_clear
    );

    let test_precipitation = normalized_test.iter().filter(|d| d.output).count();
    let test_clear = normalized_test.len() - test_precipitation;
    println!(
        "Données de test : {} précipitations, {} ciel dégagé",
        test_precipitation, test_clear
    );

    let trainer = trainer::BinaryTrainer::new(0.05, 1000, 20);

    let hidden_layers = vec![8, 4]; // Première couche cachée : 8 neurones, Deuxième : 4 neurones

    println!(
        "Création du réseau de neurones avec l'architecture : 4 -> {} -> {} -> 1",
        hidden_layers[0], hidden_layers[1]
    );

    let mut neural_network = trainer.create_weather_network(4, &hidden_layers);

    println!("Début de l'entraînement...");
    let accuracy = trainer.train(&mut neural_network, &normalized_train, &normalized_test);

    println!(
        "Entraînement terminé ! Précision finale : {:.2}%",
        accuracy * 100.0
    );

    let model_path = Path::new("weather_model.json");
    println!("Sauvegarde du modèle dans {:?}", model_path);
    pickle::save_model(&neural_network, &normalization_params, model_path)?;

    let sample_input = dataset_loader::WeatherInput {
        temp: 22.0,       // Température modérée
        pressure: 1016.0, // Pression moyenne
        altitude: 300.0,  // Altitude basse à moyenne
        humidity: 70.0,   // Humidité modérément élevée
    };

    println!("Prédiction pour : temp=22°C, pression=1016hPa, altitude=300m, humidité=70%");

    let normalized_input =
        dataset_loader::normalize_with_params(&sample_input, &normalization_params);

    let input_vector = vec![
        normalized_input.temp,
        normalized_input.pressure,
        normalized_input.altitude,
        normalized_input.humidity,
    ];

    let outputs = neural_network.activate(&input_vector);
    let prediction = outputs.last().unwrap()[0]; // Obtenir la valeur de sortie unique

    println!("Valeur brute de la prédiction : {:.4}", prediction);
    println!(
        "Prédiction binaire : {}",
        if prediction >= 0.5 {
            "Précipitations probables (pluie/averses)"
        } else {
            "Conditions dégagées (pas de précipitations)"
        }
    );

    Ok(())
}
