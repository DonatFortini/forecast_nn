use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeatherInput {
    pub temp: f32,
    pub pressure: f32,
    pub altitude: f32,
    pub humidity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeatherOutput {
    pub forecast: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeatherDataPoint {
    pub input: WeatherInput,
    pub output: WeatherOutput,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimplifiedWeatherDataPoint {
    pub input: WeatherInput,
    pub output: bool, // true = precipitation, false = clear/dry
}

pub fn load_dataset<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<WeatherDataPoint>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let data = serde_json::from_reader(reader)?;
    Ok(data)
}

pub fn simplify_forecasts(dataset: &[WeatherDataPoint]) -> Vec<SimplifiedWeatherDataPoint> {
    dataset
        .iter()
        .map(|data_point| {
            let precipitation_keywords = [
                "pluie",
                "averse",
                "orage",
                "tonnerre",
                "précipitation",
                "neige",
                "rafales",
                "humide",
                "bruine",
                "humidité",
                "lourd",
            ];

            let has_precipitation = precipitation_keywords
                .iter()
                .any(|&keyword| data_point.output.forecast.to_lowercase().contains(keyword));

            SimplifiedWeatherDataPoint {
                input: data_point.input.clone(),
                output: has_precipitation,
            }
        })
        .collect()
}

pub fn normalize_inputs(
    dataset: &[SimplifiedWeatherDataPoint],
) -> (Vec<SimplifiedWeatherDataPoint>, [f32; 8]) {
    // Find min and max values for each feature
    let mut min_temp = f32::MAX;
    let mut max_temp = f32::MIN;
    let mut min_pressure = f32::MAX;
    let mut max_pressure = f32::MIN;
    let mut min_altitude = f32::MAX;
    let mut max_altitude = f32::MIN;
    let mut min_humidity = f32::MAX;
    let mut max_humidity = f32::MIN;

    for data_point in dataset {
        min_temp = min_temp.min(data_point.input.temp);
        max_temp = max_temp.max(data_point.input.temp);
        min_pressure = min_pressure.min(data_point.input.pressure);
        max_pressure = max_pressure.max(data_point.input.pressure);
        min_altitude = min_altitude.min(data_point.input.altitude);
        max_altitude = max_altitude.max(data_point.input.altitude);
        min_humidity = min_humidity.min(data_point.input.humidity);
        max_humidity = max_humidity.max(data_point.input.humidity);
    }

    let norm_params = [
        min_temp,
        max_temp,
        min_pressure,
        max_pressure,
        min_altitude,
        max_altitude,
        min_humidity,
        max_humidity,
    ];

    let normalized_dataset = dataset
        .iter()
        .map(|data_point| {
            let normalized_temp = (data_point.input.temp - min_temp) / (max_temp - min_temp);
            let normalized_pressure =
                (data_point.input.pressure - min_pressure) / (max_pressure - min_pressure);
            let normalized_altitude =
                (data_point.input.altitude - min_altitude) / (max_altitude - min_altitude);
            let normalized_humidity =
                (data_point.input.humidity - min_humidity) / (max_humidity - min_humidity);

            SimplifiedWeatherDataPoint {
                input: WeatherInput {
                    temp: normalized_temp,
                    pressure: normalized_pressure,
                    altitude: normalized_altitude,
                    humidity: normalized_humidity,
                },
                output: data_point.output,
            }
        })
        .collect();

    (normalized_dataset, norm_params)
}

pub fn normalize_with_params(input: &WeatherInput, params: &[f32; 8]) -> WeatherInput {
    let min_temp = params[0];
    let max_temp = params[1];
    let min_pressure = params[2];
    let max_pressure = params[3];
    let min_altitude = params[4];
    let max_altitude = params[5];
    let min_humidity = params[6];
    let max_humidity = params[7];

    WeatherInput {
        temp: (input.temp - min_temp) / (max_temp - min_temp),
        pressure: (input.pressure - min_pressure) / (max_pressure - min_pressure),
        altitude: (input.altitude - min_altitude) / (max_altitude - min_altitude),
        humidity: (input.humidity - min_humidity) / (max_humidity - min_humidity),
    }
}

pub fn prepare_inputs(dataset: &[SimplifiedWeatherDataPoint]) -> Vec<Vec<f32>> {
    dataset
        .iter()
        .map(|data_point| {
            vec![
                data_point.input.temp,
                data_point.input.pressure,
                data_point.input.altitude,
                data_point.input.humidity,
            ]
        })
        .collect()
}

pub fn prepare_outputs(dataset: &[SimplifiedWeatherDataPoint]) -> Vec<Vec<f32>> {
    dataset
        .iter()
        .map(|data_point| vec![if data_point.output { 1.0 } else { 0.0 }])
        .collect()
}
