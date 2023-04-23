# Stress Prediction using Wearable Device Data

This project aims to predict stress levels using data from wearable devices, leveraging the [WESAD dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29). We use various machine learning and deep learning models to classify the affective states of individuals as baseline or stress.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Model Training](#model-training)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Stress prediction is important for monitoring mental health and well-being. Using wearable devices, we can gather physiological and motion data to develop models that accurately forecast an individual's stress.

Add more about the negative effects of stress.

In this project, we preprocess and analyze the WESAD dataset, extract relevant features, and apply machine learning and deep learning techniques to predict affective states. Our goal is to provide a useful tool for researchers, developers, and health professionals to monitor and manage stress levels in real-time.

## File Directory



## Data Information

The dataset used in this project is the WESAD (Wearable Stress and Affect Detection) dataset, which is a multimodal dataset for the detection of stress using wearable physiological and motion sensors. The dataset includes data collected from various sensors placed on the subjects' chest and wrist, providing valuable information on Electrodermal Activity (EDA), Respiration (RSP), Heart Rate Variability (HRV), and other physiological signals. This section provides an overview of the dataset used for this  project. The description includes the dataset's features, labels, preprocessing steps, and the methods used to prepare the data for machine learning training.

### Dataset

* **Source**: The dataset used in this project is the WESAD (Wearable Stress and Affect Detection) dataset, a publicly available dataset for wearable stress and affect detection. The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29).
* **Subjects**: The dataset contains data from 15 subjects with varying demographics, including age, gender, and handedness.
* **Features**: Features are extracted using the neurokit2 python library
* **Labels**: The dataset contains binary labels representing stress (1) and non-stress (0) states.

### Features

The following features are available from the raw data:

- Chest data:
  - Accelerometer (ACC)
  - Electrocardiogram (ECG)
  - Electromyogram (EMG)
  - Electrodermal Activity (EDA)
  - Temperature (Temp)
  - Respiration (Resp)

- Wrist data:
  - Blood Volume Pulse (BVP)
  - Electrodermal Activity (EDA)
  - Temperature (TEMP)

#### Feature Selection
* Only EDA collected from the wrist was used to train the models.

### Labels

The dataset includes labels indicating the stress level of the subjects during the experiment. The labels are as follows:

- 0: No stress
- 1: Stress

### Preprocessing

The following preprocessing steps are performed on the raw data:

1. Import the raw data from pickle files and store them as `SubjectData` objects using the `subject_data_import` function.
2. Extract wrist data
3. Clean and process the raw data using the Neurokit2 library. The processing includes filtering, finding peaks, and other necessary steps to prepare the data for further analysis and model training.
5. Selected differentiable features based on graphical analysis
	* 'EDA_Phasic'
6. **Computing features**: The `compute_features` function was used to compute the mean and standard deviation of EDA over different time intervals (5 and 10 minutes) using the rolling function. This function was then applied to each dataframe in the list.
9. **Standardization**: The data was standardized using the `MinMaxScaler` from the `sklearn.preprocessing` library. The scaler was applied to the computed features for each subject.
10. **Train-test split and downsampling**: The data was split into training, testing, and validation sets by subject. The dataset was then downsampled to a target frequency of 4 Hz. The train-test split was performed using a 80-20 ratio, and the remaining training data was further split into training and validation sets with a 75-25 ratio.
11. **Data generators**: Data generators were created using the `TimeseriesGenerator` from the `tensorflow.keras.preprocessing.sequence` library. The generators were created for training, validation, and testing data with a sequence length of 5 minutes and a batch size of 64.

After preprocessing, the dataset is ready for model training.

### Files

The following data files are used in this project:

* `WESAD_model_data.pickle`: Pickle file containing the processed dataset ready for machine learning training.
* `WESAD_labels_model.pickle`: Pickle file containing the labels for the dataset.
* 

## Model Training

(Describe the models you've experimented with, how you selected features, and any hyperparameter tuning techniques used.)

## Results

(Summarize the results of your experiments, including evaluation metrics and any visualizations.)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



---

You can modify this template to better suit your project's specific requirements. Remember to replace placeholders with the actual information related to your project.