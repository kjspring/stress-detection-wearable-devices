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
3. Clean and process the raw data using the [Neurokit2](https://github.com/neuropsychology/NeuroKit) library. The processing includes filtering, finding peaks, and other necessary steps to prepare the data for further analysis and model training.
5. Selected differentiable features based on graphical analysis
	* 'EDA_Phasic'
6. **Standardization**: The data was standardized using the `MinMaxScaler` from the `sklearn.preprocessing` library. The scaler was applied to the computed features for each subject.
7. **Train-test split and downsampling**: The data was split into training, testing, and validation sets by subject. The dataset was then downsampled to a target frequency of 4 Hz. The train-test split was performed using a 80-20 ratio, and the remaining training data was further split into training and validation sets with a 75-25 ratio.
8. **Data generators**: Data generators were created using the `TimeseriesGenerator` from the `tensorflow.keras.preprocessing.sequence` library. The generators were created for training, validation, and testing data with a sequence length of 5 minutes and a batch size of 64.

After preprocessing, the dataset is ready for model training.

### Files

The following data files are used in this project:

* Raw Data: [WESAD.zip](https://drive.google.com/file/d/1q0WNZGxjuCOfEXhBeZcBbBtno8GI_sYA/view?usp=sharing)
* `WESAD_model_data.pickle`: Pickle file containing the processed dataset ready for machine learning training.

To repeat the preprocessing steps using the Jupyter Notebooks download the raw data file to your working directory.

## Model Training

Here's a markdown template for the "Model Training" section of your README:

---

## Model Training

In this project, we train several neural network models to predict stress based on various input features. Below is a brief description of each model and their respective training parameters.

### Model 0: Simple Dense Model

This model is a simple feed-forward neural network with two dense layers. The training parameters are as follows:

- Batch size: 64
- Learning rate: 1e-3
- Optimizer: Adam
- Loss function: Binary crossentropy
- Metrics: Binary accuracy
- Number of epochs: 200

### Model 1: LSTM Model

This model uses an LSTM layer followed by a dense layer. The training parameters are as follows:

- Batch size: 64
- Learning rate: 1e-3
- Optimizer: RMSprop
- Loss function: Binary crossentropy
- Metrics: Binary accuracy
- Number of epochs: 200

### Model 5: Convolutional Neural Network (CNN) Model

This model is a convolutional neural network with a 1D convolutional layer, a MaxPooling layer, and two dense layers. The training parameters are as follows:

- Batch size: 64
- Learning rate: 1e-3
- Optimizer: RMSprop
- Loss function: Binary crossentropy
- Metrics: Binary accuracy
- Number of epochs: 200

---

## Model Evaluation

The following models have been tested:

1. Model 0: Baseline Model - All predictions are class 1 (stressed)
2. Model 1: Simple LSTM Model
3. Model 5: LSTM with Attention Mechanism

The model evaluation is based on precision, accuracy, and recall scores for each model. The table below summarizes the results:

| Model | Train Accuracy | Train F1-Score | Validation Accuracy | Validation F1-Score | Test Accuracy | Test F1-Score |
|-------|----------------|----------------|---------------------|---------------------|---------------|---------------|
| 0     | 50.46%         | 67.07%         | 11.86%              | 21.20%              | 11.30%        | 20.30%        |
| 1     | 57.21%         | 34.98%         | 85.08%              | 8.40%               | 85.33%        | 6.97%         |
| 5     | 58.33%         | 69.47%         | 14.17%              | 21.02%              | 13.81%        | 20.21%        |

## Conclusion

The LSTM model (Model 1) performed best in terms of accuracy but had low F1-scores in the validation and test sets. Further analysis and improvements are needed to enhance the model's performance.

## Recommendations

1. Experiment with different architectures, such as bidirectional LSTMs, GRUs, or 1D convolutional layers.
2. Use techniques like oversampling or cost-sensitive learning to address class imbalance in the dataset.
3. Investigate feature importance and try feature selection or engineering methods to enhance model performance.
4. Perform hyperparameter tuning to optimize model performance.
5. Implement an ensemble of different models to improve overall predictions.

---

### Interpretation of Results

Based on the accuracy scores obtained from the evaluation, we can observe that:

- Model 5 (CNN) performs the best on both the validation and testing datasets, with the highest accuracy scores.
- Model 1 (LSTM) shows a slightly lower performance than Model 5, but it still outperforms Model 0.
- Model 0 (Dense) has the lowest accuracy scores among the three models.

These results suggest that the CNN-based Model 5 is the most suitable model for predicting stress in our dataset. However, the LSTM-based Model 1 is also a viable option, with only a small difference in performance.

---

(Summarize the results of your experiments, including evaluation metrics and any visualizations.)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
