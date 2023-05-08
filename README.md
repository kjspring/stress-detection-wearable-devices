# Stress Prediction using Wearable Device Data

This project aims to predict stress levels using data from wearable devices, leveraging the [WESAD dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29). We use various machine learning and deep learning models to classify the affective states of individuals as baseline or stress.

## Table of Contents

1. [Introduction](#introduction)
2. [Business Problem](#business-problem)
3. [File Directory](#file-directory)
4. [Data](#data)
5. [Model Evaluation](#model-evaluation)
6. [Conclusions](#conclusions)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Stress prediction is important for monitoring mental health and well-being. Using wearable devices, we can gather physiological and motion data to develop models that accurately forecast an individual's stress.

Stress can have detrimental effects on both mental and physical health. When individuals experience prolonged or chronic stress, it can contribute to various mental health disorders such as anxiety, depression, and burnout. Additionally, stress can become physical symptoms and lead to conditions such as cardiovascular diseases, gastrointestinal problems, weakened immune system, and impaired cognitive functioning. Furthermore, chronic stress can exacerbate existing health conditions and hinder the recovery process.

Stress is a prevalent issue in society, and a significant portion of the population experiences stressful events each year. A substantial portion of the population encounters stressful events within a given year. These events can range from work-related stress, financial pressures, relationship difficulties, health concerns, and other life challenges.

Recognizing the significant economic costs, high prevalence, and negative impacts on individuals, it is crucial to prioritize stress management and promote strategies that reduce stress levels.

In this project, we preprocess and analyze the WESAD dataset, extract relevant features, and apply machine learning and deep learning techniques to predict affective states. Our goal is to provide a useful tool for researchers, developers, and health professionals to monitor and manage stress levels in real-time.

## Business Problem
HealthOn is focused on improving the health of users through the development of wearable devices. The goal is to collect and analyze physiological data in real time to provide insights and interventions to improve users health. 

Stress is a significant health related burden for individuals. It can lead to mental and somatic health issues. Changes in the body are correlated when a person is going into a stressful situation, for example, they will sweat more and their heart rate will increase.

We want to forecast if a user will be in a stressful state in the immediate future. Our device will then notify the user with recommendations to mitigate the stess.

### Project Objective

This project aims to assess the feasability of using physiological data to forecast a subject going into stress. It also aims to identify the most import features in detecting stress for the aim of improving wearable device development and design.

### Stakeholders
- Project Manager
- Business executives

## File Directory
* [/deliverables](../deliverables)
* [/img](../img)
* [/notebooks](../notebooks)
    * data_load_process.ipynb: Load and preprocess raw data
    * modeling_WESAD_all_subjects.ipynb: Process data and model

## Data

The dataset used in this project is the WESAD (Wearable Stress and Affect Detection) dataset, which is a multimodal dataset for the detection of stress using wearable physiological and motion sensors. The dataset includes data collected from various sensors placed on the subjects' chest and wrist, providing valuable information on Electrodermal Activity (EDA), Respiration (RSP), Heart Rate Variability (HRV), and other physiological signals. This section provides an overview of the dataset used for this  project. The description includes the dataset's features, labels, preprocessing steps, and the methods used to prepare the data for machine learning training.

### Raw dataset

* **Source**: The dataset used in this project is the [WESAD (Wearable Stress and Affect Detection) dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29), a publicly available dataset for wearable stress and affect detection.
* **Subjects**: The dataset contains data from 15 subjects with varying demographics, including age, gender, and handedness.
* **Features**: Features are extracted using the neurokit2 python library
* **Labels**: The processed dataset contains binary labels representing stress (1) and non-stress (0) states.

The following data files are used in this project:

* Raw Data: [WESAD.zip](https://drive.google.com/file/d/1q0WNZGxjuCOfEXhBeZcBbBtno8GI_sYA/view?usp=sharing)
* 
* [WESAD_model_data.pickle](https://drive.google.com/file/d/1KuIc5ypFB30McV7tl3ziwQndEIjvtgBJ/view?usp=share_link): Pickle file containing the processed dataset ready for machine learning training.

To repeat the preprocessing steps using the Jupyter Notebooks download the raw data file to your working directory.

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
* Data Preparation: Used the NeuroKit library to preprocess and clean the physiological data. The bio_process() function in NeuroKit allows the  automated processing of different signals, such as heart beat (ECG), respiration (RSP), electrodermal activity (EDA), and electromyography (EMG). This step involved removing noise, filtering the signals, and extracting relevant features.

* Physiological Feature Extraction: With Neurokit various physiological features from the preprocessed data was extracted. Obtained features such as heart rate, R peaks indexes, respiration rate, amplitude, tonic and phasic components of EDA. These features provide valuable insights into the individual's physiological state.

* Statistical Feature Extraction: In addition to physiological features, statistical features from the processed data was extracted. This step involved calculating statistical measures such as the median and standard deviation over 1 and 5 minute intervals. These statistical features help capture the distribution and variability of the data, providing further insights into the individual's physiological profile.

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

## Model Evaluation

In this project, we train several machine learning and neural network models to predict stress based on the processed input features. Below is a table showing the testing metrics for each model.

| Model  | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
|--------|----------|---------------------|------------------|--------------------|
| Model 0|   0.905  |         0.661           |        0.554            |       0.603          |
| Model 1|   0.829  |         0.402           |        0.642            |       0.494          |
| Model 2|   0.790  |         0.343           |        0.676            |       0.455          |
| Model 3|   0.647  |         0.130           |        0.303            |       0.182          |
| Model 4|   0.870  |         0.000           |        0.000            |       0.000          |
| Model 5|   0.870  |         0.000           |        0.000            |       0.000          |
| Model 6|   0.846  |         0.073           |        0.016            |       0.026          |
| Model 7|   0.130  |         0.130           |        1.000            |       0.230          |
| Model 8|   0.870  |         0.000           |        0.000            |       0.000          |
| Model 9|   0.870  |         0.000           |        0.000            |       0.000          |
| Model 10|   0.910 |         0.630           |        0.670            |       0.650          |
| Model 11|   0.840 |         0.420           |        0.700            |       0.530          |       |

## Conclusion

Physiological data, such as heart rate, electrodermal activity (EDA), body temperature, and respiration rate, can provide valuable insights into the emotional and physical state of a person. These physiological parameters are influenced by the autonomic nervous system, which regulates the body's response to various stimuli and can reflect changes in emotional arousal, stress levels, and physical well-being.

* Heart Rate: Heart rate is the number of times the heart beats per minute and is influenced by factors such as physical exertion, stress, and emotional arousal. Higher heart rate can indicate increased physiological arousal, which may be associated with emotions like excitement, anxiety, or fear. Changes in heart rate variability (HRV), the variation in time intervals between heartbeats, can also provide information about emotional regulation and stress levels.

* Electrodermal Activity (EDA): EDA measures the electrical conductance of the skin, which is influenced by sweat gland activity. EDA is commonly used as an indicator of sympathetic nervous system activity, which is associated with emotional arousal and stress. Increased EDA may reflect heightened emotional responses, such as excitement, fear, or anxiety.

* Body Temperature: Body temperature can fluctuate based on environmental conditions, physical activity, and emotional states. Increased body temperature may occur during periods of physical exertion, stress, or emotional arousal. Conversely, decreased body temperature may indicate relaxation or a lower emotional state.

* Respiration Rate: Respiration rate refers to the number of breaths taken per minute. Emotional and physical states can impact respiration patterns. For instance, during states of stress or anxiety, respiration rate may increase, leading to rapid and shallow breathing. In contrast, during calm or relaxed states, respiration rate tends to be slower and deeper.

Machine learning algorithms and statistical techniques can be applied to these data to develop models that predict emotional states, stress levels, or physical conditions, such as stress forecasting.

Based on the F1-score for the stress case, the best models to forecast if a person will be in a stressful state in the next 5 minutes is with model 10, the baseline model 0, and model 11.

Model 10 (XGBoost) and Model 0 (fully connected neural network) are considered to be only slightly better than guessing because their performance metrics, such as accuracy, precision, recall, and F1-score, are not significantly higher than random chance. In the context of stress prediction, these models exhibit limited predictive power and may not provide reliable or accurate forecasts.

For example, if we look at Model 10's F1-score for stress (class 1), it is 0.65, indicating that the model can correctly identify only 65% of the stressed instances in the dataset. Similarly, Model 0's F1-score for stress is 0.60, which means it can correctly identify 60% of the stressed instances. These scores suggest that the models are not performing significantly better than randomly assigning labels, which is not suitable for reliable stress prediction.

Additionally, the performance of these models may be attributed to the limited dataset used for training. Both models were trained on a specific dataset (WESAD dataset) with a limited number of subjects and potentially limited variability in stressful conditions. To build more robust and accurate stress prediction models, a larger and more diverse dataset is necessary. This would involve collecting data from a broader range of individuals, encompassing various stress-inducing situations and conditions.

To gather more data, our next generation of wearable devices should be equipped with appropriate sensors to measure physiological parameters like respiration rate, electrodermal activity (EDA), heart rate, and body temperature. These sensors can provide a more comprehensive and reliable set of inputs for stress prediction models. The user can indicate on the device if they are experiencing stress and that data can be used for further training. Also the device can predict if a user is in stress or will be in stress and ask for the user's feedback on their stressful state.By incorporating additional features with high causal relationships to stress, the models can potentially improve in their ability to accurately forecast stress.

In summary, while the results from Model 10 and Model 0 indicate the feasibility of stress prediction, their limited performance and the need for a larger dataset suggest that they are not appropriate for deployment in production. Expanding the dataset and developing devices with suitable sensors would be crucial steps in enhancing the accuracy and reliability of stress prediction models.

## Recommendations
1. Enhance Feature Measurement: In order to improve the accuracy of stress forecasting, it is recommended to focus on measuring physiological features that have a high causality with stress. Specifically, consider incorporating measurements such as respiration rate, heart rate variability, body temperature, and electrodermal activity. These features have been found to be closely linked to stress responses and can provide valuable insights for stress prediction models.

2. Expand Data Collection: To further improve the forecasting models, it is crucial to gather a more diverse and comprehensive dataset. Collecting data from a larger sample size of individuals, particularly in both stressful and normal conditions, will allow for a better understanding of the variations and patterns associated with stress. Encourage voluntary data collection from users, ensuring privacy and consent, to increase the dataset's size and diversity.

3. Focus on Stressful Conditions: To specifically address stress forecasting, it is important to prioritize data collection during high-stress situations or events. This can be achieved by designing studies or collecting data from individuals undergoing stressful experiences, such as work-related stress, performance anxiety, or challenging life events. This targeted data collection will help train the models to better identify and predict stress states accurately.

4. Continuous Model Improvement: As stress forecasting is a complex task, it is essential to continuously refine and enhance the machine learning models. Regularly analyze the performance of the models, identify areas for improvement, and iterate on the algorithms and techniques used. As more data becomes available and the models evolve, periodically reassess their performance and implement necessary updates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
