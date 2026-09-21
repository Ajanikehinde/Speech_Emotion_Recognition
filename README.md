# Speech Emotion Recognition with RAVDESS

A machine learning project for classifying emotions from speech audio using acoustic features extracted with **Librosa** and an **MLP (Multi-Layer Perceptron) classifier**.

The project uses a four-emotion subset of the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset and combines **MFCC, chroma, and log-scaled mel-spectrogram features** for emotion classification.

> **Important:** The RAVDESS audio dataset is not included in this repository. The notebook expects the dataset to be downloaded separately and the `DATA_DIR` variable updated to its local location.

---

## Project Overview

Speech emotion recognition is the task of identifying a speaker's emotional state from their voice. It has applications in conversational AI, human-computer interaction, customer service analytics, accessibility, and intelligent voice systems.

This project extracts acoustic characteristics from speech recordings and uses them as input to a neural-network classifier.

The original implementation was revised to address several methodological issues, including:

* Lack of feature scaling before neural-network training
* Raw mel-spectrogram power having a substantially larger magnitude than other feature groups
* Non-reproducible model training
* Non-stratified train/test splitting
* Evaluation based only on a single accuracy value
* Lack of cross-validation
* Lack of per-class precision and recall
* Use of a machine-specific absolute dataset path

The revised pipeline addresses these issues while retaining the original project's overall approach.

---

## Objectives

The project aims to:

1. Load and process RAVDESS speech recordings.
2. Extract meaningful acoustic features from each audio file.
3. Combine MFCC, chroma, and mel-spectrogram representations.
4. Standardise the extracted features before neural-network training.
5. Train an MLP classifier to distinguish four emotions.
6. Evaluate model performance using:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion matrix
   * 5-fold stratified cross-validation
7. Compare model performance with and without feature scaling.

---

## Dataset

The project uses the **RAVDESS Audio Speech** dataset.

RAVDESS contains speech recordings from 24 professional actors. This project uses four emotions from the dataset:

| Emotion | RAVDESS Code |
| ------- | -----------: |
| Calm    |           02 |
| Happy   |           03 |
| Fearful |           06 |
| Disgust |           07 |

The original RAVDESS dataset contains eight emotion categories, but this project intentionally uses only the four classes above.

### Dataset Distribution

Before feature extraction, the selected classes contain 192 recordings per emotion. Two recordings were excluded because their duration exceeded the project's five-second limit.

The resulting dataset contained:

**766 usable audio samples**

| Emotion   | Samples |
| --------- | ------: |
| Calm      |     192 |
| Happy     |     192 |
| Fearful   |     191 |
| Disgust   |     191 |
| **Total** | **766** |

The resulting classes remain approximately balanced.

---

## Feature Engineering

Three acoustic feature groups are extracted from each audio recording.

### 1. MFCC

**Mel-Frequency Cepstral Coefficients (MFCCs)** capture characteristics of the speech spectrum associated with human auditory perception.

The project extracts:

```text
40 MFCC features
```

### 2. Chroma Features

Chroma features represent the distribution of spectral energy across the twelve pitch classes.

The project extracts:

```text
12 chroma features
```

### 3. Mel-Spectrogram Features

A mel-spectrogram represents the frequency content of the audio using the mel scale.

Instead of using raw mel-spectrogram power, the revised implementation converts the representation to decibels using:

```python
librosa.power_to_db()
```

This reduces the very large magnitude differences between mel-spectrogram values and the other feature groups.

### Final Feature Vector

The three feature groups are concatenated:

```text
40 MFCC
+ 12 Chroma
+ 128 Mel-spectrogram
-----------------------
180 features
```

Each usable audio recording therefore produces a **180-dimensional feature vector**.

---

## Audio Preprocessing

The audio is loaded using Librosa with a consistent target sample rate:

```python
TARGET_SR = 22050
```

This ensures that all recordings are processed using the same sampling frequency.

Audio recordings longer than five seconds are excluded:

```text
Maximum duration = 5 seconds
```

Two files were excluded during the recorded notebook run because they exceeded this threshold.

---

## Data Splitting

The dataset is divided into:

```text
75% Training
25% Testing
```

The split uses:

```python
random_state = 42
stratify = y
```

The recorded run produced:

| Dataset  | Samples |
| -------- | ------: |
| Training |     574 |
| Testing  |     192 |
| Total    |     766 |

Stratification preserves the relative class proportions between the training and test sets.

---

## Feature Scaling

Feature scaling is particularly important because MFCC, chroma, and mel-spectrogram features operate on substantially different numerical scales.

A `StandardScaler` is fitted only on the training data:

```python
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

The test set is transformed using the scaler fitted on the training set.

This prevents test-set statistics from influencing the training preprocessing.

The recorded feature standard deviation range changed from:

```text
Before scaling: 0.07 to 99.06
After scaling:  1.00 to 1.00
```

---

## Model

The primary classifier is an `MLPClassifier` from Scikit-learn.

The final configuration is:

```python
MLPClassifier(
    alpha=0.01,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(300,),
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)
```

The architecture contains one hidden layer with:

```text
300 neurons
```

---

## Scaling Comparison

The project directly compares the same MLP architecture using unscaled and scaled features.

| Configuration     | Test Accuracy |
| ----------------- | ------------: |
| Unscaled features |        40.10% |
| Scaled features   |        73.96% |

The difference is:

```text
73.96% - 40.10% = 33.86 percentage points
```

This experiment demonstrates the importance of feature scaling for this particular MLP pipeline.

---

## Cross-Validation

A **5-fold Stratified Cross-Validation** procedure is performed on the training set.

Feature scaling is refitted independently inside each fold to avoid validation leakage.

Recorded fold accuracies:

```text
Fold 1: 83.48%
Fold 2: 80.87%
Fold 3: 73.91%
Fold 4: 81.74%
Fold 5: 80.70%
```

Mean cross-validation accuracy:

```text
80.14% ± 3.27%
```

The standard deviation represents variation across the five validation folds.

---

## Held-Out Test Results

After training the final MLP on the scaled training data, performance was evaluated on the held-out test set.

### Overall Accuracy

```text
73.96%
```

The test set contained:

```text
192 samples
```

### Classification Report

| Emotion          | Precision |   Recall | F1-score | Support |
| ---------------- | --------: | -------: | -------: | ------: |
| Calm             |      0.82 |     0.88 |     0.85 |      48 |
| Disgust          |      0.85 |     0.69 |     0.76 |      48 |
| Fearful          |      0.65 |     0.81 |     0.72 |      48 |
| Happy            |      0.67 |     0.58 |     0.62 |      48 |
| **Macro Avg**    |  **0.75** | **0.74** | **0.74** | **192** |
| **Weighted Avg** |  **0.75** | **0.74** | **0.74** | **192** |

The results show that performance varies across emotion classes rather than being uniform across the four categories. `Calm` is the most reliably recognised emotion (0.85 F1); `happy` is the weakest (0.62 F1), driven by low recall rather than low precision — see the confusion matrix below for why.

---

## Confusion Matrix

The confusion matrix shows where the classifier's errors are concentrated, which the accuracy and classification-report numbers alone don't reveal.

```python
confusion_matrix(y_test, y_pred, labels=observed_emotions)
```

Running that cell produces the following breakdown (actual class in rows, predicted class in columns):

| Actual \\ Predicted | Calm | Happy | Fearful | Disgust |
| -------------------- | ---: | ----: | ------: | ------: |
| **Calm**              |   42 |     2 |       2 |       2 |
| **Happy**              |    2 |    28 |      16 |       2 |
| **Fearful**            |    1 |     6 |      39 |       2 |
| **Disgust**            |    6 |     6 |       3 |      33 |

**The dominant error is `happy` being misclassified as `fearful`: 16 of 48 `happy` test samples (33%) were predicted as `fearful` — by a wide margin the single largest off-diagonal count anywhere in the matrix.** This is the direct cause of `happy`'s low recall (0.58) in the classification report above; its precision (0.67) is comparatively less affected because the confusion runs mostly one-directional (only 2 `fearful` samples were mistaken for `happy`). Every other class is confused with its neighbours far less often (at most 6 samples in any other off-diagonal cell), so this specific `happy` → `fearful` boundary is where the model struggles most, and would be the natural place to focus future feature engineering or additional training data.

---

## Project Workflow

```text
RAVDESS Audio
      │
      ▼
Load WAV Files
      │
      ▼
Filter Four Emotion Classes
      │
      ▼
Audio Preprocessing
      │
      ├── Resample to 22,050 Hz
      └── Remove recordings > 5 seconds
      │
      ▼
Feature Extraction
      │
      ├── 40 MFCC
      ├── 12 Chroma
      └── 128 Mel Features
      │
      ▼
180-Dimensional Feature Vector
      │
      ▼
Stratified Train/Test Split
      │
      ├── Training: 574
      └── Testing: 192
      │
      ▼
StandardScaler
      │
      ▼
MLP Classifier
      │
      ├── Scaling comparison
      ├── 5-Fold Cross-Validation
      └── Held-Out Test Evaluation
      │
      ▼
Accuracy / Precision / Recall / F1
      │
      ▼
Confusion Matrix
```

---

## Technologies and Libraries

### Programming Language

* Python

### Libraries

* [Librosa](https://librosa.org/) — audio loading and feature extraction
* [NumPy](https://numpy.org/) — numerical computing
* [Scikit-learn](https://scikit-learn.org/) — preprocessing, modelling and evaluation
* [Matplotlib](https://matplotlib.org/) — visualisation
* [Seaborn](https://seaborn.pydata.org/) — confusion-matrix visualisation

### Main Machine Learning Components

* `StandardScaler`
* `MLPClassifier`
* `StratifiedKFold`
* `train_test_split`
* `classification_report`
* `confusion_matrix`

---

## Project Structure

```text
Speech_Emotion_Recognition/
│
├── Speech_Emotion_Recognition_corrected.ipynb
├── Speech Emotion Recognition_Document.docx
├── README.md
└── .gitignore
```

The RAVDESS dataset should be stored locally and **should not be committed to this repository**.

Example local structure:

```text
Speech_Emotion_Recognition/
│
├── data/
│   └── Audio_Speech_Actors_01-24/
│       ├── Actor_01/
│       ├── Actor_02/
│       ├── ...
│       └── Actor_24/
│
├── Speech_Emotion_Recognition_corrected.ipynb
├── Speech Emotion Recognition_Document.docx
└── README.md
```

---

## Installation

Create a Python environment and install the required libraries:

```bash
pip install librosa soundfile scikit-learn matplotlib seaborn numpy
```

---

## Dataset Setup

Download the **RAVDESS Audio Speech** dataset separately.

After extraction, update the notebook's `DATA_DIR` variable to point to the directory containing `Actor_01` through `Actor_24`.

For example:

```python
DATA_DIR = r"C:\path\to\Audio_Speech_Actors_01-24"
```

The expected structure is:

```text
Audio_Speech_Actors_01-24/
├── Actor_01/
├── Actor_02/
├── ...
└── Actor_24/
```

---

## Running the Project

1. Download and extract the RAVDESS Audio Speech dataset.
2. Set `DATA_DIR` in the notebook.
3. Install the required Python libraries.
4. Open:

```text
Speech_Emotion_Recognition_corrected.ipynb
```

5. Run the notebook from top to bottom.
6. Review the feature-scaling comparison, cross-validation results, classification report, and confusion matrix.

---

## Limitations

Several limitations should be considered when interpreting the results.

### Four-Class Scope

The project uses only four of RAVDESS's eight emotions:

```text
Calm
Happy
Fearful
Disgust
```

The remaining RAVDESS emotions are not included in the current classification task.

### Single Model

Only an `MLPClassifier` is evaluated in the current implementation. Other models such as SVM, Logistic Regression, Random Forest, or gradient-boosting methods could be compared using the same feature representation.

### Hyperparameter Tuning

The MLP hyperparameters were retained from the original project rather than extensively optimised.

### Sample Size

After filtering, the project uses 766 audio samples with 180 extracted features. This is a relatively small dataset for a neural-network-based classification problem.

### Sample Rate

The project uses 22,050 Hz as the target sample rate. This is a practical preprocessing choice but was not systematically compared against alternative sample rates.

### Class-Level Weakness

The `happy` class is the current model's weakest point (0.62 F1), specifically because of confusion with `fearful` (see Confusion Matrix above). Any future iteration should treat this pair as the priority, rather than treating overall accuracy as the only signal of progress.

### Dataset Availability

The RAVDESS audio files are not included in this GitHub repository. Users must obtain the dataset separately and configure `DATA_DIR`.

---

## Future Improvements

Potential improvements include:

* Extend classification from four to all eight RAVDESS emotions.
* Compare MLP with SVM, Logistic Regression, Random Forest and other classifiers.
* Perform systematic hyperparameter tuning.
* Experiment with alternative sample rates.
* Investigate additional audio features such as:

  * Zero-crossing rate
  * Spectral centroid
  * Spectral bandwidth
  * Spectral contrast
  * Tonnetz
* Compare traditional machine-learning approaches with CNN or RNN architectures.
* Apply data augmentation to increase training diversity.
* Investigate speaker-independent evaluation strategies.
* Use additional speech-emotion datasets to assess generalisation.
* Package the trained model into an inference application or API.

---

## Key Takeaways

This project demonstrates an end-to-end speech emotion classification workflow:

```text
Audio
→ Feature Engineering
→ Preprocessing
→ Feature Scaling
→ Neural Network
→ Cross-Validation
→ Test Evaluation
```

The recorded experiment achieved:

```text
73.96% held-out test accuracy
80.14% mean 5-fold CV accuracy
```

A direct comparison also showed that the scaled-feature pipeline substantially outperformed the same MLP using unscaled features on the recorded test split, and the confusion matrix shows that remaining errors are concentrated almost entirely in one direction: `happy` being mistaken for `fearful`.

These results should be interpreted specifically for the **four-class RAVDESS subset and the documented experimental setup**, rather than as a general benchmark for speech emotion recognition.

---

## Author

**Kehinde Azeez Ajani**

Data Science | Data Analytics | Machine Learning | AI

GitHub: [Ajanikehinde](https://github.com/Ajanikehinde)
