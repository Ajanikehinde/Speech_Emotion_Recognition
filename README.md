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
* No check that evaluation honoured speaker boundaries, which lets the same actors appear in both training and validation data

The revised pipeline addresses these issues while retaining the original project's overall approach, and adds a **speaker-generalization experiment** that quantifies how much of the reported accuracy depends on that speaker overlap.

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
8. Compare random stratified cross-validation with **actor-grouped** cross-validation, to measure how much speaker overlap inflates the reported accuracy.

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

## Speaker-Generalization Evaluation

Stratified splitting preserves the class balance, but it does not preserve **speaker** balance. RAVDESS has only 24 actors, and each actor contributes roughly 32 recordings, so a random fold split almost guarantees that every actor appears in both the training and validation portion. That gives a model the opportunity to exploit speaker-specific cues — vocal tract shape, timbre, recording level — rather than the emotion itself, which is the main reason a random-split accuracy may not transfer to a voice the model has never heard. The experiment below tests for that effect rather than assuming it.

The project therefore evaluates the **same features and the same model** under two different splitting rules. The split rule is the only thing that changes:

```python
make_model()   # StandardScaler -> MLPClassifier, identical hyperparameters and seed

cross_val_score(make_model(), x, y,
                cv=StratifiedKFold(6, shuffle=True, random_state=42))

cross_val_score(make_model(), x, y, groups=groups,
                cv=GroupKFold(n_splits=6))
```

`groups` is the actor ID parsed from the RAVDESS filename (the last dash-separated field), giving 24 distinct groups.

### Results

| Splitting rule                                   | Mean accuracy | Std. dev. |
| ------------------------------------------------ | ------------: | --------: |
| Random stratified 6-fold (speakers overlap)       |    **82.89%** |     2.33% |
| Actor-grouped 6-fold (speakers disjoint)          |    **54.58%** |     7.91% |

```text
82.89% - 54.58% = 28.31 percentage points
```

When no actor appears in the fold that evaluates their own recordings, mean accuracy falls by **28.31 percentage points** — from **82.89% to 54.58%**, about **34%** relative. This should be read as evidence that the random-split figure is **optimistic when actors appear on both sides of a split**, not as a measurement of how much accuracy is caused by speaker identity. The grouping changes more than who has been heard before: it also changes which recordings share a fold and which recording conditions are held out, so the decline cannot be attributed to speaker identity specifically.

Against the four-class random-guess floor of 25%, the grouped result remains well above chance, so the acoustic features do carry emotion information — just not enough to support the higher, speaker-overlapping figure as an estimate of performance on an unseen voice.

### Per-Fold Actor-Grouped Accuracies

Because only 24 actors are available, six folds means **roughly four actors per fold** (approximately 128 recordings per fold) — the exact assignment is decided by `GroupKFold` balancing group sizes, not by a fixed number of actors. That is a small, coarse unit of evaluation, and the fold-to-fold spread is part of the result rather than noise to be averaged away:

| Fold | Accuracy |
| ---- | -------: |
| 1    |   56.25% |
| 2    |   41.41% |
| 3    |   48.44% |
| 4    |   66.41% |
| 5    |   59.06% |
| 6    |   55.91% |
| Mean |   54.58% ± 7.91% |

The 25-point range between the best fold (66.41%) and the worst (41.41%) is driven by *which actors* were held out in each fold, which is why the standard deviation (±7.91%) is reported next to the mean rather than the mean alone.

### Held-Out-Actor Test Set

A second, independent check holds out entire actors rather than folds of actors:

```python
GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(x, y, groups)

assert set(groups[train_idx]).isdisjoint(groups[test_idx])   # no actor in both sets
```

Six actors were held out — `1, 9, 10, 12, 17, 19` — giving 190 test recordings. Accuracy on that set was **55%** (macro F1 0.54), consistent with the grouped cross-validation estimate.

| Emotion      | Precision |  Recall | F1-score | Support |
| ------------ | --------: | ------: | -------: | ------: |
| Calm         |      0.62 |    0.71 |     0.66 |      48 |
| Disgust      |      0.69 |    0.38 |     0.49 |      47 |
| Fearful      |      0.48 |    0.66 |     0.56 |      47 |
| Happy        |      0.47 |    0.44 |     0.45 |      48 |
| **Macro Avg**|  **0.57** |**0.55** | **0.54** | **190** |

Confusion matrix (rows = actual, columns = predicted, in `observed_emotions` order):

| Actual \\ Predicted | Calm | Happy | Fearful | Disgust |
| -------------------- | ---: | ----: | ------: | ------: |
| **Calm**              |   34 |     9 |       4 |       1 |
| **Happy**             |    6 |    21 |      18 |       3 |
| **Fearful**           |    6 |     6 |      31 |       4 |
| **Disgust**           |    9 |     9 |      11 |      18 |

### Class-Level Effect of Removing Speaker Overlap

Comparing the per-class F1 scores with the speaker-overlapping test set is more informative than the accuracy drop alone:

| Emotion | F1, speakers overlap | F1, speakers unseen | Change |
| ------- | -------------------: | ------------------: | -----: |
| Calm    |                 0.85 |                0.66 |  −0.19 |
| Disgust |                 0.76 |                0.49 |  −0.27 |
| Fearful |                 0.72 |                0.56 |  −0.16 |
| Happy   |                 0.62 |                0.45 |  −0.17 |

Every class degrades, and the ordering changes: **disgust** falls furthest (−0.27), from the second-strongest class to the second-weakest, and its recall collapses to 0.38 — 11 of its 47 recordings are predicted as `fearful` and 9 as `calm`. **Calm** remains the strongest class under both protocols, so that result is not an artefact of speaker overlap.

The `happy` → `fearful` confusion identified in the confusion matrix above persists with unseen speakers (18 of 48, 37.5%), which makes it the one error pattern that survives the stricter evaluation. That is the pair a future iteration should target.

### Leakage Controls

The comparison is only meaningful if the surrounding preprocessing does not leak. Three controls are in place:

1. **Scaling is fitted inside each fold.** The model is a `make_pipeline(StandardScaler(), MLPClassifier(...))`, so the scaler is refitted on each fold's training portion and only ever *applied* to the held-out portion. No validation statistics reach the scaler.
2. **Actor groups are disjoint by construction and by assertion.** `GroupKFold` never splits an actor across folds, and the held-out split is verified with `isdisjoint` before it is scored.
3. **Nothing else differs between the two arms.** Same 180-dimensional feature vectors, same hyperparameters, same `random_state`; only the splitting rule changes, so any difference between the two figures is associated with the split rather than with a change in features, model or tuning.

### What the Gap Does and Does Not Show

The 28-point gap is best described as **evidence of optimistic random-split evaluation under actor overlap**. It is not a causal decomposition and it is not a significance test. In particular:

* **No significance test was performed.** Both arms score the same 766 recordings and differ only in how folds are formed, so the comparison is descriptive. The `±` values quoted above are fold-to-fold standard deviations, not standard errors; no p-value or confidence interval is reported, and nothing here should be read as establishing that the difference is statistically significant.
* **The gap does not measure the speaker effect.** Grouping by actor also changes fold composition, per-fold class balance and which recording conditions are held out, so the decline cannot be attributed to speaker identity alone.
* **Speaker identity is confounded with actor-correlated nuisance factors.** Each actor was recorded in one session with one microphone, so "unseen speaker" also means "unseen channel and session". Some of the drop may be acoustic-domain shift rather than speaker physiology.
* **Six folds over 24 actors is coarse.** Roughly four actors per fold, one `GroupShuffleSplit` draw, no confidence interval on the 55% held-out figure. The ±7.91% spread is a better guide to uncertainty than the mean.
* **`GroupKFold` is deterministic and non-shuffled, but its folds are not simply sequential actor IDs.** The splitter assigns groups by balancing the number of samples per fold, so the same data always produces the same actor-disjoint folds — but the composition follows that balancing rule rather than ascending actor order, and it is not randomised. Repeatedly re-drawing the grouping would separate "which actors" from "how many actors".
* **The grouped folds are not stratified.** Class counts per fold vary slightly (47 vs 48 in the held-out set), so a small part of the fold variance is class balance rather than speaker difficulty.
* **One model, one feature set, one seed.** This measures the sensitivity of *this* pipeline to the splitting rule; it does not establish that 54.58% is the generalization ceiling for speech emotion recognition on RAVDESS.

The defensible claim is therefore narrow: **for this four-class RAVDESS subset, with these features and this MLP, the random stratified figure of 82.89% is optimistic — actors appear on both sides of every split — and when actors are held out, mean accuracy falls to 54.58%.** The experiment establishes the direction and rough size of that evaluation gap. It does not isolate speaker identity as the cause of the decline, and it does not show that the gap is statistically significant.

---

## Held-Out Test Results

After training the final MLP on the scaled training data, performance was evaluated on the held-out test set.

> **Reading these numbers correctly:** this section uses the random, stratified 75/25 split from *Data Splitting*, so the same actors appear in both the training and test halves. It measures generalisation to unseen **recordings**, not unseen **speakers**. The actor-disjoint result is in *Speaker-Generalization Evaluation* above and is substantially lower.

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

**The dominant error is `happy` being misclassified as `fearful`: 16 of 48 `happy` test samples (33%) were predicted as `fearful` — by a wide margin the single largest off-diagonal count anywhere in the matrix.** This is the direct cause of `happy`'s low recall (0.58) in the classification report above; its precision (0.67) is comparatively less affected because the confusion runs mostly one-directional (6 `fearful` samples were mistaken for `happy`, under half as many). Every other class is confused with its neighbours far less often (at most 6 samples in any other off-diagonal cell), so this specific `happy` → `fearful` boundary is where the model struggles most, and would be the natural place to focus future feature engineering or additional training data.

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
      ├── Held-Out Test Evaluation
      └── Random vs. Actor-Grouped 6-Fold CV
      │
      ▼
Accuracy / Precision / Recall / F1
      │
      ▼
Confusion Matrix
      │
      ▼
Speaker-Generalization Gap
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
* `GroupKFold`
* `GroupShuffleSplit`
* `cross_val_score`
* `make_pipeline`
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

The `happy` class is the current model's weakest point (0.62 F1), specifically because of confusion with `fearful` (see Confusion Matrix above). That weakness is not an artefact of speaker overlap: with actors held out entirely, `happy` still scores 0.45 F1 and still sends 18 of 48 recordings to `fearful`. Any future iteration should treat this pair as the priority, rather than treating overall accuracy as the only signal of progress.

### Speaker Overlap in the Default Protocol

The headline 73.96% held-out accuracy, and the 80.14% mean 5-fold CV accuracy, both come from splits in which the same actors appear in the training and evaluation data. Under actor-grouped evaluation the same pipeline scores 54.58%. The higher numbers are not wrong, but they answer a different question — "can the model recognise a new recording from a familiar voice?" — and should not be read as evidence of generalization to new speakers.

### Small Actor Pool

Only 24 actors are available, so actor-grouped six-fold CV holds out roughly four actors per fold and the single held-out-actor test set uses 6 actors (190 recordings). Fold accuracies range from 41.41% to 66.41% (±7.91%), and the held-out figure is one draw with no confidence interval. The grouped results are therefore directional evidence of an evaluation gap, not a precise estimate of speaker-independent accuracy and not a significance test.

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
* Train and tune under a speaker-disjoint protocol by default, rather than treating it as a secondary check.
* Report repeated grouped cross-validation with shuffled actor assignment, so the speaker effect is separated from fold composition.
* Use additional speech-emotion datasets to assess generalisation across corpora and recording conditions.
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
73.96% held-out test accuracy          (speakers overlap train/test)
80.14% mean 5-fold CV accuracy          (speakers overlap across folds)
82.89% mean 6-fold CV accuracy          (random stratified, speakers overlap)
54.58% mean 6-fold CV accuracy          (actor-grouped, speakers disjoint)
```

The central finding is the difference between the last two lines. Under the same features and the same MLP, changing only the splitting rule so that no speaker appears in both training and evaluation data drops mean accuracy from **82.89% to 54.58%** — a 28.31-point gap. The model is still well above the 25% random-guess floor, so the acoustic features carry real emotion signal. What the gap establishes is that the speaker-overlapping figure is **optimistic — it cannot be read as performance on an unseen voice**. It does not establish how much of the decline is due to speaker identity specifically (the grouping also changes fold composition and recording conditions), and no significance test is reported.

A direct comparison also showed that the scaled-feature pipeline substantially outperformed the same MLP using unscaled features on the recorded test split. The `happy` → `fearful` confusion is the one error pattern that survives the stricter evaluation, appearing under both splitting rules.

These results should be interpreted specifically for the **four-class RAVDESS subset and the documented experimental setup**, rather than as a general benchmark for speech emotion recognition.

---

## Author

**Kehinde Azeez Ajani**

Data Science | Data Analytics | Machine Learning | AI

GitHub: [Ajanikehinde](https://github.com/Ajanikehinde)
