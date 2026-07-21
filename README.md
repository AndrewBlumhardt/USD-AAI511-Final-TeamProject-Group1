# Final Team Project: Music Genre and Composer Classification Using Deep Learning

**Course:** AAI-511: Neural Networks and Deep Learning  
**University:** University of San Diego  
**Assignment Completion Date:** August 10, 2026

## Team Members

- Andrew Blumhardt
- Brian Covington
- Erika Gallegos

## Project Overview

This repository contains the final team project for AAI-511: Neural Networks and Deep Learning.

The assignment required the team to design, build, train, and evaluate deep learning models capable of classifying classical music compositions from MIDI files. Although the assignment title includes both music genre and composer classification, the implemented project focuses on **composer classification** using compositions from four required composers:

- Johann Sebastian Bach
- Ludwig van Beethoven
- Frédéric Chopin
- Wolfgang Amadeus Mozart

Two deep learning architectures were developed and compared:

- A **Long Short-Term Memory (LSTM)** network using ordered note sequences
- A **Convolutional Neural Network (CNN)** using piano-roll image representations

The final notebook demonstrates the complete deep learning workflow, including data acquisition, preprocessing, exploratory analysis, augmentation, model development, training, evaluation, and comparison.

## Assignment Objectives

The project required the team to:

- Obtain and prepare a real-world music dataset
- Extract meaningful features from MIDI files
- Build and train multiple deep learning models
- Optimize model training and reduce overfitting
- Evaluate each model with appropriate classification metrics
- Compare the strengths and limitations of different architectures
- Document the project, results, and lessons learned

The assignment emphasized understanding the complete machine learning lifecycle rather than focusing only on the highest possible accuracy.

## Dataset

The project uses the public **MIDI Classic Music** dataset hosted on Kaggle:

`blanderbuss/midi-classic-music`

The dataset is downloaded directly from Kaggle by the notebook using `kagglehub`. The dataset is not stored in this repository, which keeps the repository smaller and makes the data-acquisition process reproducible.

Only MIDI files associated with Bach, Beethoven, Chopin, and Mozart are retained for the classification task.

The source dataset is imbalanced. Before preprocessing, the distribution used in the final notebook was:

| Composer | MIDI Files |
|---|---:|
| Bach | 1,036 |
| Mozart | 257 |
| Beethoven | 213 |
| Chopin | 136 |

The project addresses this imbalance through stratified training, validation, and test splits; balanced class weights during training; and macro-averaged precision, recall, and F1 metrics.

## Project Workflow

The final notebook performs the following major steps:

1. Configures and validates the Python, TensorFlow, and GPU environment.
2. Downloads the MIDI dataset directly from Kaggle.
3. Discovers, labels, and filters MIDI files for the four required composers.
4. Parses MIDI files and extracts musical features.
5. Creates ordered note sequences for the LSTM.
6. Creates fixed-size piano-roll representations for the CNN.
7. Examines the class distribution and descriptive musical characteristics.
8. Creates stratified training, validation, and test sets.
9. Applies pitch-transposition augmentation only to the training data.
10. Trains the LSTM and CNN models.
11. Uses early stopping and learning-rate reduction to control training.
12. Evaluates both models using accuracy, macro precision, macro recall, macro F1, classification reports, and confusion matrices.
13. Compares the final models and summarizes their strengths and limitations.

## Model Architectures

### LSTM

The LSTM model analyzes ordered sequences of MIDI note pitches. Its main components include:

- An embedding layer for MIDI pitch values
- An LSTM recurrent layer
- Dense classification layers
- Dropout for regularization
- A four-class softmax output layer

This approach is intended to learn sequential relationships among notes.

### CNN

The CNN analyzes piano-roll matrices representing pitch across time. Its main components include:

- Three convolutional blocks
- Batch normalization
- Max pooling
- Dropout
- Global average pooling
- Dense classification layers
- A four-class softmax output layer

This approach treats each composition as a two-dimensional musical representation and learns local pitch-and-time patterns.

## Training Approach

Both models were configured for a maximum of 30 epochs. Early stopping monitored validation loss and ended training when additional epochs no longer produced meaningful improvement. The best validation weights were restored automatically.

The models were also trained with:

- Balanced class weights
- Training-only pitch-transposition augmentation
- `ReduceLROnPlateau`
- Separate validation data
- Reproducible random seeds

The LSTM and CNN completed different numbers of epochs because they use different representations, model structures, learning rates, and optimization paths.

GPU acceleration is strongly recommended when running the notebook in Google Colab. The CNN can take several minutes per epoch when trained on a CPU.

## Results Summary

The final run showed that the CNN performed better overall than the LSTM.

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| CNN | 0.7683 | 0.6364 |
| LSTM | 0.6494 | 0.5564 |

The CNN performed especially well on Bach, which was also the largest class in the dataset. However, the CNN also produced the higher macro F1 score, indicating that its overall advantage was not explained only by performance on the largest class.

Performance varied considerably among composers. This reinforced the importance of reviewing per-class precision, recall, F1 scores, and confusion matrices rather than relying only on overall accuracy.

The final training run completed:

- 15 LSTM epochs, with the best weights restored from epoch 10
- 25 CNN epochs

Stopping before the 30-epoch maximum was expected because early stopping detected that validation performance had stopped improving.

## Lessons Learned

Several important lessons emerged during the project:

### Data preparation required substantial effort

MIDI files vary in structure, instrumentation, length, encoding, and quality. Reliable preprocessing, filtering, error handling, and validation were necessary before model training could begin.

### Representation affects what a model can learn

The LSTM and CNN received different representations of the same compositions. The LSTM learned from note order, while the CNN learned from visual pitch-and-time patterns. The CNN representation produced stronger results in the final run.

### More epochs do not always produce a better model

An epoch is one complete pass through the training data. Early epochs often produce the largest improvements, while later epochs produce smaller returns and may lead to overfitting. Early stopping prevented unnecessary training and restored the strongest validation weights.

### Different models converge at different rates

The CNN required more epochs than the LSTM in the final run. This was not an error. Different architectures, learning rates, parameter counts, and feature representations can require different amounts of training.

### Class imbalance must be considered throughout the project

Bach had substantially more source files than the other composers. Stratification, balanced class weights, macro metrics, classification reports, and confusion matrices helped provide a more complete view of performance.

### Accuracy alone is not sufficient

A model can achieve strong overall accuracy by performing well on the largest class. Macro F1 and per-class results were necessary to understand whether the models generalized across all four composers.

### Hardware affects practicality

The CNN was computationally expensive when trained without a GPU. Selecting a GPU in Google Colab greatly improves the practical execution time of the notebook.

### Reproducibility improves the value of the project

Downloading the dataset directly from Kaggle, validating each substantial processing step, fixing random seeds, and documenting the runtime environment made the final notebook easier to review and reproduce.

## Repository Contents

The repository includes:

- The final executed Jupyter notebook
- This README file
- Supporting project files, if applicable

The primary notebook is:

`USD_AAI511_Final_TeamProject_Group1.ipynb`

## Running the Notebook

The notebook was developed for Google Colab.

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type**.
3. Select an available **GPU** hardware accelerator.
4. Save the runtime setting.
5. Run the environment-validation cell and confirm that TensorFlow detects a GPU.
6. Run the notebook from beginning to end.

The notebook installs required packages and downloads the dataset automatically.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- PrettyMIDI
- Matplotlib
- KaggleHub
- Google Colab

## Contributors

This project was completed collaboratively by:

- Andrew Blumhardt
- Brian Covington
- Erika Gallegos

The team contributed to project planning, implementation, testing, review, analysis, documentation, and refinement of the final submission.

## Attribution

This project was completed for:

**University of San Diego**  
**AAI-511: Neural Networks and Deep Learning**

The MIDI data used by the project comes from the public **MIDI Classic Music** dataset hosted on Kaggle by the dataset publisher.

The project also relies on open-source Python libraries, including TensorFlow, Keras, Scikit-learn, PrettyMIDI, NumPy, Pandas, Matplotlib, and KaggleHub.

Generative AI tools were used to assist with code refinement, debugging, documentation, explanation, and review. All generated content was reviewed, modified where appropriate, tested, and validated by the project team before submission.

## Limitations

The project has several limitations:

- MIDI files may reflect transcription and arrangement choices in addition to composer style.
- Fixed sequence lengths truncate some longer compositions.
- Piano-roll resizing compresses timing information.
- Pitch augmentation creates variations of existing works rather than entirely new compositions.
- The class distribution is uneven.
- The results apply to this dataset and test split.
- The notebook implements composer classification, not a separate music-genre classifier.

The results should therefore be interpreted as a comparison of two deep learning approaches within the scope of this assignment, rather than as a universal composer-identification system.
