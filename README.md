# Privacy-Preserving AI for Public Health Surveillance

This project investigates privacy-preserving machine learning techniques for public health surveillance systems. The goal is to evaluate how different privacy methods affect model performance, privacy protection, and ethical considerations when using sensitive healthcare data.

The project compares multiple approaches, which includes:
- Differential Privacy
- Federated Learning (FL)
- k-Anonymity
- Homomorphic Encryption

Experiments are conducted using public health datasets including the COVID-19 Case Surveillance dataset and NHANES 2013 - 2014.

# Project Structure

```text
privacy-ai-public-health/
│
├── datasets/        # Input datasets
├── graphs/          # Generated visualisations
├── results/         # CSV experiment outputs
├── src/             # Python experiment scripts
├── README.md
├── requirements.txt
└── .gitignore
```
# Requirements

This project was tested using the following:
- Python 3.11
- scikit-learn 1.6.1
- diffprivlib 0.66
- TenSEAL 0.3.16
- Micorosft SEAL backend (installed automatically through TenSEAL)
Python 3.14 may cause compatibility issues with diffprivlib and newer scikit-learn versions.

# Differential Privacy Setup

Create and activate a virtual enviornment:

python3.11 -m venv privacy_env
source privacy_env/bin/activate

Install dependencies:

pip install -r requirements.txt

# Homomorphic Encryption Setup

The Homomorphic Encryption experiments use TenSEAL, which wraps the Microsoft SEAL encrpytion library. 

Install additional dependencies if required:

pip install tenseal
pip install cmake

If installation fails on macOS:

brew install cmake
brew install protobuf

Then retry: 

pip install tenseal

# Federated Learning Setup

The Federated Learning experiments use a simulated cross-silo setup where the training data is split into five local clients, with each client training a local model. The model updates are combined using federated averaging, and raw training data is not transferred between clients in the simulation.

This is a local simulation rather than a real distributed deployment, and thus communication security, secure aggregation and network latency are outside the scope of this implementation.

# COVID-19 Dataset

The COVID-19 Case Surveillance dataset is too large to store directly in the Github repo. 

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/arashnic/covid19-case-surveillance-public-use-dataset

After downloading, place place the CSV file inside the datasets/ folder and rename it to:

covid.csv

Expected location:

datasets/covid.csv

# Running the Experiments

## Merge NHANES files

python src/merge_nhanes.py

This creates:

datasets/nhanes_merged.csv

## Run NHANES Differential Privacy Experiment

python src/nhanes_dp_experiment.py

## Run COVID-19 Differential Privacy Experiment

python src/covid_dp_experiment.py

## Run NHANES Homorphic Encryption Experiment

python src/he_nhanes_experiment.py

## Run NHANES Federated Learning Experiment

python src/nhanes_fl_experiment.py

## Run COVID-19 Federated Learning Experiment

python src/covid_fl_experiment.py

## Generate Differential Privacy Visualisations

python src/visualise_dp_results.py

Generated graphs will be saved inside:

graphs/

## Generate Homomorphic Encryption Visualisations

python src/visualise_he_results.py

## Generate Federated Learning Visualisations

python src/visualise_fl_results.py

# Outputs

Experiment results are automatically saved inside:

results/

Including:
- Raw experiment outputs
- Summary statistics
- Averaged metrics across random seeds

Metrics include:
- Accuracy
- F1 Score
- Precision
- Recall
- Training runtime