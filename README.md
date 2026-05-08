# Privacy and Ethical Frameworks for AI Use in Public Health Surveillance

This project investigates privacy-preserving machine learning techniques for public health surveillance using the NHANES and COVID-19 datasets.

The project evaluates the trade-off between privacy protection and predictive utility across multiple machine learning models and privacy-preserving approaches.

## Techniques

Current implementation:
- Differential Privacy (DP)

Planned/extended techniques:
- Federated Learning (FL)
- k-Anonymity
- Homomorphic Encryption (HE)

## Models

- Logistic Regression
- Random Forest

## Datasets

### NHANES
National Health and Nutrition Examination Survey (NHANES) dataset used for diabetes classification.

### COVID-19 Case Surveillance
COVID-19 Case Surveillance Public Use dataset used for mortality prediction.

## Evaluation Metrics

The experiments evaluate:
- Accuracy
- F1 Score
- Precision
- Recall
- Training Runtime
- Confusion Matrices

Experiments are repeated across 5 random seeds:
- 1
- 21
- 42
- 100
- 123

Differential Privacy experiments evaluate multiple epsilon values:
- 10.0
- 5.0
- 1.0
- 0.5
- 0.1

## Project Structure

```text
privacy-ai-public-health/
├── datasets/
├── graphs/
├── results/
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

## Python Version

This project was tested using:

- Python 3.11

Python 3.14 caused compatibility issues between:
- diffprivlib
- scikit-learn

Specifically, diffprivlib's Logistic Regression implementation is currently incompatible with newer scikit-learn versions bundled with Python 3.14 environments.

## Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv privacy_env
source privacy_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the NHANES preprocessing script:

```bash
python src/merge_nhanes.py
```

Run NHANES Differential Privacy experiments:

```bash
python src/nhanes_dp_experiment.py
```

Run COVID-19 Differential Privacy experiments:

```bash
python src/covid_dp_experiment.py
```

Generate visualisations:

```bash
python src/visualise_results.py
```

## Outputs

### Results CSVs
Saved in:

```text
results/
```

### Visualisations
Saved in:

```text
graphs/
```

## Notes

- Differential Privacy models use `diffprivlib`.
- Baseline models use standard scikit-learn implementations.
- Differential Privacy experiments compare privacy-performance trade-offs across multiple epsilon values.
- Lower epsilon values provide stronger privacy but may reduce predictive performance.

## Authors

- Ansh Humagain
- Akshay Singh
- Luca Martins
- Sanskar Fadatare
- Dylan Chum