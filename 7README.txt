# Supplementary Code for "A Parallel Iterative Proportional Scaling Algorithm for Gaussian Graphical Models Based on Graph Coloring"
This code is supplementary material for the paper "[A Parallel Iterative Proportional Scaling Algorithm for Gaussian Graphical Models Based on Graph Coloring]" and is intended solely for review and verification purposes. 
It is not intended for general redistribution or production use.
This repository contains the code to reproduce all numerical experiments in the paper.

## Structure

- `common.py`: Shared utilities and the GGM_Fitter class (serial and parallel CovIPS).
- `experiment_accuracy.py`: Accuracy evaluation using true graphical models.
- `experiment_efficiency_grid.py`: Efficiency test on rectangular grid graphs.
- `experiment_efficiency_tree.py`: Efficiency test on random trees with extra edges.
- `experiment_real.py`: Real data experiment on prostate cancer gene expression data.

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt


Data for the Real Experiment
The real data used in this paper is from the TCGA breast cancer dataset.
Users can obtain the data from the TCGA portal and adjust the file path accordingly.



Running the Experiments
Each experiment script can be run independently, e.g.:

bash
python experiment_accuracy.py
python experiment_efficiency_grid.py
python experiment_efficiency_tree.py
python experiment_real.py
Output
Each script saves a CSV file containing the results (timing, iterations, accuracy measures, etc.) with a timestamp.

Note on Data Generation
Accuracy experiment (experiment_accuracy.py): Uses generate_true_model to create a true precision matrix K0 and covariance Σ0, then samples from N(0, Σ0). This allows computing estimation errors.

Efficiency experiments (experiment_efficiency_grid.py, experiment_efficiency_tree.py): Use independent standard normal data (covariance = identity). Only timing and convergence are measured.

Real experiment (experiment_real.py): Uses the TCGA breast cancer dataset.(https://www.cancer.gov/tcga).

License
This code is provided for review purposes only. All rights reserved.
For any other use, please contact the corresponding author.