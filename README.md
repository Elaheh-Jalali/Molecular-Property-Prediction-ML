# Molecular Property Prediction

This project demonstrates a simple machine learning workflow to predict the solubility (logS) of small molecules based on their molecular structure. It uses **RDKit** for feature extraction and **Scikit-learn** for model training. The project is intended as a starting point for machine learning applications in computational chemistry and drug discovery.


# -------------- Dataset --------------

- **File:** `esol.csv`
- **Source:** [Kaggle - Delaney ESOL dataset](https://www.kaggle.com/datasets/debajyotipodder/delaney-esol-dataset)
- **Description:** Contains SMILES strings and measured solubility values (logS) for small molecules.
- **Columns:**
  - `smiles` → Molecule in SMILES format
  - `measured log solubility in mols per litre` → Experimental solubility (logS)


# -------------- Files in this repository --------------
- molecular_property_prediction.py  # main script
- esol.csv                          # full dataset from Kaggle
- README.md                         # this file


# -------------- Feature Extraction --------------

# Using RDKit, the following molecular descriptors are extracted:

- Molecular Weight
- LogP (octanol-water partition coefficient)
- Number of H-bond Acceptors
- Number of H-bond Donors
- Topological Polar Surface Area (TPSA)
- Number of Rotatable Bonds

# -------------- Model --------------

- **Algorithm:** Random Forest Regressor
- **Metrics:** Mean Absolute Error (MAE), R² Score
- **Goal:** Predict solubility values from molecular descriptors

# -------------- Usage --------------

1. Install dependencies:

`pip install rdkit pandas scikit-learn matplotlib`

2. Run the script:

`python molecular_property_prediction.py`



# The script will:

- load the dataset

- calculate molecular features using RDKit

- train a Random Forest model

- make predictions on test data

- print error metrics (MAE, R²)
                       
- show a plot of predicted vs actual solubility


# -------------- Results --------------
- Scatter plot visualizing measured vs predicted solubility

- Quantitative performance metrics (MAE, R²)




