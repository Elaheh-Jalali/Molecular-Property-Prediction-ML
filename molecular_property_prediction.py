import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------- Step 1: Load dataset --------------
data = pd.read_csv("esol.csv")
print("Dataset loaded successfully!")
print(data.head())

# -------------- Step 2: Feature extraction with RDKit --------------
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
    ]

X = data["smiles"].apply(featurize).tolist()
y = data["measured log solubility in mols per litre"]

# -------------- Step 3: Train-test split --------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# --- Step 4: Train Random Forest ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------- Step 5: Evaluate --------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R² Score: {r2:.3f}")

# -------------- Step 6: Visualization --------------
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Measured Solubility")
plt.ylabel("Predicted Solubility")
plt.title("Measured vs Predicted Solubility")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
plt.show()

