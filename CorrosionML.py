import os

print(os.listdir())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("EIS_MgCa-CG.csv")

# Show first rows
print(df.head())

# Dataset shape
print("Shape:", df.shape)

# Column names
print(df.columns)

print(df.info())

print(df.describe())

print(df.isnull().sum())

df.columns = ['f1','Re1','Im1','Z1','phi1',
              'f2','Re2','Im2','Z2','phi2']

print(df.head())

import pandas as pd

df = pd.read_csv("EIS_MgCa-CG.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
df.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("EIS_MgCa-CG.csv")

# Remove useless column
if "Fit" in df.columns:
    df = df.drop(columns=["Fit"])

print("Shape:", df.shape)
print(df.head())

# =========================
# SPLIT DATASETS
# =========================
f1, Re1, Im1, Z1, phi1 = df["f"], df["Re"], df["Im"], df["Z"], df["phi"]

f2, Re2, Im2, Z2, phi2 = df["f.1"], df["Re.1"], df["Im.1"], df["Z.1"], df["phi.1"]

# =========================
# NYQUIST PLOT
# =========================
plt.figure(figsize=(6,6))
plt.plot(Re1, -Im1, 'o-', label="Sample 1")
plt.plot(Re2, -Im2, 's-', label="Sample 2")
plt.xlabel("Z' (Ω)")
plt.ylabel("-Z'' (Ω)")
plt.title("Nyquist Plot")
plt.legend()
plt.grid()
plt.show()

# =========================
# BODE MAGNITUDE
# =========================
plt.figure(figsize=(7,5))
plt.semilogx(f1, Z1, label="Sample 1")
plt.semilogx(f2, Z2, label="Sample 2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|Z|")
plt.title("Bode Magnitude")
plt.legend()
plt.grid()
plt.show()

# =========================
# BODE PHASE
# =========================
plt.figure(figsize=(7,5))
plt.semilogx(f1, phi1, label="Sample 1")
plt.semilogx(f2, phi2, label="Sample 2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (°)")
plt.title("Bode Phase")
plt.legend()
plt.grid()
plt.show()

# =========================
# SIMPLE FEATURE EXTRACTION
# (for future ML dataset building)
# =========================
features = pd.DataFrame({
    "Rct_proxy_1": [Re1.max()],
    "Rct_proxy_2": [Re2.max()],
    "Im_min_1": [Im1.min()],
    "Im_min_2": [Im2.min()],
    "Z_max_1": [Z1.max()],
    "Z_max_2": [Z2.max()],
    "phi_min_1": [phi1.min()],
    "phi_min_2": [phi2.min()],
})

print("\nExtracted Features:")
print(features)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

import shap

df = pd.read_csv("EIS_MgCa-CG.csv")

if "Fit" in df.columns:
    df = df.drop(columns=["Fit"])

print(df.head())
print(df.shape)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

print("Imports successful")

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("RF R2:", r2_score(y_test, pred))

import numpy as np
import pandas as pd

np.random.seed(42)

n = 300

data = pd.DataFrame({
    "TA": np.random.uniform(0, 5, n),
    "Ce": np.random.uniform(0, 3, n),
    "GO": np.random.uniform(0, 2, n),
    "Thickness": np.random.uniform(5, 50, n),
    "ContactAngle": np.random.uniform(60, 120, n),
    "Roughness": np.random.uniform(0.1, 2.0, n),
    "pH": np.random.uniform(6, 9, n),
    "ExposureTime": np.random.uniform(1, 100, n)
})

data["Rct"] = (
    5000 + 800*data["GO"] + 600*data["Ce"] +
    400*data["TA"] - 50*data["Roughness"]
)

X = data.drop(columns=["Rct"])
y = data["Rct"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train-test split done")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "GPR": GaussianProcessRegressor()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results[name] = {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred))
    }

results

import pandas as pd

results_df = pd.DataFrame(results).T
print(results_df)

from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=300, random_state=42)

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("CV R2 scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())

rf.fit(X_train, y_train)

import matplotlib.pyplot as plt

importance = rf.feature_importances_

plt.figure(figsize=(8,5))
plt.barh(X.columns, importance)
plt.xlabel("Importance")
plt.title("Feature Importance (Corrosion Resistance Drivers)")
plt.show()

pred = rf.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, pred)

plt.xlabel("Actual Rct")
plt.ylabel("Predicted Rct")
plt.title("Parity Plot (RF Model)")
plt.grid()
plt.show()

import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor

models = {
    "RF": RandomForestRegressor(n_estimators=300, random_state=42),
    "GPR": GaussianProcessRegressor(),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5
    )
}

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results[name] = {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred))
    }

results

import pandas as pd
import matplotlib.pyplot as plt

df_results = pd.DataFrame(results).T

df_results["R2"].plot(kind="bar", color="steelblue")
plt.title("Model Comparison (R² Score)")
plt.ylabel("R²")
plt.ylim(0,1)
plt.grid()
plt.show()

from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=300)

cv_scores = cross_val_score(
    rf,
    X_scaled,
    y,
    cv=5,
    scoring="r2"
)

print("CV R2 scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())

best_model = RandomForestRegressor(n_estimators=300, random_state=42)
best_model.fit(X_train, y_train)

pred = best_model.predict(X_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, pred, alpha=0.7)

plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)],
         'r--')

plt.xlabel("Actual Rct")
plt.ylabel("Predicted Rct")
plt.title("Parity Plot (ML Model)")
plt.grid()
plt.show()

best_model.fit(X_train, y_train)

import matplotlib.pyplot as plt

importance = best_model.feature_importances_

plt.figure(figsize=(8,5))
plt.barh(data.drop(columns=["Rct"]).columns, importance)
plt.title("Corrosion Feature Importance")
plt.xlabel("Importance")
plt.show()

import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

import matplotlib.pyplot as plt
import numpy as np

best_model.fit(X_train, y_train)

importance = best_model.feature_importances_
features = data.drop(columns=["Rct"]).columns

plt.figure(figsize=(8,5))
plt.barh(features, importance, color="teal")
plt.xlabel("Feature Importance")
plt.title("Corrosion Feature Influence (ML Interpretation)")
plt.grid()
plt.show()

from sklearn.inspection import permutation_importance

result = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

importance = result.importances_mean
features = data.drop(columns=["Rct"]).columns

plt.figure(figsize=(8,5))
plt.barh(features, importance, color="orange")
plt.xlabel("Permutation Importance")
plt.title("Feature Importance (Robust Method)")
plt.grid()
plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

search = RandomizedSearchCV(
    rf,
    param_grid,
    n_iter=20,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("Best Parameters:", search.best_params_)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

pred = best_model.predict(X_test)

print("R2:", r2_score(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("MAE:", mean_absolute_error(y_test, pred))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    best_model,
    X,
    y,
    cv=10,
    scoring="r2"
)

print("CV R2 mean:", scores.mean())
print("CV R2 std:", scores.std())

from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X,
    y,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Train Score")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation Score")

plt.xlabel("Training Size")
plt.ylabel("R² Score")
plt.title("Learning Curve")
plt.legend()
plt.grid()
plt.show()

residuals = y_test - pred

plt.scatter(pred, residuals)
plt.axhline(0, color="red")

plt.xlabel("Predicted Rct")
plt.ylabel("Residual Error")
plt.title("Residual Plot")
plt.grid()
plt.show()

plt.figure(figsize=(6,6))

plt.scatter(y_test, pred, alpha=0.7)

plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)],
         'r--')

plt.xlabel("Actual Rct")
plt.ylabel("Predicted Rct")
plt.title("Parity Plot (Optimized Model)")
plt.grid()
plt.show()



