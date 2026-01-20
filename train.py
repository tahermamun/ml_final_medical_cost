# imports
import pandas as pd
import numpy as np
import pickle

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor


# load data
df = pd.read_csv("./data/insurance_data.csv")

# remove duplicates
df = df.drop_duplicates()
df.duplicated().sum()

# Separate target
X = df.drop("charges", axis=1)
y = df["charges"]

# identify numarical and categorical feature
num_cols = X.select_dtypes(include=np.number).columns
cat_cols =  X.select_dtypes(exclude=np.number).columns

# numerical pipeline
num_pipeline = Pipeline(
    steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # polynomial features
    ("scaler", StandardScaler())
])

# categorical pipeline
cat_pipeline = Pipeline(
    steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# combine numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])



# split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# model train and tuning
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator= pipe,
    param_grid = param_grid,
    cv= 5,
    scoring="r2",
    n_jobs=-1,
    verbose = 1
)

grid_search.fit(X,y)


best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)


# Save model (IMPORTANT)
with open("medical_insurance_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model File Ready.pkl")