import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self, csvFile):
        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(csvFile)

    def preprocess(self):
        # Define features and target
        features = ['Age', 'FBS', 'Cholesterol', 'Triglycerides', 'HDL Cholesterol', 'HGB', 'HCT']
        X = self.data[features]
        y = self.data['Creatinine']  # Assuming 'Creatinine' is the column name for the target variable

        # Define the preprocessing steps for numerical features
        num_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
                ('scaler', StandardScaler()),  # Standard scaling
                ('minmax', MinMaxScaler())  # Min-max scaling (note: using both StandardScaler and MinMaxScaler is redundant)
            ]
        )

        # Combine the transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, features)
            ]
        )

        # Fit and transform the data
        X_preprocessed = preprocessor.fit_transform(X)

        return X_preprocessed, y
