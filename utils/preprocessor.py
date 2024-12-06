import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, csvFile, correlation_threshold=0.9):
        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(csvFile)
        self.correlation_threshold = correlation_threshold

    def clean_data(self):
        # Remove duplicate rows from the dataset
        self.data.drop_duplicates(inplace=True)

        # Drop rows that have more than two missing values
        self.data.dropna(thresh=len(self.data.columns) - 2, inplace=True)

    def split_features_target(self, target='Creatinine'):
        # Automatically detect numerical and categorical features
        self.features = self.data.columns.difference([target])
        self.numerical_features = self.data[self.features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.data[self.features].select_dtypes(include=['object', 'category']).columns.tolist()

        # Split dataset into features (X) and target (y)
        X = self.data[self.features]
        y = self.data[target]

        return X, y

    def preprocess(self, target='Creatinine'):
        # Clean the dataset
        self.clean_data()

        # Split features and target
        X, y = self.split_features_target(target)

        # Define the preprocessing steps for numerical features
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numerical values with the mean
            ('scaler', StandardScaler())  # Standardize numerical features to have mean=0 and std=1
        ])

        # Define the preprocessing steps for categorical features
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with the most frequent value
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
        ])

        # Combine preprocessing for numerical and categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numerical_features),  # Apply numerical transformer to numerical features
                ('cat', cat_transformer, self.categorical_features)  # Apply categorical transformer to categorical features
            ],
            remainder='passthrough'  # Keep other columns unchanged
        )

        # Fit the preprocessing pipeline to the features and transform them
        X_preprocessed = preprocessor.fit_transform(X)

        return X_preprocessed, y

    def get_train_test_data(self, test_size=0.2, random_state=42, target='Creatinine'):
        # Preprocess the data to get preprocessed features and target
        X, y = self.preprocess(target=target)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
