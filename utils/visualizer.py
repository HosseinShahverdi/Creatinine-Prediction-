import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

class DataVisualizer:
    def __init__(self, data, numerical_features, categorical_features, X_preprocessed=None):
        self.data = data
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.X_preprocessed = X_preprocessed

    def plot_distributions(self, feature):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(self.data[feature], ax=ax[0], kde=True)
        ax[0].set_title(f'Original {feature} Distribution')

        if self.X_preprocessed is not None:
            sns.histplot(self.X_preprocessed[:, self.numerical_features.index(feature)], ax=ax[1], kde=True)
            ax[1].set_title(f'Scaled {feature} Distribution')

        plt.show()

    def plot_boxplots(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data[self.numerical_features])
        plt.title('Boxplot for Numerical Features Before Preprocessing')
        plt.xticks(rotation=45)
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data[self.numerical_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap Before Preprocessing')
        plt.show()

    def plot_pca(self):
        if self.X_preprocessed is not None:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X_preprocessed)

            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            plt.title('PCA Visualization After Preprocessing')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()

    def plot_count_plots(self):
        for cat in self.categorical_features:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=self.data[cat])
            plt.title(f'Count Plot for {cat} Before Preprocessing')
            plt.xticks(rotation=45)
            plt.show()

    def analyze_variance(self):
        original_variance = self.data[self.numerical_features].var()
        print("Original Variance:\n", original_variance)

        if self.X_preprocessed is not None:
            scaled_variance = np.var(self.X_preprocessed[:, :len(self.numerical_features)], axis=0)
            print("Variance After Scaling:\n", scaled_variance)
