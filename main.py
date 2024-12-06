from utils.preprocessor import Preprocessor
from utils.visualizer import DataVisualizer

if __name__ == "__main__":
    # Instantiate the Preprocessor with your CSV file path
    preprocessor = Preprocessor("Datasets\data.csv")
    
    # Get preprocessed features and labels
    X_preprocessed, y = preprocessor.preprocess()

    # Get numerical and categorical features
    numerical_features = preprocessor.numerical_features
    categorical_features = preprocessor.categorical_features

    # Instantiate the DataVisualizer
    visualizer = DataVisualizer(preprocessor.data, numerical_features, categorical_features, X_preprocessed)

    # Plot different visualizations
    visualizer.plot_distributions('Age')
    visualizer.plot_boxplots()
    visualizer.plot_correlation_heatmap()
    visualizer.plot_pca()
    visualizer.plot_count_plots()
    visualizer.analyze_variance()
