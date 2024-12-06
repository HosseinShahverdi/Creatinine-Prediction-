from utils.preprocessor import Preprocessor
from utils.visualizer import DataVisualizer
from models.randomForest import RandomForestModel
from models.logisticRegressor import LogisticRegressionModel
from models.SGDclassifier import SGDClassifierModel
from sklearn.model_selection import train_test_split

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
    
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    
    # Train RandomForest with Grid Search
    rf_model = RandomForestModel()
    best_model_rf, best_params_rf = rf_model.train_grid_search(X_train, y_train)
    print(f"Best Parameters for RandomForest: {best_params_rf}")
    print(f"RandomForest Model Accuracy: {best_model_rf.score(X_test, y_test)}")

    # Train Logistic Regression with Grid Search
    lr_model = LogisticRegressionModel()
    best_model_lr, best_params_lr = lr_model.train_grid_search(X_train, y_train)
    print(f"Best Parameters for Logistic Regression: {best_params_lr}")
    print(f"Logistic Regression Model Accuracy: {best_model_lr.score(X_test, y_test)}")
    
    # SGD classifier
    sgd_model = SGDClassifierModel()
    best_model_sgd, best_params_sgd = sgd_model.train_grid_search(X_train, y_train)
    print(f"Best Parameters for SGD Classifier: {best_params_sgd}")
    print(f"SGD Classifier Model Accuracy: {best_model_sgd.score(X_test, y_test)}")
    

