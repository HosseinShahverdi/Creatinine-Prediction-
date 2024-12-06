from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


class SGDClassifierModel:
    def __init__(self):
        self.model = SGDClassifier()
        
    def train_grid_search(self, X_train, y_train):
        param_grid = {
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'n_jobs': [-1],
            'max_iter': [1000, 2000, 3000, 4000],
            'epsilon': [0.1, 0.01, 0.001, 0.0001],
            'class_weight': ['balanced', None],
        }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_