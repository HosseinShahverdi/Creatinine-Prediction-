from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()
        
    def train_grid_search(self, X_train, y_train):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['liblinear', 'saga', 'newton-cholesky', 'sag', 'lbfgs'],
            'max_iter': [100, 200, 300, 400, 500]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

