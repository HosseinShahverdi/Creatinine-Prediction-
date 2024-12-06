from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class RandomForestModel:
    def __init__(self):
        self.model= RandomForestClassifier()
        
    def train_grid_search(self, X_train, y_train):
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_