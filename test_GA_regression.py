from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from genetic_selection import GeneticSelectionCV
import pandas as pd
import numpy as np

def main():
    # data = datasets.load_breast_cancer()
    # df = pd.DataFrame(data.data, columns=data.feature_names)
    # df['target'] = data.target
    # X = df.drop(['target'], axis=1)
    # y = df['target'].astype(float)
    
    X, y = datasets.load_diabetes(return_X_y=True)
    data = datasets.load_diabetes()
    diabetes_columns=data.feature_names

    # estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    # estimator = DecisionTreeClassifier()
    estimator = DecisionTreeRegressor()

    selector = GeneticSelectionCV(
                            estimator, cv=10, verbose=1,
                            scoring="neg_mean_squared_error", max_features=10,
                            n_population=100, crossover_proba=0.5,
                            mutation_proba=0.2, n_generations=50,
                            crossover_independent_proba=0.5,
                            mutation_independent_proba=0.04,
                            tournament_size=3, n_gen_no_change=10,
                            caching=True, n_jobs=-1)
    selector = selector.fit(X, y)
    # MSE -5363.715484800969 RMSE 73
    print('Features:', np.array(diabetes_columns)[selector.support_])
    # ['bp']
    
if __name__ == "__main__":
    main()
