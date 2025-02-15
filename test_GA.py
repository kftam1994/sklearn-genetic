from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from genetic_selection import GeneticSelectionCV
import pandas as pd

def main():
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    X = df.drop(['target'], axis=1)
    y = df['target'].astype(float)

    # estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    estimator = DecisionTreeClassifier()

    selector = GeneticSelectionCV(
                            estimator, cv=5, verbose=0,
                            scoring="accuracy", max_features=5,
                            n_population=100, crossover_proba=0.5,
                            mutation_proba=0.2, n_generations=50,
                            crossover_independent_proba=0.5,
                            mutation_independent_proba=0.04,
                            tournament_size=3, n_gen_no_change=10,
                            caching=True, n_jobs=-1)
    selector = selector.fit(X, y)

    print('Features:', X.columns[selector.support_])
    # ['mean area', 'compactness error', 'worst texture', 'worst smoothness','worst concavity']

if __name__ == "__main__":
    main()