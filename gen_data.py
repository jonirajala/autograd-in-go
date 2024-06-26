from sklearn.datasets import fetch_california_housing
import pandas as pd 

housing = fetch_california_housing()
features, targets = pd.DataFrame(housing.data), pd.DataFrame(housing.target)

features.to_csv("features.csv", index=False, header=False)
targets.to_csv("targets.csv", index=False, header=False)
