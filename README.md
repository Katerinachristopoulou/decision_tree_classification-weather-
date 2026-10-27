# decision_tree_classification-weather-

Weather type prediction with decision trees (scikit-learn) | Data mining exercise

## What it does

`main.py` downloads the [Weather History dataset](https://www.kaggle.com/datasets/muthuj7/weather-dataset)
from Kaggle and predicts the hourly weather summary, grouped into 5 classes
(Clear, Cloudy, Foggy, Rainy, Windy), from temperature, humidity, wind, visibility,
pressure, precipitation type, hour and month.

Steps:

1. Clean the data (drop leakage-prone and empty columns, treat pressure 0 as missing, add hour/month features).
2. Pick the tree depth with 5-fold cross-validation (scored by macro F1, with balanced class weights).
3. Grid search over `max_depth`, `min_samples_leaf` and `ccp_alpha`.
4. Compare two parameter sets (Gini/no pruning vs Entropy/pruned) using 5- and 10-fold CV
   and repeated 70/30 and 80/20 train/test splits.

## Outputs

- `DecisionTree_Comparison.xlsx`: accuracy and macro F1 for each parameter set and validation method
- `cv_accuracy.png`: CV score vs tree depth
- `decision_tree.png`: top levels of the depth-selected tree
- `tree_set1.png`, `tree_set2.png`: top levels of the two compared trees

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
