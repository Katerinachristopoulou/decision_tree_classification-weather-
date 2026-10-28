# decision_tree_classification-weather-

Weather type prediction using decision trees (scikit-learn) | Data mining exercise

## What it does

`main.py` downloads the [Weather History dataset](https://www.kaggle.com/datasets/muthuj7/weather-dataset)
from Kaggle and predicts the hourly weather summary, grouped into 5 classes
(Clear, Cloudy, Foggy, Rainy, Windy) using the following predictive parameters:

- **Temperature** (°C)
- **Humidity** (0-1)
- **Wind speed** (km/h)
- **Wind bearing** (degrees)
- **Visibility** (km)
- **Pressure** (millibars; readings of 0 treated as missing)
- **Hour** of the day (extracted from the timestamp)
- **Month** (extracted from the timestamp)
- **Precipitation type** (rain / snow / none, one-hot encoded)
- Excluded columns: `Apparent Temperature (C)` (almost a copy of temperature) and `Daily Summary` (leaks the target), plus `Loud Cover`, which is all zeros.
  
## Methodology

### Preprocessing

- **Time features:** Extracted `Hour` and `Month` from the date, since weather depends on the time of day and the season.
- **Missing pressure:** Pressure readings of 0 were treated as missing values.
- **Dropped columns:** `Daily Summary` (leaks the target), `Apparent Temperature (C)` (adds nothing beyond temperature) and `Loud Cover` (all zeros).

  ### Target Grouping

The original `Summary` column contains many very specific labels (e.g. "Breezy and Overcast"). These were generalised into 5 categories (Rainy, Windy, Foggy, Cloudy and Clear), which are descriptive enough to be meaningful and still predictable from the weather measurements.

Labels that mention several conditions were assigned to the rarer, more specific one, using the priority Rainy > Windy > Foggy > Cloudy > Clear.

| Class | Assigned when the label contains |
|---|---|
| Rainy | Rain, Drizzle |
| Windy | Breezy, Windy |
| Foggy | Foggy |
| Cloudy | Cloudy, Overcast |
| Clear | Anything else (Clear, Dry and their combinations) |

For example, "Breezy and Overcast" becomes **Windy**, because Windy ranks above Cloudy.
### Class Imbalance

The classes are very imbalanced: most hours are Cloudy, while classes like Foggy and Rainy are rare. A model that always predicts Cloudy would therefore score a high accuracy without being actually able to learn to recognise patterns. Two measures were used to deal with this:

- **Class weights:** The trees were trained with `class_weight="balanced"`, so penalisation is based on rarity.Classes that are more rare have heavier penalisation
- **Macro F1 for model selection:** Models were chosen by macro F1, which is the average of F1 score of the 5 indvidual classes so that every class counts equally. Accuracy is still reported, but only as a secondary measure.
- 
### Model Selection

- **20% stratified hold-out test set:** The model is trained 80% of the data and then makes predictions on the remaining 20%, which is not known. Stratified means every class keeps the same share in both training and testing. The test set is not used as a parameter during selection, so the performance is evaluated with more honesty.
- **5-fold cross-validation over `max_depth` (1-15):** The training data is split into 5 parts, and each part becomes a validation model for the remaining testing sets,taking turns. This is repeated for every depth from 1 to 15 and scored by macro F1. It stops the tree from growing too deep, because a overly deep tree memorises the training data and does worse on unseen data (overfitting problem).
- **Grid search over `max_depth`, `min_samples_leaf` and `ccp_alpha`:** Every combination of these parameters is evaluated repeatedly with 5-fold CV, and the combination with the best macro F1 is chosen. This finds the best model after many evaluations, and it also tests pruning (`ccp_alpha`) and leaf size, which the depth search alone does not.

Best depth: **XX**. Best grid search parameters: **{...}** .


## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
