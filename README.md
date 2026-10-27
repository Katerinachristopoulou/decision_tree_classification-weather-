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
- **Target grouping:** Specific `Summary` labels (e.g. "Breezy and Overcast") were generalised into 5 categories (Rainy, Windy, Foggy, Cloudy, Clear) that are both descriptive and predictable, which also improves performance through generalisation. Labels with several conditions take the rarer one (Rainy > Windy > Foggy > Cloudy > Clear).
- **Encoding:** Filled missing `Precip Type` values with "none" and one-hot encoded the column.

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



## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
