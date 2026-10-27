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
- **Missing pressure:** Pressure readings of 0 were treated as missing values
- **Dropped columns:** `Daily Summary` (leaks the target), `Apparent Temperature (C)` (no improvement from temperature) and `Loud Cover` (all zeros).
- **Target grouping:** Grouped the many `Summary` labels into general classes: Rainy, Windy, Foggy, Cloudy and Clear. This was done for improving performance through generalisation.
- **Encoding:** Filled missing `Precip Type` values with "none" and one-hot encoded the column.
  

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
