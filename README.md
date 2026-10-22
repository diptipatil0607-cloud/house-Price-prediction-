# House Price Prediction – Regression

**Workflow:** Dataset → EDA → Preprocessing → Train/Test Split → ML Model → Evaluation → Results → Conclusion

## 1. Problem & Objective
Predict the **median house value of a California district** from census features
(income, house age, rooms, location, etc.). This is a **supervised regression**
problem. Objective: build a model that predicts prices accurately on unseen data
and identify which features drive price most.

## 2. Dataset
- **Name:** California Housing (1990 U.S. Census)
- **Source:** Pace, R. K. & Barry, R. (1997), *Sparse Spatial Autoregressions*,
  Statistics & Probability Letters. Loaded via scikit-learn:
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- **Size:** 20,640 rows, 8 features + 1 target
- **Target variable:** `MedHouseVal` (median house value, in $100,000s)
- **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## 3. How to run
```bash
pip install -r requirements.txt
python house_price_prediction.py
```
Needs internet on the first run (scikit-learn downloads the dataset). If you
are offline, place a CSV with the same columns at `data/housing.csv`.

The script prints everything to the console and saves to `outputs/`:
- `1_target_distribution.png`, `2_correlation_heatmap.png`, `3_top_feature_vs_price.png` (EDA graphs)
- `4_actual_vs_predicted.png`, `5_feature_importance.png` (model graphs)
- `results_summary.md` – **auto-generated report with your actual numbers**:
  missing values, duplicates, observations, metrics, results, conclusion

## 4. Steps performed
| Stage | What the code does |
|---|---|
| EDA | Shape, info, statistics, missing values, duplicates, 3 graphs with computed observations |
| Preprocessing | Drop duplicates, add `BedroomRatio` feature, median imputation, clip outliers (1st–99th percentile, from train data only), standard scaling |
| Split | 80% train / 20% test (`random_state=42`) |
| Models | Linear Regression (baseline) and **Random Forest Regressor** (main model) |
| Evaluation | MAE, RMSE, R² on the test set + 5-fold cross-validation R² |

## 5. Expected observations (verify with your graphs)
- Prices are right-skewed, and many districts sit at the maximum value (the target is capped).
- Median income (`MedInc`) is normally the strongest predictor of price.
- Latitude/Longitude matter too, since location strongly affects price.
- Random Forest usually beats Linear Regression because it captures non-linear patterns.

## 6. Results & Conclusion
Copy the final metrics and conclusion from `outputs/results_summary.md` after
running; it is filled with the real values from your run.
