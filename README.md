# PowerCo Customer Churn Analysis

Exploratory data analysis, feature engineering and a Random Forest model to predict customer churn for an energy supplier. Completed as part of the **BCG X Data Science Job Simulation** on Forage (June – October 2026).

## Business Problem

PowerCo supplies electricity and gas. About 1 in 10 small and medium-sized business customers leaves (churn), and the company has no reliable way to predict who will go. The goal is to identify at-risk customers early enough for the retention team to act.

## Project Structure

| File | Content |
|---|---|
| `01_Exploratory_Data_Analysis.ipynb` | Data types, descriptive statistics, hidden missing values, outliers, visualizations |
| `02_Feature_Engineering.ipynb` | Date, price, consumption and categorical features |
| `03_Modeling_Random_Forest.ipynb` | Train/test split, Random Forest, threshold tuning, evaluation |
| `Executive_Summary.pdf` | One-page summary with recommendations |

*The raw datasets are not included in this repository.*

## Approach

1. **Exploratory data analysis (EDA):** checked data types, hidden missing values, skewed distributions and the class balance.
2. **Feature engineering:** built price-change features (January vs. December, monthly averages and maximum monthly differences), tenure and date features, forecast-vs-actual consumption features, encoded categorical columns and log-transformed skewed consumption columns. The final dataset has 14,606 rows and 76 columns.
3. **Modeling:** trained a Random Forest (200 trees, `class_weight="balanced"`) on a stratified 75/25 train/test split, then compared decision thresholds on a separate validation set.

## Key Findings

- **Imbalanced target:** only about 9.7% of customers churned, so accuracy alone is misleading.
- **Hidden missing values:** the data has no `NaN` values, but missing data hides in other forms, for example a `MISSING` category in `channel_sales` (about 25.5%) and many `0` values in `forecast_cons_year` (about 42%).
- **Skewed consumption data:** consumption columns are strongly right-skewed with extreme outliers, so I applied a log transformation.
- **Avoiding data leakage:** with a prediction cutoff of 1 January 2016, dates in the future (61 product-modification dates and 903 renewal dates) were treated as unavailable when creating date-based features.

## Model Results

Test set: 3,652 customers, of which 355 churned. ROC-AUC: **0.698**. Average precision: **0.305** (baseline 0.097).

| Threshold | Precision | Recall | F1 | Churners found |
|---|---|---|---|---|
| 0.5 | 0.521 | 0.177 | 0.265 | 63 of 355 |
| 0.2 | 0.164 | 0.634 | 0.260 | 225 of 355 |

A low threshold finds more churners but flags many customers who would have stayed. A high threshold is more precise but misses most churners. The right choice depends on the retention team's capacity and cost per follow-up. The model can help prioritize customers, but its results are not strong enough to be the only basis for decisions.

## Business Recommendation

Rank customers by churn risk and focus retention offers on the highest-risk group. Test a targeted offer on a high-risk group against a control group and measure retention before scaling up. See `Executive_Summary.pdf`.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook, VS Code

## Author

Israel P. Sebastian Joksan Asbanu, Bandung, Indonesia
sebastianjoksan@gmail.com
