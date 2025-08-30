
# Streamlit Dashboard for TP2 Assessment 4 (NHI – Demand Prediction)

## Run locally
1) Ensure Python 3.10+ is installed.
2) Create/activate a virtual environment (optional but recommended).
3) Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4) Place real CSVs in `data/` (if available):
   - `ghs2024_households.csv`
   - `ghs2024_persons.csv`
   - `ghs2024_health_visits.csv`
   - `facilities_masterlist.csv`
   > If these are missing, the app will use sample data automatically.
5) Start the app:
   ```bash
   streamlit run app.py
   ```

## Tabs (at least two as required)
- **Overview**: KPIs & macro summaries
- **Explore Data**: Distributions, filters, previews
- **Model & Insights**: Train/evaluate a Random Forest, confusion matrix, province-level demand, feature importances

## Notes
- App respects your filters (province, urban/rural) in the sidebar.
- For assessment, include this dashboard in your repo to cover presentation marks.
