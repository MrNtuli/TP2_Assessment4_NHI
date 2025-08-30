
import os
import pathlib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="NHI Demand Dashboard", layout="wide")

st.title("NHI – Predicting Healthcare Service Demand (South Africa)")
st.caption("Technical Programming 2 – Assessment 4 | Streamlit Dashboard")

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------- Data Loading -----------------------------
@st.cache_data(show_spinner=True)
def load_or_sample():
    def file_if_exists(name):
        p = DATA_DIR / name
        return p if p.exists() else None

    hh_path = file_if_exists('ghs2024_households.csv')
    pr_path = file_if_exists('ghs2024_persons.csv')
    hv_path = file_if_exists('ghs2024_health_visits.csv')
    fc_path = file_if_exists('facilities_masterlist.csv')

    use_sample = not all([hh_path, pr_path, hv_path, fc_path])

    if use_sample:
        np.random.seed(42)
        provinces = ['EC','FS','GP','KZN','LP','MP','NC','NW','WC']
        facility_types = ['Clinic','Hospital','CHC','GP']
        ownerships = ['Public','Private']

        facilities = pd.DataFrame({
            'facility_id': range(1,121),
            'province': np.random.choice(provinces, 120),
            'district': np.random.randint(1, 45, 120),
            'ownership': np.random.choice(ownerships, 120, p=[0.75,0.25]),
            'level': np.random.choice(['Primary','Secondary','Tertiary'], 120, p=[0.6,0.3,0.1]),
            'lat': -34 + np.random.rand(120)*10,
            'lon': 18 + np.random.rand(120)*8,
            'services_offered': np.random.choice(['General','Maternity','HIV/TB','Chronic'], 120)
        })

        households = pd.DataFrame({
            'hh_id': range(1,1001),
            'province': np.random.choice(provinces, 1000),
            'district': np.random.randint(1,45,1000),
            'urban_rural': np.random.choice(['Urban','Rural'], 1000, p=[0.65,0.35]),
            'income': np.random.lognormal(mean=9.5, sigma=0.6, size=1000).round(0),
            'household_size': np.random.randint(1,9,1000)
        })

        persons = pd.DataFrame({
            'person_id': range(1,3801),
            'hh_id': np.random.choice(households['hh_id'], 3800),
            'age': np.clip(np.random.normal(35, 18, 3800).round(0), 0, 95).astype(int),
            'sex': np.random.choice(['Male','Female'], 3800),
            'education': np.random.choice(['None','Primary','Secondary','Tertiary'], 3800, p=[0.05,0.25,0.5,0.2]),
            'employment': np.random.choice(['Employed','Unemployed','Inactive'], 3800, p=[0.45,0.35,0.2]),
            'medical_aid': np.random.choice([0,1], 3800, p=[0.75,0.25]),
            'chronic_condition': np.random.choice([0,1], 3800, p=[0.7,0.3])
        })

        hv = []
        for pid in persons['person_id']:
            n = np.random.poisson(1.7)
            for _ in range(n):
                fac = facilities.sample(1).iloc[0]
                hv.append({
                    'visit_id': len(hv)+1,
                    'person_id': pid,
                    'facility_id': int(fac['facility_id']),
                    'facility_type': np.random.choice(facility_types, p=[0.55,0.25,0.1,0.1]),
                    'reason': np.random.choice(['Acute','Chronic','Maternal','Injury','Other'], p=[0.35,0.25,0.1,0.1,0.2]),
                    'inpatient': np.random.choice([0,1], p=[0.9,0.1]),
                    'month': np.random.randint(1,13),
                    'distance_km': abs(np.random.normal(6, 4))
                })
        health_visits = pd.DataFrame(hv)
    else:
        households = pd.read_csv(hh_path)
        persons = pd.read_csv(pr_path)
        health_visits = pd.read_csv(hv_path)
        facilities = pd.read_csv(fc_path)

    return households, persons, health_visits, facilities, use_sample

households, persons, health_visits, facilities, using_sample = load_or_sample()

# ----------------------------- Feature Join -----------------------------
@st.cache_data(show_spinner=True)
def build_features(households, persons, health_visits, facilities):
    visit_counts = health_visits.groupby('person_id').size().rename('visits_year').reset_index()
    visit_counts['demand'] = (visit_counts['visits_year'] > 0).astype(int)
    persons_aug = persons.merge(visit_counts[['person_id','demand']], on='person_id', how='left')
    persons_aug['demand'] = persons_aug['demand'].fillna(0).astype(int)

    fac_per_district = facilities.groupby(['province','district']).size().rename('fac_count').reset_index()
    hh_pop = persons.groupby('hh_id').size().rename('hh_persons').reset_index()
    households_aug = households.merge(hh_pop, on='hh_id', how='left').fillna({'hh_persons':0})
    dist_pop = households_aug.groupby(['province','district'])['hh_persons'].sum().rename('district_pop').reset_index()
    dist_fac = fac_per_district.merge(dist_pop, on=['province','district'], how='left')
    dist_fac['fac_per_10k'] = (dist_fac['fac_count'] / dist_fac['district_pop'].replace(0, np.nan) * 10000).fillna(0)

    X = persons_aug.merge(households[['hh_id','province','district','urban_rural','income','household_size']],
                          on='hh_id', how='left')
    X = X.merge(dist_fac[['province','district','fac_per_10k']], on=['province','district'], how='left')
    X['fac_per_10k'] = X['fac_per_10k'].fillna(0)

    y = X['demand']
    return X, y

X, y = build_features(households, persons, health_visits, facilities)

# ----------------------------- Sidebar Filters -----------------------------
st.sidebar.header("Filters")
prov_options = sorted(X['province'].dropna().unique().tolist())
sel_prov = st.sidebar.multiselect("Province", options=prov_options, default=prov_options)
sel_ur = st.sidebar.multiselect("Urban/Rural", options=sorted(X['urban_rural'].dropna().unique().tolist()),
                                default=sorted(X['urban_rural'].dropna().unique().tolist()))

mask = X['province'].isin(sel_prov) & X['urban_rural'].isin(sel_ur)
Xf = X[mask].copy()
yf = y[mask].copy()

st.sidebar.markdown("---")
st.sidebar.write("**Data source mode:**")
if using_sample:
    st.sidebar.success("Using SAMPLE data")
else:
    st.sidebar.info("Using REAL CSVs from data/")

# ----------------------------- Tabs -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Explore Data", "Model & Insights"])

with tab1:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households", f"{len(households):,}")
    c2.metric("Persons", f"{len(persons):,}")
    c3.metric("Health Visits", f"{len(health_visits):,}")
    c4.metric("Facilities", f"{len(facilities):,}")

    # Province-level counts
    st.markdown("### Facility counts by province")
    st.bar_chart(facilities.groupby('province').size())

    st.markdown("### Population (proxy) by district")
    hh_pop = persons.groupby('hh_id').size().rename('hh_persons').reset_index()
    households_aug = households.merge(hh_pop, on='hh_id', how='left').fillna({'hh_persons':0})
    dist_pop = households_aug.groupby(['province','district'])['hh_persons'].sum().reset_index()
    st.dataframe(dist_pop.head(20))

with tab2:
    st.subheader("Explore Data")
    left, right = st.columns(2)
    with left:
        st.markdown("**Age distribution**")
        st.bar_chart(Xf['age'].value_counts().sort_index())

        st.markdown("**Income distribution (binned)**")
        if 'income' in Xf.columns:
            bins = pd.cut(Xf['income'], bins=20)
            st.bar_chart(bins.value_counts().sort_index())
    with right:
        st.markdown("**Facility type counts**")
        if 'facility_type' in health_visits.columns:
            st.bar_chart(health_visits['facility_type'].value_counts())

        st.markdown("**Urban vs Rural sample size**")
        st.bar_chart(Xf['urban_rural'].value_counts())

    st.markdown("**Raw preview (filtered)**")
    st.dataframe(Xf[['person_id','age','sex','education','employment','medical_aid','chronic_condition',
                     'province','urban_rural','income','household_size','fac_per_10k']].head(50))

with tab3:
    st.subheader("Model & Insights")

    # Train model on filtered subset
    numeric_features = ['age','income','household_size','fac_per_10k']
    categorical_features = ['sex','education','employment','medical_aid','chronic_condition','urban_rural','province']

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[('prep', preprocess), ('model', clf)])

    X_model = Xf[numeric_features + categorical_features]
    X_train, X_test, y_train, y_test = train_test_split(X_model, yf, test_size=0.2, random_state=42, stratify=yf)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]

    # Metrics
    metrics = {
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    }
    st.markdown("### Evaluation (filtered subset)")
    mcols = st.columns(len(metrics))
    for (k,v), col in zip(metrics.items(), mcols):
        col.metric(k, f"{v:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm_df)

    # Province-level probabilities
    all_prob = pipe.predict_proba(X_model)[:,1]
    scored = Xf.copy()
    scored['prob'] = all_prob

    st.markdown("### Predicted demand probability by province (filtered)")
    st.bar_chart(scored.groupby('province')['prob'].mean().sort_values())

    st.markdown("### Top feature importances")
    rf = pipe.named_steps['model']
    # Get OHE feature names
    ohe = pipe.named_steps['prep'].named_transformers_['cat']
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names
    importances = pd.Series(rf.feature_importances_, index=all_feature_names).sort_values(ascending=False).head(20)
    st.bar_chart(importances[::-1])

st.markdown("---")
st.caption("Tip: Place real CSVs in a `data/` folder (ghs2024_households.csv, ghs2024_persons.csv, ghs2024_health_visits.csv, facilities_masterlist.csv). The app will detect them automatically.")
