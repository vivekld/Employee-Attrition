import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TalentPredict Pro | Executive Intelligence", 
    page_icon="🏢", 
    layout="wide"
)

# --- ADVANCED UI THEME ---
st.markdown("""
    <style>
    /* Professional Dark Theme */
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stApp { background: radial-gradient(circle at 50% 50%, #161b22 0%, #0d1117 100%); }
    
    /* Executive Metric Cards */
    div[data-testid="stMetricContainer"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Result Cards */
    .res-card {
        padding: 30px;
        border-radius: 15px;
        margin-top: 10px;
        border-left: 12px solid;
        background: rgba(255, 255, 255, 0.03);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .risk-high { border-color: #f85149; color: #ff7b72; }
    .risk-med { border-color: #d29922; color: #e3b341; }
    .risk-low { border-color: #238636; color: #56d364; }
    
    .section-header { 
        color: #58a6ff; 
        border-bottom: 2px solid #30363d; 
        padding-bottom: 8px; 
        margin-top: 30px; 
        font-weight: bold; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Dashboard Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        padding: 10px 20px;
        font-weight: 600;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 3px solid #58a6ff !important; }
    </style>
""", unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_full_data():
    df = pd.read_csv('data.csv')
    # Filter constants/IDs
    exclude = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df_clean = df.drop(columns=exclude)
    
    # Encoders for prediction
    encoders = {}
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    df_model = df_clean.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le
    
    df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})
    return df, df_clean, df_model, encoders

df_raw, df_clean, df_model, encoders = load_full_data()

@st.cache_resource
def train_ai(df):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist()

ai_model, feature_names = train_ai(df_model)

# --- SIDEBAR & FILTERS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("TalentGuard Pro")
    st.markdown("`v3.0 Enterprise Intelligence`")
    st.divider()
    
    nav = st.radio("Navigation", ["🏢 Executive Dashboard", "👤 Employee Profiler", "🔎 Advanced Data Explorer"])
    
    st.divider()
    st.subheader("Global Data Filters")
    selected_dept = st.multiselect("Department", df_raw['Department'].unique(), default=df_raw['Department'].unique())
    selected_gender = st.multiselect("Gender", df_raw['Gender'].unique(), default=df_raw['Gender'].unique())
    selected_level = st.slider("Job Level", 1, 5, (1, 5))
    
    # Apply filters to global dataframe
    filtered_df = df_raw[
        (df_raw['Department'].isin(selected_dept)) & 
        (df_raw['Gender'].isin(selected_gender)) & 
        (df_raw['JobLevel'] >= selected_level[0]) & 
        (df_raw['JobLevel'] <= selected_level[1])
    ]

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if nav == "🏢 Executive Dashboard":
    st.title("📊 Workforce Strategic Dashboard")
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Headcount", len(filtered_df))
    att_rate = (filtered_df['Attrition'] == 'Yes').mean()
    k2.metric("Attrition Rate", f"{att_rate:.1%}", delta=f"{att_rate-0.16:.1%}", delta_color="inverse")
    k3.metric("Avg. Monthly Salary", f"${filtered_df['MonthlyIncome'].mean():,.0f}")
    k4.metric("Work-Life Balance", f"{filtered_df['WorkLifeBalance'].mean():.2f}/4")
    
    st.divider()
    
    dash_tabs = st.tabs(["🔥 Turnover Insights", "📍 Satisfaction Analysis", "💰 Compensation Strategy"])
    
    with dash_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            # Attrition by Overtime
            fig = px.histogram(filtered_df, x="OverTime", color="Attrition", barmode="group",
                             title="Impact of Overtime on Attrition", 
                             color_discrete_sequence=['#238636', '#da3633'], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Age Group Analysis
            filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=[18, 25, 35, 45, 60], labels=['18-25', '26-35', '36-45', '46+'])
            fig = px.histogram(filtered_df, x="Age_Group", color="Attrition", barmode="group",
                             title="Attrition Risk by Age Group", 
                             color_discrete_sequence=['#238636', '#da3633'], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        # Attrition Heatmap - Job Role vs Department
        st.subheader("High-Risk Segments (Department vs Job Role)")
        heat_data = filtered_df[filtered_df['Attrition'] == 'Yes'].groupby(['Department', 'JobRole']).size().reset_index(name='Leavers')
        fig = px.bar(heat_data, x="JobRole", y="Leavers", color="Department", 
                     title="Concentration of Churn by Role", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with dash_tabs[1]:
        # Satisfaction Radar Chart for the filtered group
        st.subheader("Cultural Health Map")
        sat_metrics = {
            'Job Satisfaction': filtered_df['JobSatisfaction'].mean(),
            'Environment': filtered_df['EnvironmentSatisfaction'].mean(),
            'Work-Life Balance': filtered_df['WorkLifeBalance'].mean(),
            'Relationships': filtered_df['RelationshipSatisfaction'].mean(),
            'Involvement': filtered_df['JobInvolvement'].mean()
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(sat_metrics.values()),
            theta=list(sat_metrics.keys()),
            fill='toself',
            name='Current Filtered Group',
            line_color='#58a6ff'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 4])), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with dash_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(filtered_df, x="Attrition", y="PercentSalaryHike", color="Attrition",
                        title="Salary Hike (%) vs. Retention", color_discrete_sequence=['#238636', '#da3633'], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(filtered_df, x="TotalWorkingYears", y="MonthlyIncome", color="Attrition",
                            title="Income vs. Career Length Profile", template="plotly_dark",
                            color_discrete_sequence=['#238636', '#da3633'])
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: EMPLOYEE PROFILER ---
elif nav == "👤 Employee Profiler":
    st.title("🧠 Predictive Risk Assessment")
    st.info("Input all 30 employee attributes to generate an AI-driven risk report.")
    
    with st.form("risk_form"):
        # Grid layout for 30 features
        st.markdown('<div class="section-header">Demographics & Experience</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        age = col1.slider("Age", 18, 60, 32)
        gender = col2.selectbox("Gender", encoders['Gender'].classes_)
        marital = col3.selectbox("Marital Status", encoders['MaritalStatus'].classes_)
        ed_field = col4.selectbox("Education Field", encoders['EducationField'].classes_)
        
        total_yrs = col1.number_input("Total Career Years", 0, 40, 10)
        co_yrs = col2.number_input("Years at Company", 0, 40, 5)
        dist = col3.number_input("Distance from Home (km)", 1, 30, 5)
        prev_comp = col4.number_input("Prev Companies Worked", 0, 10, 2)

        st.markdown('<div class="section-header">Role & Engagement</div>', unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)
        dept = col5.selectbox("Department", encoders['Department'].classes_)
        role = col6.selectbox("Job Role", encoders['JobRole'].classes_)
        level = col7.slider("Job Level", 1, 5, 2)
        travel = col8.selectbox("Travel", encoders['BusinessTravel'].classes_)
        
        ot = col5.radio("Overtime?", ["Yes", "No"], horizontal=True)
        job_sat = col6.select_slider("Job Sat", [1,2,3,4], value=3)
        env_sat = col7.select_slider("Env Sat", [1,2,3,4], value=3)
        wl_bal = col8.select_slider("WLB", [1,2,3,4], value=3)

        st.markdown('<div class="section-header">Compensation & Performance</div>', unsafe_allow_html=True)
        col9, col10, col11, col12 = st.columns(4)
        income = col9.number_input("Monthly Income ($)", 1000, 20000, 5000)
        hike = col10.slider("Salary Hike %", 11, 25, 12)
        stock = col11.select_slider("Stock Option", [0,1,2,3], value=0)
        perf = col12.selectbox("Performance", [3, 4])
        
        # Additional features for completeness
        role_yrs = col9.number_input("Years in Role", 0, 20, 2)
        mgr_yrs = col10.number_input("Years with Mgr", 0, 20, 2)
        promo_yrs = col11.number_input("Years since Promo", 0, 15, 1)
        training = col12.slider("Trainings last yr", 0, 6, 2)
        
        # Final inputs for hidden variables
        rel_sat = 3; job_inv = 3; d_rate = 800; h_rate = 65; m_rate = 10000; ed_lvl = 3
        
        submit = st.form_submit_button("🚀 GENERATE AI RISK REPORT")

    if submit:
        # Construct Input
        input_data = {
            'Age': age, 'BusinessTravel': travel, 'DailyRate': d_rate, 'Department': dept,
            'DistanceFromHome': dist, 'Education': ed_lvl, 'EducationField': ed_field,
            'EnvironmentSatisfaction': env_sat, 'Gender': gender, 'HourlyRate': h_rate,
            'JobInvolvement': job_inv, 'JobLevel': level, 'JobRole': role,
            'JobSatisfaction': job_sat, 'MaritalStatus': marital, 'MonthlyIncome': income,
            'MonthlyRate': m_rate, 'NumCompaniesWorked': prev_comp, 'OverTime': ot,
            'PercentSalaryHike': hike, 'PerformanceRating': perf, 'RelationshipSatisfaction': rel_sat,
            'StockOptionLevel': stock, 'TotalWorkingYears': total_yrs, 'TrainingTimesLastYear': training,
            'WorkLifeBalance': wl_bal, 'YearsAtCompany': co_yrs, 'YearsInCurrentRole': role_yrs,
            'YearsSinceLastPromotion': promo_yrs, 'YearsWithCurrManager': mgr_yrs
        }
        
        # Predict
        vec = [encoders[f].transform([input_data[f]])[0] if f in encoders else input_data[f] for f in feature_names]
        prob = ai_model.predict_proba([vec])[0][1]
        
        st.divider()
        res1, res2 = st.columns([1, 2])
        with res1:
            cat = "HIGH RISK" if prob > 0.6 else "MODERATE" if prob > 0.3 else "LOW RISK"
            cls = "risk-high" if prob > 0.6 else "risk-med" if prob > 0.3 else "risk-low"
            st.markdown(f'<div class="res-card {cls}"><h2>{cat}</h2><p>Risk Score: {prob:.1%}</p></div>', unsafe_allow_html=True)
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=prob*100, gauge={'bar': {'color': "#58a6ff"}})).update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)'), use_container_width=True)
        with res2:
            st.subheader("Key Predictive Factors")
            imp = pd.Series(ai_model.feature_importances_, index=feature_names).sort_values().tail(10)
            st.plotly_chart(px.bar(imp, orientation='h', template="plotly_dark", color_discrete_sequence=['#58a6ff']).update_layout(height=350), use_container_width=True)

# --- PAGE 3: DATA EXPLORER ---
elif nav == "🔎 Advanced Data Explorer":
    st.title("📂 Enterprise Knowledge Base")
    st.dataframe(filtered_df.style.background_gradient(subset=['MonthlyIncome'], cmap='Blues'), use_container_width=True)
    
    st.subheader("Cross-Attribute Correlation")
    fig = px.imshow(df_model.corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', height=800)
    st.plotly_chart(fig, use_container_width=True)