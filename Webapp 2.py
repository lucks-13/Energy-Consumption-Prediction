import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Energy Consumption Prediction", layout="wide")

@st.cache_data
def load_model_data():
    with open('model_for_Web2.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_data = load_model_data()
except FileNotFoundError:
    st.error("Model file not found. Please run code.ipynb first for model_for_Web2.pkl.")
    st.stop()

st.title("Energy Consumption Prediction Dashboard")
st.caption("A Athena Award Project, Used all light weight high accuracy model derived from my analysis if you want highest accuracy use the below link for .pkl file")
st.markdown("[Click Here For Source Code](https://github.com/lucks-13/Energy-Consumption-Prediction)", unsafe_allow_html=True)
st.markdown("---")

st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weather Conditions")
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=40.0, value=15.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=30.0, max_value=95.0, value=65.0, step=0.1)
    wind_speed = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
    solar_radiation = st.slider("Solar Radiation (W/m²)", min_value=0.0, max_value=800.0, value=200.0, step=1.0)
    pressure = st.slider("Pressure (hPa)", min_value=990.0, max_value=1030.0, value=1013.0, step=0.1)

with col2:
    st.subheader("Time Features")
    month = st.selectbox("Month", range(1, 13), index=5)
    day_of_year = st.slider("Day of Year", min_value=1, max_value=365, value=150)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)
    week_of_year = st.slider("Week of Year", min_value=1, max_value=52, value=20)
    day_of_week = st.selectbox("Day of Week", range(7), index=2, format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    is_weekend = st.checkbox("Is Weekend")

season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
season = season_map[month]

if temperature <= 5:
    temp_category = 'Cold'
elif temperature <= 15:
    temp_category = 'Cool'
elif temperature <= 25:
    temp_category = 'Moderate'
else:
    temp_category = 'Hot'

st.markdown("---")
st.header("Model Selection")
model_names = list(model_data['models'].keys())

col1, col2, col3, col4, col5 = st.columns(5)
selected_models = []

with col1:
    if st.checkbox("LinearRegression", value=True):
        selected_models.append("LinearRegression")
    if st.checkbox("Ridge", value=True):
        selected_models.append("Ridge")
    if st.checkbox("Lasso", value=True):
        selected_models.append("Lasso")

with col2:
    if st.checkbox("ElasticNet"):
        selected_models.append("ElasticNet")
    if st.checkbox("DecisionTree"):
        selected_models.append("DecisionTree")
    if st.checkbox("RandomForest", value=True):
        selected_models.append("RandomForest")

with col3:
    if st.checkbox("GradientBoosting", value=True):
        selected_models.append("GradientBoosting")
    if st.checkbox("AdaBoost"):
        selected_models.append("AdaBoost")
    if st.checkbox("ExtraTrees"):
        selected_models.append("ExtraTrees")

with col4:
    if st.checkbox("Bagging"):
        selected_models.append("Bagging")
    if st.checkbox("SVR"):
        selected_models.append("SVR")
    if st.checkbox("KNN"):
        selected_models.append("KNN")

with col5:
    if st.checkbox("MLP"):
        selected_models.append("MLP")
    if st.checkbox("VotingRegressor"):
        selected_models.append("VotingRegressor")
    if st.checkbox("StackingRegressor"):
        selected_models.append("StackingRegressor")

st.markdown("---")
predict_button = st.button("Start Prediction", type="primary", use_container_width=True)

if selected_models and predict_button:
    input_features = {
        'Temperature_C': temperature,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        'Solar_Radiation': solar_radiation,
        'Pressure': pressure,
        'Month': month,
        'Day_of_Year': day_of_year,
        'Quarter': quarter,
        'Week_of_Year': week_of_year,
        'Day_of_Week': day_of_week,
        'Is_Weekend': int(is_weekend),
        'Temp_Humidity_Interaction': temperature * humidity,
        'Wind_Solar_Interaction': wind_speed * solar_radiation,
        'Pressure_Temp_Interaction': pressure * temperature,
        'Temperature_Squared': temperature ** 2,
        'Humidity_Squared': humidity ** 2,
        'Wind_Speed_Sqrt': np.sqrt(wind_speed),
        'Temp_MA_7': temperature,
        'Temp_MA_30': temperature,
        'Energy_MA_7': 1000,
        'Temp_Lag_1': temperature,
        'Temp_Lag_7': temperature,
        'Energy_Lag_1': 1000,
        'Season_Encoded': model_data['le_season'].transform([season])[0],
        'Temp_Category_Encoded': model_data['le_temp_cat'].transform([temp_category])[0]
    }
    
    for col in model_data['season_dummies_columns']:
        input_features[col] = 1 if col == f'Season_{season}' else 0
    
    for col in model_data['temp_cat_dummies_columns']:
        input_features[col] = 1 if col == f'TempCat_{temp_category}' else 0
    
    input_df = pd.DataFrame([input_features])
    input_df = input_df.reindex(columns=model_data['feature_columns'], fill_value=0)
    
    st.markdown("---")
    st.header("Predictions")
    predictions = {}
    
    for model_name in selected_models:
        model = model_data['models'][model_name]
        scaler_name = model_data['scaler_mapping'].get(model_name, 'StandardScaler')
        
        if model_name in ['VotingRegressor', 'StackingRegressor']:
            scaler = model_data['scalers']['StandardScaler']
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
        elif scaler_name in model_data['scalers']:
            scaler = model_data['scalers'][scaler_name]
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
        else:
            pred = model.predict(input_df)[0]
        
        predictions[model_name] = max(0, pred)
    
    first_batch = list(predictions.items())[:5]
    if first_batch:
        cols = st.columns(len(first_batch))
        for i, (model_name, pred) in enumerate(first_batch):
            with cols[i]:
                delta = f"{pred-1000:.1f}" if pred != 1000 else None
                st.metric(f"{model_name}", f"{pred:.2f} kWh", delta=delta)
    
    if len(predictions) > 5:
        remaining_predictions = list(predictions.items())[5:10]
        if remaining_predictions:
            cols2 = st.columns(len(remaining_predictions))
            for i, (model_name, pred) in enumerate(remaining_predictions):
                with cols2[i]:
                    delta = f"{pred-1000:.1f}" if pred != 1000 else None
                    st.metric(f"{model_name}", f"{pred:.2f} kWh", delta=delta)
    
    if len(predictions) > 10:
        final_batch = list(predictions.items())[10:]
        if final_batch:
            cols3 = st.columns(len(final_batch))
            for i, (model_name, pred) in enumerate(final_batch):
                with cols3[i]:
                    delta = f"{pred-1000:.1f}" if pred != 1000 else None
                    st.metric(f"{model_name}", f"{pred:.2f} kWh", delta=delta)
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_pred = np.mean(list(predictions.values()))
        std_pred = np.std(list(predictions.values()))
        min_pred = np.min(list(predictions.values()))
        max_pred = np.max(list(predictions.values()))
        
        st.subheader("Summary Statistics")
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.metric("Average Prediction", f"{avg_pred:.2f} kWh")
        with met_col2:
            st.metric("Prediction Range", f"{max_pred-min_pred:.2f} kWh")
    
    with col2:
        fig_pred = px.bar(x=list(predictions.keys()), y=list(predictions.values()),
                         title="Predictions by Selected Models",
                         color=list(predictions.values()),
                         color_continuous_scale='viridis')
        fig_pred.update_layout(xaxis_title="Models", yaxis_title="Energy Consumption (kWh)",
                              xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_pred, use_container_width=True)

elif selected_models and not predict_button:
    st.info("Click the 'Start Prediction' button to generate predictions with the selected models.")
elif not selected_models:
    st.warning("Please select at least one model to make predictions.")

st.markdown("---")
st.header("Comprehensive Visualizations")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Performance", "Feature Analysis", "Data Insights", "Correlation Analysis", "Time Series Analysis"])

with tab1:
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scores_df = pd.DataFrame(model_data['model_scores']).T
        fig_r2 = px.bar(scores_df.reset_index(), x='index', y='R2',
                       title="R² Score Comparison",
                       color='R2', color_continuous_scale='blues')
        fig_r2.update_layout(xaxis_title="Models", yaxis_title="R² Score", xaxis_tickangle=-45)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        fig_rmse = px.bar(scores_df.reset_index(), x='index', y='RMSE',
                         title="RMSE Comparison",
                         color='RMSE', color_continuous_scale='reds')
        fig_rmse.update_layout(xaxis_title="Models", yaxis_title="RMSE", xaxis_tickangle=-45)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_mae = px.scatter(scores_df.reset_index(), x='R2', y='MAE', text='index',
                            title="R² vs MAE Performance",
                            size='RMSE', color='MSE')
        fig_mae.update_traces(textposition="top center")
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col4:
        performance_radar = scores_df[['R2', 'MAE', 'RMSE']].head(5)
        fig_radar = go.Figure()
        
        for idx, (model_name, row) in enumerate(performance_radar.iterrows()):
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['R2'], 1/row['MAE']*100, 1/row['RMSE']*10],
                theta=['R²', 'MAE (inv)', 'RMSE (inv)'],
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Top 5 Models Performance Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'RandomForest' in model_data['feature_importance']:
            feat_imp = model_data['feature_importance']['RandomForest']
            feat_data = list(feat_imp.items())
            feat_df = pd.DataFrame(feat_data, columns=['Feature', 'Importance'])
            feat_df = feat_df.sort_values('Importance', ascending=True).tail(15)
            fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                             title="Random Forest - Top 15 Features",
                             color='Importance', color_continuous_scale='greens')
            st.plotly_chart(fig_feat, use_container_width=True)
    
    with col2:
        weather_features = ['Temperature_C', 'Humidity', 'Wind_Speed', 'Solar_Radiation', 'Pressure']
        df_sample = model_data['df_features'].sample(min(1000, len(model_data['df_features'])))
        fig_3d = px.scatter_3d(df_sample, x='Temperature_C', y='Humidity', z='Energy_Consumption',
                              color='Wind_Speed', size='Solar_Radiation',
                              title="3D Weather vs Energy Consumption")
        st.plotly_chart(fig_3d, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        correlation_with_target = model_data['df_features'][weather_features + ['Energy_Consumption']].corr()['Energy_Consumption'].drop('Energy_Consumption')
        fig_corr_bar = px.bar(x=correlation_with_target.index, y=correlation_with_target.values,
                             title="Feature Correlation with Energy Consumption",
                             color=correlation_with_target.values,
                             color_continuous_scale='rdylbu')
        fig_corr_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_corr_bar, use_container_width=True)
    
    with col4:
        interaction_features = ['Temp_Humidity_Interaction', 'Wind_Solar_Interaction', 'Pressure_Temp_Interaction']
        df_interactions = model_data['df_features'][interaction_features + ['Energy_Consumption']]
        fig_parallel = px.parallel_coordinates(df_interactions.sample(500),
                                             title="Parallel Coordinates - Interaction Features",
                                             dimensions=interaction_features)
        st.plotly_chart(fig_parallel, use_container_width=True)

with tab3:
    st.subheader("Data Distribution and Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(model_data['df_features'], x='Energy_Consumption',
                               title="Energy Consumption Distribution",
                               nbins=50, marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_violin = px.violin(model_data['df_features'], y='Energy_Consumption', x='Season',
                              box=True, title="Energy Distribution by Season")
        st.plotly_chart(fig_violin, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        monthly_stats = model_data['df_features'].groupby('Month')['Energy_Consumption'].agg(['mean', 'std']).reset_index()
        fig_monthly = px.bar(monthly_stats, x='Month', y='mean', error_y='std',
                            title="Monthly Energy Consumption (Mean ± Std)")
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col4:
        hourly_pattern = model_data['df_features'].groupby('Day_of_Week')['Energy_Consumption'].mean().reset_index()
        fig_weekly = px.line(hourly_pattern, x='Day_of_Week', y='Energy_Consumption',
                            title="Weekly Energy Pattern", markers=True)
        fig_weekly.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(7)),
                               ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))
        st.plotly_chart(fig_weekly, use_container_width=True)

with tab4:
    st.subheader("Advanced Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_matrix = model_data['correlation_matrix']
        weather_features = ['Temperature_C', 'Humidity', 'Wind_Speed', 'Solar_Radiation', 'Pressure', 'Energy_Consumption']
        corr_subset = corr_matrix.loc[weather_features, weather_features]
        
        fig_corr = px.imshow(corr_subset, text_auto=True, aspect="auto",
                           title="Weather Features Correlation Matrix",
                           color_continuous_scale='rdbu')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        df_sample = model_data['df_features'].sample(min(800, len(model_data['df_features'])))
        fig_scatter_matrix = px.scatter_matrix(df_sample[weather_features[:4] + ['Energy_Consumption']],
                                             title="Scatter Matrix - Key Features")
        fig_scatter_matrix.update_layout(height=600)
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    
    pivot_data = model_data['df_features'].copy()
    pivot_data['Month'] = pivot_data.index.month
    pivot_data['Day_of_Week'] = pivot_data.index.dayofweek
    heatmap_data = pivot_data.pivot_table(values='Energy_Consumption', 
                                         index='Month', columns='Day_of_Week', aggfunc='mean')
    
    fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto",
                           title="Energy Consumption Heatmap (Month vs Day of Week)",
                           labels=dict(x="Day of Week", y="Month", color="Energy (kWh)"))
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab5:
    st.subheader("Time Series Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_ts = model_data['df_features'].reset_index()
        df_sample_ts = df_ts.iloc[::5]
        fig_ts = px.line(df_sample_ts, x='Date', y='Energy_Consumption',
                        title="Energy Consumption Time Series")
        st.plotly_chart(fig_ts, use_container_width=True)
    
    with col2:
        df_ts['Year'] = df_ts['Date'].dt.year
        yearly_avg = df_ts.groupby('Year')['Energy_Consumption'].mean().reset_index()
        fig_yearly = px.bar(yearly_avg, x='Year', y='Energy_Consumption',
                           title="Annual Average Energy Consumption")
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        df_ts['Cumulative_Energy'] = df_ts['Energy_Consumption'].cumsum()
        df_sample_cum = df_ts.iloc[::10]
        fig_cum = px.area(df_sample_cum, x='Date', y='Cumulative_Energy',
                         title="Cumulative Energy Consumption")
        st.plotly_chart(fig_cum, use_container_width=True)
    
    with col4:
        rolling_mean = df_ts.set_index('Date')['Energy_Consumption'].rolling(window=30).mean().reset_index()
        fig_rolling = px.line(rolling_mean.iloc[::5], x='Date', y='Energy_Consumption',
                             title="30-Day Rolling Average")

        st.plotly_chart(fig_rolling, use_container_width=True)

