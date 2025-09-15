import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date

st.set_page_config(page_title="Energy Consumption Predictor", layout="wide")

@st.cache_data
def load_model_package():
    return joblib.load('model_for_Web1.pkl')

def prepare_input_features(temp, humidity, wind, solar, pressure, month, day_of_year, quarter, 
                          week, day_of_week, is_weekend, season, temp_category):
    
    model_package = load_model_package()
    le_season = model_package['label_encoders']['season']
    le_temp_cat = model_package['label_encoders']['temp_category']
    
    temp_humidity_int = temp * humidity
    wind_solar_int = wind * solar
    pressure_temp_int = pressure * temp
    temp_squared = temp ** 2
    humidity_squared = humidity ** 2
    wind_sqrt = np.sqrt(wind)
    
    temp_ma_7 = temp
    temp_ma_30 = temp
    energy_ma_7 = 1000
    temp_lag_1 = temp
    temp_lag_7 = temp
    energy_lag_1 = 1000
    
    season_encoded = le_season.transform([season])[0]
    temp_cat_encoded = le_temp_cat.transform([temp_category])[0]
    
    season_dummies = {f'Season_{s}': 0 for s in le_season.classes_}
    season_dummies[f'Season_{season}'] = 1
    
    temp_cat_dummies = {f'TempCat_{c}': 0 for c in le_temp_cat.classes_}
    temp_cat_dummies[f'TempCat_{temp_category}'] = 1
    
    feature_dict = {
        'Temperature_C': temp, 'Humidity': humidity, 'Wind_Speed': wind,
        'Solar_Radiation': solar, 'Pressure': pressure, 'Month': month,
        'Day_of_Year': day_of_year, 'Quarter': quarter, 'Week_of_Year': week,
        'Day_of_Week': day_of_week, 'Is_Weekend': is_weekend,
        'Temp_Humidity_Interaction': temp_humidity_int,
        'Wind_Solar_Interaction': wind_solar_int,
        'Pressure_Temp_Interaction': pressure_temp_int,
        'Temperature_Squared': temp_squared, 'Humidity_Squared': humidity_squared,
        'Wind_Speed_Sqrt': wind_sqrt, 'Temp_MA_7': temp_ma_7,
        'Temp_MA_30': temp_ma_30, 'Energy_MA_7': energy_ma_7,
        'Temp_Lag_1': temp_lag_1, 'Temp_Lag_7': temp_lag_7,
        'Energy_Lag_1': energy_lag_1, 'Season_Encoded': season_encoded,
        'Temp_Category_Encoded': temp_cat_encoded
    }
    
    feature_dict.update(season_dummies)
    feature_dict.update(temp_cat_dummies)
    
    feature_vector = []
    for col in model_package['feature_columns']:
        if col in feature_dict:
            feature_vector.append(feature_dict[col])
        else:
            feature_vector.append(0)
    
    return np.array(feature_vector).reshape(1, -1)

def main():
    st.title("Energy Consumption Prediction Dashboard")
    st.caption("A Athena Award Project, Used all light weight high accuracy model derived from my analysis if you want highest accuracy use the below link for .pkl file")
    st.markdown("[Click Here For Source Code](https://github.com/lucks-13/Energy-Consumption-Prediction)", unsafe_allow_html=True)

    
    model_package = load_model_package()
    models = model_package['models']
    scaler = model_package['scaler']
    original_data = model_package['original_data']
    correlation_matrix = model_package['correlation_matrix']
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Performance"])
    
    with tab1:
        st.header("Energy Consumption Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Parameters")
            
            temp = st.slider("Temperature (°C)", -10.0, 40.0, 15.0, 0.1)
            humidity = st.slider("Humidity (%)", 30.0, 95.0, 65.0, 0.1)
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 2.0, 0.1)
            solar_radiation = st.slider("Solar Radiation (W/m²)", 0.0, 800.0, 200.0, 1.0)
            pressure = st.slider("Pressure (hPa)", 990.0, 1030.0, 1013.0, 0.1)
            
            selected_date = st.date_input("Date", date.today())
            month = selected_date.month
            day_of_year = selected_date.timetuple().tm_yday
            quarter = (month - 1) // 3 + 1
            week = selected_date.isocalendar()[1]
            day_of_week = selected_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
            season = season_map[month]
            
            if temp <= 5:
                temp_category = 'Cold'
            elif temp <= 15:
                temp_category = 'Cool'
            elif temp <= 25:
                temp_category = 'Moderate'
            else:
                temp_category = 'Hot'
            
            st.subheader("Select Models")
            selected_models = []
            for model_name in models.keys():
                if st.checkbox(model_name, value=True):
                    selected_models.append(model_name)
        
        with col2:
            if selected_models:
                st.subheader("Predictions")
                
                features = prepare_input_features(temp, humidity, wind_speed, solar_radiation,
                                                pressure, month, day_of_year, quarter, week,
                                                day_of_week, is_weekend, season, temp_category)
                features_scaled = scaler.transform(features)
                
                predictions = {}
                for model_name in selected_models:
                    pred = models[model_name].predict(features_scaled)[0]
                    predictions[model_name] = pred
                
                pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Prediction'])
                
                fig = px.bar(pred_df, x='Model', y='Prediction', 
                           title='Energy Consumption Predictions by Model',
                           color='Prediction', color_continuous_scale='viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
                
                avg_prediction = np.mean(list(predictions.values()))
                st.metric("Average Prediction", f"{avg_prediction:.2f} kWh")
                
                st.subheader("Model Comparison")
                for model_name, pred in predictions.items():
                    score = model_package['model_scores'][model_name]
                    st.write(f"**{model_name}**: {pred:.2f} kWh (R² = {score['R2']:.3f})")
    
    with tab2:
        st.header("Energy Consumption Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Energy Consumption Over Time")
            fig1 = px.line(x=original_data.index, y=original_data['Energy_Consumption'],
                          title='Energy Consumption Time Series')
            fig1.update_layout(xaxis_title='Date', yaxis_title='Energy (kWh)')
            st.plotly_chart(fig1, width='stretch')
            
            st.subheader("Energy Distribution")
            fig2 = px.histogram(original_data, x='Energy_Consumption', nbins=50,
                               title='Energy Consumption Distribution')
            st.plotly_chart(fig2, width='stretch')
        
        with col2:
            st.subheader("Temperature vs Energy")
            fig3 = px.scatter(original_data.sample(min(2000, len(original_data))), 
                             x='Temperature_C', y='Energy_Consumption',
                             color='Humidity', title='Temperature vs Energy (colored by Humidity)',
                             color_continuous_scale='viridis')
            st.plotly_chart(fig3, width='stretch')
            
            st.subheader("Monthly Average Energy")
            monthly_avg = original_data.groupby('Month')['Energy_Consumption'].mean().reset_index()
            fig4 = px.bar(monthly_avg, x='Month', y='Energy_Consumption',
                         title='Average Energy Consumption by Month')
            st.plotly_chart(fig4, width='stretch')
        
        st.subheader("Correlation Matrix")
        numeric_cols = ['Temperature_C', 'Humidity', 'Wind_Speed', 'Solar_Radiation', 
                       'Pressure', 'Energy_Consumption']
        corr_data = correlation_matrix.loc[numeric_cols, numeric_cols]
        
        fig5 = px.imshow(corr_data, text_auto=True, aspect="auto",
                        title='Feature Correlation Matrix', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig5, width='stretch')
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Energy by Season")
            seasonal_data = original_data.groupby('Season')['Energy_Consumption'].agg(['mean', 'std']).reset_index()
            fig6 = px.bar(seasonal_data, x='Season', y='mean', error_y='std',
                         title='Average Energy Consumption by Season')
            st.plotly_chart(fig6, width='stretch')
        
        with col4:
            st.subheader("Daily Pattern")
            daily_avg = original_data.groupby('Day_of_Week')['Energy_Consumption'].mean().reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_avg['Day_Name'] = daily_avg['Day_of_Week'].apply(lambda x: day_names[x])
            fig7 = px.line(daily_avg, x='Day_Name', y='Energy_Consumption', markers=True,
                          title='Average Energy by Day of Week')
            st.plotly_chart(fig7, width='stretch')
        
        st.subheader("Feature Relationships")
        feature_options = ['Temperature_C', 'Humidity', 'Wind_Speed', 'Solar_Radiation', 'Pressure']
        col_x = st.selectbox("X-axis feature", feature_options, index=0)
        col_y = st.selectbox("Y-axis feature", ['Energy_Consumption'] + feature_options, index=0)
        
        sample_data = original_data.sample(min(3000, len(original_data)))
        fig8 = px.scatter(sample_data, x=col_x, y=col_y, opacity=0.6,
                         title=f'{col_x} vs {col_y} Relationship')
        st.plotly_chart(fig8, width='stretch')
    
    with tab3:
        st.header("Model Performance Analysis")
        
        scores_df = pd.DataFrame(model_package['model_scores']).T
        scores_df = scores_df.round(4)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Scores Comparison")
            st.dataframe(scores_df)
            
            fig9 = px.bar(scores_df.reset_index(), x='index', y='R2',
                         title='Model R² Scores Comparison', color='R2',
                         color_continuous_scale='viridis')
            fig9.update_layout(xaxis_title='Model', yaxis_title='R² Score')
            st.plotly_chart(fig9, width='stretch')
        
        with col2:
            st.subheader("Error Metrics")
            fig10 = px.bar(scores_df.reset_index(), x='index', y='MAE',
                          title='Mean Absolute Error by Model', color='MAE',
                          color_continuous_scale='Reds_r')
            fig10.update_layout(xaxis_title='Model', yaxis_title='MAE')
            st.plotly_chart(fig10, width='stretch')
            
            fig11 = px.bar(scores_df.reset_index(), x='index', y='MSE',
                          title='Mean Squared Error by Model', color='MSE',
                          color_continuous_scale='Reds_r')
            fig11.update_layout(xaxis_title='Model', yaxis_title='MSE')
            st.plotly_chart(fig11, width='stretch')
        
        st.subheader("Model Performance Summary")
        best_model = scores_df.loc[scores_df['R2'].idxmax()]
        st.success(f"**Best Model**: {scores_df['R2'].idxmax()} with R² = {best_model['R2']:.4f}")
        
        performance_metrics = pd.DataFrame({
            'Metric': ['Best R²', 'Lowest MAE', 'Lowest MSE'],
            'Model': [scores_df['R2'].idxmax(), scores_df['MAE'].idxmin(), scores_df['MSE'].idxmin()],
            'Value': [scores_df['R2'].max(), scores_df['MAE'].min(), scores_df['MSE'].min()]
        })
        st.dataframe(performance_metrics)

if __name__ == "__main__":
    main()
