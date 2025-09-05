import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI-Powered Entomological Analysis",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E8B57;
        --secondary-color: #FF6B35;
        --accent-color: #4ECDC4;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #2E8B57 0%, #4ECDC4 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: var(--card-shadow);
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: var(--card-shadow);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--card-shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Navigation styling */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .nav-item:hover {
        background: var(--primary-color);
        color: white;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed var(--accent-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    
    /* Success/Warning/Error styling */
    .stSuccess {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 8px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        border-radius: 8px;
    }
    
    /* Risk assessment styling */
    .risk-low {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Feature highlight */
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
    }
</style>
""", unsafe_allow_html=True)

class PestAnalysisApp:
    def __init__(self):
        self.df = None
        self.column_mapping = {}
        self.models = {}
        self.scaler = None
        
    def map_columns(self, df):
        """Map actual column names to expected names"""
        column_variations = {
            'year': ['YEAR', 'Year', 'year', 'Yr'],
            'week': ['Std Week', 'Standard Week', 'Week', 'SMW', 'week', 'std_week'],
            'tmax': ['Tmax', 'T max', 'Maximum Temperature', 'Max Temp', 'tmax', 'TMAX'],
            'tmin': ['Tmin', 'T min', 'Minimum Temperature', 'Min Temp', 'tmin', 'TMIN'],
            'rh_max': ['RH max', 'RH Max', 'Maximum RH', 'Max RH', 'rh_max', 'RH_MAX'],
            'rh_min': ['RH min', 'RH Min', 'Minimum RH', 'Min RH', 'rh_min', 'RH_MIN'],
            'rainfall': ['RF', 'Rainfall', 'Rain', 'Precipitation', 'rf', 'RAINFALL'],
            'sunshine': ['BSS', 'Bright Sunshine Hours', 'Sunshine', 'bss', 'BSH'],
            'wind': ['Wspeed', 'Wind speed', 'Wind Speed', 'Wind', 'wind_speed', 'WIND'],
            'trap_catch': ['Trap Flies catch', 'Trap Flies Catch', 'Trap Catch', 'Flies Catch', 'Catch', 'trap_flies', 'TRAP']
        }
        
        actual_columns = df.columns.tolist()
        mapping = {}
        
        for standard_name, variations in column_variations.items():
            for variation in variations:
                if variation in actual_columns:
                    mapping[standard_name] = variation
                    break
        
        return mapping
    
    def load_and_process_data(self, uploaded_file):
        """Load and process uploaded data"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            self.df = df
            self.column_mapping = self.map_columns(df)
            
            # Clean data
            numeric_mappings = ['tmax', 'tmin', 'rh_max', 'rh_min', 'rainfall', 'sunshine', 'wind', 'trap_catch']
            for mapping_key in numeric_mappings:
                if mapping_key in self.column_mapping:
                    col_name = self.column_mapping[mapping_key]
                    self.df[col_name] = pd.to_numeric(self.df[col_name], errors='coerce')
                    self.df[col_name].fillna(self.df[col_name].median(), inplace=True)
            
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def train_models(self):
        """Train machine learning models"""
        if self.df is None or 'trap_catch' not in self.column_mapping:
            return False, "No data or target column found"
        
        weather_mappings = ['tmax', 'tmin', 'rh_max', 'rh_min', 'rainfall', 'sunshine', 'wind']
        available_cols = []
        
        for mapping in weather_mappings:
            if mapping in self.column_mapping:
                available_cols.append(self.column_mapping[mapping])
        
        if len(available_cols) == 0:
            return False, "No weather columns found"
        
        trap_col = self.column_mapping['trap_catch']
        
        # Prepare data
        X = self.df[available_cols].dropna()
        y = self.df.loc[X.index, trap_col]
        
        if len(X) < 10:
            return False, "Insufficient data for training"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            try:
                if name == 'Linear Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'scaled': name == 'Linear Regression'
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    results[name]['feature_importance'] = dict(zip(available_cols, model.feature_importances_))
                
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
        
        self.models = results
        return True, f"Successfully trained {len(results)} models"

def create_header():
    """Create modern header"""
    st.markdown("""
    <div class="main-header">
        <h1>🦟 Pest  Analysis </h1>
        <p>AI-Powered Entomological Analysis & Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

def create_info_card(title, content, icon="📊"):
    """Create styled info card"""
    st.markdown(f"""
    <div class="info-card">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle=""):
    """Create styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <h2>{value}</h2>
        <h4>{title}</h4>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Create header
    create_header()
    
    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = PestAnalysisApp()
    
    app = st.session_state.app
    
    # Enhanced Sidebar with modern styling
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Add some spacing and styling
        st.markdown("<br>", unsafe_allow_html=True)
        
        page = st.selectbox(
            "Choose a section:",
            [
                "🏠 Home",
                "📤 Data Upload", 
                "📊 Data Analysis", 
                "🔮 AI Prediction", 
                "📈 Visualizations",
                "🤖 Model Comparison",
                "ℹ️ About"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Add quick stats if data is loaded
        if app.df is not None:
            st.markdown("### 📈 Quick Stats")
            st.metric("Dataset Size", f"{app.df.shape[0]} rows")
            st.metric("Features", f"{len(app.column_mapping)} mapped")
            if app.models:
                best_model = max(app.models.items(), key=lambda x: x[1]['R²'])
                st.metric("Best Model R²", f"{best_model[1]['R²']:.3f}")
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        - 🤖 **AI Predictions**
        - 📊 **Statistical Analysis**
        - 📈 **Interactive Charts**
        - 🔍 **Pattern Recognition**
        - 📋 **Model Comparison**
        """)
    
    # Main content area
    if page == "🏠 Home":
        st.markdown("## Welcome to Pest Analysis DeshBoard!  🎓")
        st.markdown("## Designed By Makhan Kumbhkar!  🎓")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_info_card(
                "Upload Data", 
                "Start by uploading your entomological dataset with weather variables and trap catch data.",
                "📤"
            )
        
        with col2:
            create_info_card(
                "AI Analysis", 
                "Our advanced machine learning models analyze weather patterns to predict pest populations.",
                "🤖"
            )
        
        with col3:
            create_info_card(
                "Get Insights", 
                "Receive actionable insights and recommendations for effective pest management strategies.",
                "💡"
            )
        
        st.markdown("---")
        
        # Feature highlights
        st.markdown("## 🌟 Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-highlight">
                <h4>🎯 Accurate Predictions</h4>
                <p>Advanced ML algorithms predict pest populations with high accuracy using weather data</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-highlight">
                <h4>📊 Comprehensive Analysis</h4>
                <p>Detailed statistical analysis including correlations, trends, and seasonal patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-highlight">
                <h4>📈 Interactive Visualizations</h4>
                <p>Dynamic charts and graphs for better understanding of pest-weather relationships</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-highlight">
                <h4>🚀 Real-time Processing</h4>
                <p>Fast data processing and instant predictions for timely pest management decisions</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Getting started section
        st.markdown("## 🚀 Getting Started")
        
        st.markdown("""
        <div class="info-card">
            <h4>📋 Step-by-Step Guide</h4>
            <ol>
                <li><strong>Upload Your Data:</strong> Go to the Data Upload section and upload your Excel/CSV file</li>
                <li><strong>Train Models:</strong> Click "Train Models" to build AI prediction models</li>
                <li><strong>Make Predictions:</strong> Use the AI Prediction tool to forecast pest populations</li>
                <li><strong>Analyze Results:</strong> Explore visualizations and model comparisons</li>
                <li><strong>Take Action:</strong> Implement recommended pest management strategies</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "📤 Data Upload":
        st.markdown("## 📤 Data Upload Center")
        
        create_info_card(
            "Supported Data Formats",
            "Upload Excel (.xlsx, .xls) or CSV files containing weather variables and trap catch data. Our system automatically detects and maps column names.",
            "📋"
        )
        
        # Expected columns info
        with st.expander("📊 Expected Data Structure", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📅 Time Variables:**")
                st.markdown("- Year/YEAR")
                st.markdown("- Std Week/Week/SMW")
                
                st.markdown("**🌡️ Temperature:**")
                st.markdown("- Tmax (Maximum Temperature)")
                st.markdown("- Tmin (Minimum Temperature)")
            
            with col2:
                st.markdown("**💧 Humidity & Weather:**")
                st.markdown("- RH max (Maximum Relative Humidity)")
                st.markdown("- RH min (Minimum Relative Humidity)")
                st.markdown("- RF (Rainfall)")
                st.markdown("- BSS (Bright Sunshine Hours)")
                st.markdown("- Wspeed/Wind speed")
                
                st.markdown("**🦟 Target Variable:**")
                st.markdown("- Trap Flies catch/Trap Catch")
        
        # File uploader with enhanced styling
        uploaded_file = st.file_uploader(
            "📁 Choose your dataset file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with your entomological data"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Processing your data..."):
                success, message = app.load_and_process_data(uploaded_file)
            
            if success:
                st.success(f"✅ {message}")
                
                # Enhanced data display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    create_metric_card("Dataset Rows", f"{app.df.shape[0]:,}", "Total observations")
                
                with col2:
                    create_metric_card("Total Columns", f"{app.df.shape[1]}", "Available features")
                
                with col3:
                    create_metric_card("Mapped Columns", f"{len(app.column_mapping)}", "Successfully mapped")
                
                # Column mapping display
                st.markdown("### 🗺️ Column Mapping")
                mapping_df = pd.DataFrame([
                    {"Standard Name": key.replace("_", " ").title(), "Your Column": value} 
                    for key, value in app.column_mapping.items()
                ])
                st.dataframe(mapping_df, use_container_width=True)
                
                # Data preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(app.df.head(10), use_container_width=True)
                
                # Enhanced train models button
                st.markdown("### 🤖 AI Model Training")
                
                if st.button("🚀 Train AI Models", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔄 Initializing training process...")
                    progress_bar.progress(25)
                    
                    with st.spinner("🧠 Training machine learning models..."):
                        success, message = app.train_models()
                        progress_bar.progress(75)
                        
                        if success:
                            progress_bar.progress(100)
                            status_text.text("✅ Training completed successfully!")
                            st.success(f"🎉 {message}")
                            st.balloons()
                            
                            # Show model summary
                            st.markdown("### 📊 Model Training Summary")
                            for name, info in app.models.items():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{name} - RMSE", f"{info['RMSE']:.3f}")
                                with col2:
                                    st.metric(f"{name} - MAE", f"{info['MAE']:.3f}")
                                with col3:
                                    st.metric(f"{name} - R²", f"{info['R²']:.3f}")
                        else:
                            st.error(f"❌ {message}")
            else:
                st.error(f"❌ {message}")
    
    elif page == "📊 Data Analysis":
        if app.df is None:
            st.warning("⚠️ Please upload data first!")
            return
        
        st.markdown("## 📊 Comprehensive Data Analysis")
        
        # Descriptive Statistics
        st.markdown("### 📈 Descriptive Statistics")
        
        # Enhanced stats display
        desc_stats = app.df.describe()
        st.dataframe(desc_stats.style.highlight_max(axis=0), use_container_width=True)
        
        # Correlation Analysis
        if 'trap_catch' in app.column_mapping:
            st.markdown("### 🔗 Correlation Analysis")
            
            weather_mappings = ['tmax', 'tmin', 'rh_max', 'rh_min', 'rainfall', 'sunshine', 'wind']
            available_cols = []
            
            for mapping in weather_mappings:
                if mapping in app.column_mapping:
                    available_cols.append(app.column_mapping[mapping])
            
            if available_cols:
                trap_col = app.column_mapping['trap_catch']
                
                correlations = []
                for col in available_cols:
                    try:
                        corr, p_value = pearsonr(app.df[col].dropna(), app.df[trap_col].dropna())
                        correlations.append({
                            'Weather Variable': col,
                            'Correlation': f"{corr:.3f}",
                            'P-value': f"{p_value:.4f}",
                            'Significance': '✅ Significant' if p_value < 0.05 else '❌ Not Significant',
                            'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                        })
                    except:
                        pass
                
                corr_df = pd.DataFrame(correlations)
                st.dataframe(corr_df, use_container_width=True)
        
        # Data Quality Assessment
        st.markdown("### 🔍 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_data = app.df.isnull().sum()
            if missing_data.sum() > 0:
                st.markdown("**Missing Values:**")
                st.bar_chart(missing_data[missing_data > 0])
            else:
                st.success("✅ No missing values found!")
        
        with col2:
            # Data types info
            st.markdown("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': app.df.dtypes.index,
                'Data Type': app.df.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)
    
    elif page == "🔮 AI Prediction":
        if not app.models:
            st.warning("⚠️ Please upload data and train models first!")
            return
        
        st.markdown("## 🔮 AI-Powered Pest Prediction")
        
        create_info_card(
            "Weather-Based Prediction",
            "Enter current weather conditions to get AI-powered predictions for pest populations. Our models analyze multiple weather factors to provide accurate forecasts.",
            "🤖"
        )
        
        # Enhanced prediction form
        with st.form("prediction_form", clear_on_submit=False):
            st.markdown("### 🌤️ Weather Input Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌡️ Temperature**")
                tmax = st.number_input("Maximum Temperature (°C)", value=30.0, step=0.1, min_value=-10.0, max_value=50.0)
                tmin = st.number_input("Minimum Temperature (°C)", value=20.0, step=0.1, min_value=-10.0, max_value=45.0)
                
                st.markdown("**💧 Humidity**")
                rh_max = st.number_input("Maximum Relative Humidity (%)", value=80.0, step=0.1, min_value=0.0, max_value=100.0)
                rh_min = st.number_input("Minimum Relative Humidity (%)", value=50.0, step=0.1, min_value=0.0, max_value=100.0)
            
            with col2:
                st.markdown("**🌧️ Weather Conditions**")
                rainfall = st.number_input("Rainfall (mm)", value=10.0, step=0.1, min_value=0.0, max_value=500.0)
                sunshine = st.number_input("Bright Sunshine Hours", value=8.0, step=0.1, min_value=0.0, max_value=24.0)
                wind_speed = st.number_input("Wind Speed", value=5.0, step=0.1, min_value=0.0, max_value=50.0)
            
            submitted = st.form_submit_button("🎯 Generate Prediction", type="primary", use_container_width=True)
            
            if submitted:
                # Prepare input data
                weather_mappings = ['tmax', 'tmin', 'rh_max', 'rh_min', 'rainfall', 'sunshine', 'wind']
                input_values = [tmax, tmin, rh_max, rh_min, rainfall, sunshine, wind_speed]
                
                available_cols = []
                available_values = []
                
                for i, mapping in enumerate(weather_mappings):
                    if mapping in app.column_mapping:
                        available_cols.append(app.column_mapping[mapping])
                        available_values.append(input_values[i])
                
                if available_cols:
                    input_df = pd.DataFrame([available_values], columns=available_cols)
                    
                    st.markdown("### 🎯 AI Predictions")
                    
                    predictions = {}
                    for name, model_info in app.models.items():
                        try:
                            model = model_info['model']
                            
                            if model_info['scaled']:
                                input_scaled = app.scaler.transform(input_df)
                                pred = model.predict(input_scaled)[0]
                            else:
                                pred = model.predict(input_df)[0]
                            
                            predictions[name] = max(0, pred)  # Ensure non-negative
                        except Exception as e:
                            st.error(f"Error with {name}: {str(e)}")
                    
                    # Enhanced predictions display
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (name, pred) in enumerate(predictions.items()):
                        with [col1, col2, col3][i % 3]:
                            accuracy = app.models[name]['R²']
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>{name}</h4>
                                <h2 style="color: #2E8B57;">{pred:.1f}</h2>
                                <p>Predicted Trap Catches</p>
                                <small>Model Accuracy: {accuracy:.1%}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Enhanced risk assessment
                    avg_pred = np.mean(list(predictions.values()))
                    
                    st.markdown("### 🚨 Risk Assessment")
                    
                    if avg_pred < 10:
                        st.markdown('<div class="risk-low">🟢 LOW RISK - Minimal pest pressure expected</div>', unsafe_allow_html=True)
                        recommendation = "✅ Current conditions show low pest pressure. Continue regular monitoring and maintain preventive measures."
                        icon = "🟢"
                    elif avg_pred < 50:
                        st.markdown('<div class="risk-medium">🟡 MEDIUM RISK - Moderate pest pressure expected</div>', unsafe_allow_html=True)
                        recommendation = "⚠️ Moderate pest pressure expected. Consider implementing preventive control measures and increase monitoring frequency."
                        icon = "🟡"
                    else:
                        st.markdown('<div class="risk-high">🔴 HIGH RISK - Significant pest pressure predicted</div>', unsafe_allow_html=True)
                        recommendation = "🚨 High pest pressure predicted. Implement immediate control measures and consider intensive management strategies."
                        icon = "🔴"
                    
                    # Management recommendations
                    st.markdown("### 💡 Management Recommendations")
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>{icon} Action Plan</h4>
                        <p>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "📈 Visualizations":
        if app.df is None:
            st.warning("⚠️ Please upload data first!")
            return
        
        st.markdown("## 📈 Interactive Data Visualizations")
        
        # Time Series Plot
        if 'week' in app.column_mapping and 'trap_catch' in app.column_mapping:
            st.markdown("### 📊 Trap Catches Over Time")
            
            week_col = app.column_mapping['week']
            trap_col = app.column_mapping['trap_catch']
            
            fig = px.line(
                app.df, 
                x=week_col, 
                y=trap_col,
                color='YEAR' if 'year' in app.column_mapping else None,
                title="🦟 Trap Flies Catch Trends Over Standard Weeks",
                labels={trap_col: "Trap Flies Catch", week_col: "Standard Week"},
                template="plotly_white"
            )
            fig.update_layout(
                title_font_size=20,
                title_x=0.5,
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Heatmap
        st.markdown("### 🔥 Weather Variables Correlation Matrix")
        
        weather_mappings = ['tmax', 'tmin', 'rh_max', 'rh_min', 'rainfall', 'sunshine', 'wind', 'trap_catch']
        available_cols = []
        
        for mapping in weather_mappings:
            if mapping in app.column_mapping:
                available_cols.append(app.column_mapping[mapping])
        
        if len(available_cols) > 1:
            corr_matrix = app.df[available_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="🌡️ Correlation Matrix: Weather Variables & Trap Catches",
                color_continuous_scale="RdBu_r",
                template="plotly_white"
            )
            fig.update_layout(
                title_font_size=20,
                title_x=0.5,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive Scatter Plots
        if 'trap_catch' in app.column_mapping:
            st.markdown("### 🎯 Weather vs Trap Catches Analysis")
            
            trap_col = app.column_mapping['trap_catch']
            
            # Select weather variable
            weather_options = []
            for mapping in ['tmax', 'tmin', 'rh_max', 'rh_min', 'rainfall', 'sunshine', 'wind']:
                if mapping in app.column_mapping:
                    weather_options.append(app.column_mapping[mapping])
            
            if weather_options:
                selected_weather = st.selectbox("🌤️ Select weather variable for analysis:", weather_options)
                
                fig = px.scatter(
                    app.df,
                    x=selected_weather,
                    y=trap_col,
                    color='YEAR' if 'year' in app.column_mapping else None,
                    title=f"📊 {selected_weather} vs Trap Flies Catch Relationship",
                    trendline="ols",
                    template="plotly_white"
                )
                fig.update_layout(
                    title_font_size=18,
                    title_x=0.5,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal Pattern
        if 'week' in app.column_mapping and 'trap_catch' in app.column_mapping:
            st.markdown("### 🗓️ Seasonal Pattern Analysis")
            
            week_col = app.column_mapping['week']
            trap_col = app.column_mapping['trap_catch']
            
            weekly_avg = app.df.groupby(week_col)[trap_col].mean().reset_index()
            
            fig = px.bar(
                weekly_avg,
                x=week_col,
                y=trap_col,
                title="📅 Average Trap Catches by Standard Week (Seasonal Trends)",
                labels={trap_col: "Average Trap Flies Catch", week_col: "Standard Week"},
                template="plotly_white",
                color=trap_col,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title_font_size=18,
                title_x=0.5,
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Model Comparison":
        if not app.models:
            st.warning("⚠️ Please upload data and train models first!")
            return
        
        st.markdown("## 🤖 AI Model Performance Analysis")
        
        # Model metrics comparison
        metrics_data = []
        for name, model_info in app.models.items():
            metrics_data.append({
                'Model': name,
                'RMSE': round(model_info['RMSE'], 3),
                'MAE': round(model_info['MAE'], 3),
                'R² Score': round(model_info['R²'], 3),
                'Accuracy': f"{model_info['R²']:.1%}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Enhanced metrics display
        st.markdown("### 📊 Performance Metrics Comparison")
        st.dataframe(metrics_df.style.highlight_max(subset=['R² Score']).highlight_min(subset=['RMSE', 'MAE']), use_container_width=True)
        
        # Metrics visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = px.bar(
                metrics_df, 
                x='Model', 
                y='RMSE',
                title="📉 Root Mean Square Error (Lower = Better)",
                template="plotly_white",
                color='RMSE',
                color_continuous_scale="Reds_r"
            )
            fig_rmse.update_layout(title_x=0.5, height=400)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_r2 = px.bar(
                metrics_df, 
                x='Model', 
                y='R² Score',
                title="📈 R² Score (Higher = Better)",
                template="plotly_white",
                color='R² Score',
                color_continuous_scale="Greens"
            )
            fig_r2.update_layout(title_x=0.5, height=400)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Feature Importance Analysis
        st.markdown("### 🎯 Feature Importance Analysis")
        
        for name, model_info in app.models.items():
            if 'feature_importance' in model_info:
                st.markdown(f"#### 🔍 {name} - Key Weather Factors")
                
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Weather Variable', 'Importance Score']
                ).sort_values('Importance Score', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance Score',
                    y='Weather Variable',
                    orientation='h',
                    title=f"🌟 Feature Importance: {name}",
                    template="plotly_white",
                    color='Importance Score',
                    color_continuous_scale="Blues"
                )
                fig.update_layout(title_x=0.5, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Best model recommendation
        best_model = max(app.models.items(), key=lambda x: x[1]['R²'])
        
        st.markdown("### 🏆 Model Recommendation")
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white;">
            <h3>🥇 Best Performing Model</h3>
            <h2>{best_model[0]}</h2>
            <p><strong>R² Score:</strong> {best_model[1]['R²']:.3f} ({best_model[1]['R²']:.1%} accuracy)</p>
            <p><strong>RMSE:</strong> {best_model[1]['RMSE']:.3f}</p>
            <p>This model provides the most accurate predictions for your dataset.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ Desinged By Makhan Kumbhkar ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_info_card(
                "Mission",
                "Empowering agricultural professionals with AI-driven insights for effective pest management through advanced weather-based prediction models.",
                "🎯"
            )
            
            create_info_card(
                "Technology Stack",
                "Built with Python, Streamlit, Scikit-learn, Plotly, and advanced machine learning algorithms for robust and accurate predictions.",
                "⚙️"
            )
        
        with col2:
            create_info_card(
                "Research Impact",
                "Supporting sustainable agriculture by providing data-driven solutions for pest population forecasting and management optimization.",
                "🌱"
            )
            
            create_info_card(
                "Contact & Support",
                "For technical support, feature requests, or research collaborations, please refer to the documentation and support channels.",
                "📞"
            )
        
        st.markdown("---")
        
        st.markdown("### 🔬 Scientific Approach")
        
        st.markdown("""
        Our platform employs multiple machine learning algorithms:
        
        - **Linear Regression**: Baseline statistical model for linear relationships
        - **Random Forest**: Ensemble method handling non-linear patterns and feature interactions
        - **Gradient Boosting**: Advanced ensemble technique for complex pattern recognition
        
        Each model is evaluated using standard metrics (RMSE, MAE, R²) to ensure reliability and accuracy.
        """)
        
        st.markdown("### 📈 Model Performance")
        
        if app.models:
            performance_summary = pd.DataFrame([
                {
                    'Metric': 'Average R² Score',
                    'Value': f"{np.mean([m['R²'] for m in app.models.values()]):.3f}"
                },
                {
                    'Metric': 'Best Model Accuracy',
                    'Value': f"{max(app.models.values(), key=lambda x: x['R²'])['R²']:.1%}"
                },
                {
                    'Metric': 'Models Trained',
                    'Value': f"{len(app.models)}"
                }
            ])
            st.dataframe(performance_summary, use_container_width=True)

if __name__ == "__main__":
    main()




