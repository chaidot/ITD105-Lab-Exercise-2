import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, RepeatedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, log_loss, confusion_matrix, 
                             classification_report, roc_auc_score, roc_curve,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Big Data Analytics Lab", layout="wide", page_icon="📊")

# Custom CSS for enhanced design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    /* Content area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Headers with gradient text */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
        text-align: center;
        letter-spacing: -1px;
    }
    
    h2 {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #e0e7ff !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    h3 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #c7d2fe !important;
        margin-top: 1.5rem !important;
    }
    
    /* Subtitle */
    .main h3:first-of-type {
        text-align: center;
        color: #a5b4fc !important;
        font-weight: 400 !important;
        margin-top: 0 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 8px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        padding: 1rem 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        color: #a5b4fc !important;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    /* Cards & Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 15px !important;
        border: 2px solid rgba(102, 126, 234, 0.3);
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #e0e7ff !important;
        padding: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        transform: translateX(5px);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #c7d2fe !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div {
        font-size: 1.1rem !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #e0e7ff !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Labels */
    label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #c7d2fe !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-size: 1.2rem !important;
        color: #c7d2fe !important;
    }
    
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stSlider > label {
        font-size: 1.2rem !important;
        color: #c7d2fe !important;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        font-size: 1.1rem !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stNotificationContentInfo"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%) !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    [data-testid="stNotificationContentWarning"] {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%) !important;
        border-left: 4px solid #fbbf24 !important;
    }
    
    [data-testid="stNotificationContentSuccess"] {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%) !important;
        border-left: 4px solid #22c55e !important;
    }
    
    /* Dataframes */
    .dataframe {
        font-size: 1rem !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .dataframe tbody tr {
        background: rgba(255, 255, 255, 0.03) !important;
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Captions */
    .stCaption {
        font-size: 1rem !important;
        color: #a5b4fc !important;
        font-style: italic;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Multiselect tags */
    .stMultiSelect span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #c7d2fe !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 6px !important;
        font-size: 1rem !important;
    }
    
    /* Plotly charts - dark theme */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animation for elements */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Special styling for section headers */
    [data-testid="stMarkdownContainer"] h2 {
        padding: 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing models
if 'classification_models_trained' not in st.session_state:
    st.session_state.classification_models_trained = False
if 'regression_models_trained' not in st.session_state:
    st.session_state.regression_models_trained = False

# Title
st.title("🔬 ITD105 - Big Data Analytics Lab Exercise #2")
st.markdown("### Resampling Techniques and Performance Metrics")

# Create tabs for different parts
tab1, tab2, tab3 = st.tabs(["📋 Classification Task", "🌍 Regression Task", "📚 Documentation"])

# =============================================================================
# PART 1: CLASSIFICATION TASK
# =============================================================================
with tab1:
    st.header("Part 1: Health Data Classification")
    st.markdown("**Using Logistic Regression with Different Resampling Techniques**")
    
    # File uploader for classification data
    uploaded_file_class = st.file_uploader("Upload your health dataset (CSV)", type=['csv'], key='class')
    
    if uploaded_file_class:
        # Load data
        df_class = pd.read_csv(uploaded_file_class)
        
        # Display data info
        with st.expander("📊 Dataset Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", df_class.shape[0])
            col2.metric("Total Features", df_class.shape[1])
            col3.metric("Memory Usage", f"{df_class.memory_usage().sum() / 1024:.2f} KB")
            
            st.dataframe(df_class.head(10), use_container_width=True)
            
            # Show data types
            st.write("**Column Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df_class.columns,
                'Data Type': df_class.dtypes.values,
                'Non-Null Count': df_class.count().values,
                'Unique Values': [df_class[col].nunique() for col in df_class.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            st.write("**Dataset Statistics:**")
            st.write(df_class.describe())
        
        # Feature selection
        st.subheader("⚙️ Configure Model")
        
        st.info("💡 **Note**: Categorical features (like 'Male'/'Female') will be automatically converted to numbers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("Select Target Column", df_class.columns, key='target_class')
        with col2:
            feature_cols = st.multiselect("Select Feature Columns", 
                                         [col for col in df_class.columns if col != target_col],
                                         default=[col for col in df_class.columns if col != target_col][:5])
        
        if feature_cols and target_col:
            # Prepare features with proper encoding
            X_df = df_class[feature_cols].copy()
            
            # Convert categorical columns to numeric
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            
            for col in X_df.columns:
                if X_df[col].dtype == 'object':
                    le = LabelEncoder()
                    X_df[col] = le.fit_transform(X_df[col].astype(str))
                    label_encoders[col] = le
            
            X = X_df.values
            
            # Prepare target variable
            y_series = df_class[target_col].copy()
            if y_series.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y_series.astype(str))
            else:
                y = y_series.values
            
            # Handle missing values
            X = np.nan_to_num(X)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Number of folds for K-Fold
            n_folds = st.slider("Select number of folds for K-Fold CV", 3, 10, 5)
            
            if st.button("🚀 Train Models", key='train_class') or st.session_state.classification_models_trained:
                if st.session_state.classification_models_trained:
                    if st.button("🔄 Reset & Retrain", key='reset_class'):
                        st.session_state.classification_models_trained = False
                        st.rerun()
                
                if not st.session_state.classification_models_trained:
                    with st.spinner("Training models..."):
                        st.session_state.classification_models_trained = True
                else:
                    st.info("✅ Models already trained. Use prediction section below or click 'Reset & Retrain' to train again.")
                
                if st.session_state.classification_models_trained:
                    
                    # Model A: K-Fold Cross-Validation
                    st.subheader("📈 Model A: K-Fold Cross-Validation")
                    
                    # Store in session state if not already stored
                    if 'model_a_final' not in st.session_state:
                        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                        
                        acc_scores_kfold = []
                        log_loss_scores_kfold = []
                        auc_scores_kfold = []
                        
                        # Get all unique classes
                        all_classes = np.unique(y)
                        
                        for train_idx, test_idx in kfold.split(X_scaled):
                            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            model_a = LogisticRegression(max_iter=1000, random_state=42)
                            model_a.fit(X_train, y_train)
                            
                            y_pred = model_a.predict(X_test)
                            y_pred_proba = model_a.predict_proba(X_test)
                            
                            acc_scores_kfold.append(accuracy_score(y_test, y_pred))
                            
                            # Fix log_loss by specifying labels
                            try:
                                log_loss_scores_kfold.append(log_loss(y_test, y_pred_proba, labels=model_a.classes_))
                            except:
                                pass
                            
                            if len(all_classes) == 2:
                                try:
                                    auc_scores_kfold.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
                                except:
                                    pass
                        
                        # Train final Model A on full data for download
                        model_a_final = LogisticRegression(max_iter=1000, random_state=42)
                        model_a_final.fit(X_scaled, y)
                        
                        # Store in session state
                        st.session_state.model_a_final = model_a_final
                        st.session_state.acc_scores_kfold = acc_scores_kfold
                        st.session_state.log_loss_scores_kfold = log_loss_scores_kfold
                        st.session_state.auc_scores_kfold = auc_scores_kfold
                        st.session_state.y_test_a = y_test
                        st.session_state.y_pred_a = y_pred
                        st.session_state.scaler = scaler
                        st.session_state.feature_cols = feature_cols
                        if 'label_encoders' in locals():
                            st.session_state.label_encoders = label_encoders
                    
                    # Retrieve from session state
                    model_a_final = st.session_state.model_a_final
                    acc_scores_kfold = st.session_state.acc_scores_kfold
                    log_loss_scores_kfold = st.session_state.log_loss_scores_kfold
                    auc_scores_kfold = st.session_state.auc_scores_kfold
                    y_test = st.session_state.y_test_a
                    y_pred = st.session_state.y_pred_a
                    scaler = st.session_state.scaler
                    feature_cols = st.session_state.feature_cols
                    if 'label_encoders' in st.session_state:
                        label_encoders = st.session_state.label_encoders
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Accuracy", f"{np.mean(acc_scores_kfold):.4f}")
                    if log_loss_scores_kfold:
                        col2.metric("Avg Log Loss", f"{np.mean(log_loss_scores_kfold):.4f}")
                    if auc_scores_kfold:
                        col3.metric("Avg AUC-ROC", f"{np.mean(auc_scores_kfold):.4f}")
                    # Display Model A results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Accuracy", f"{np.mean(acc_scores_kfold):.4f}")
                    if log_loss_scores_kfold:
                        col2.metric("Avg Log Loss", f"{np.mean(log_loss_scores_kfold):.4f}")
                    if auc_scores_kfold:
                        col3.metric("Avg AUC-ROC", f"{np.mean(auc_scores_kfold):.4f}")
                    
                    # Confusion Matrix for last fold
                    cm_a = confusion_matrix(y_test, y_pred)
                    fig_cm_a = px.imshow(cm_a, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                        title="Confusion Matrix - Model A (Last Fold)")
                    st.plotly_chart(fig_cm_a, use_container_width=True)
                    
                    # Classification Report
                    st.text("Classification Report - Model A:")
                    st.text(classification_report(y_test, y_pred))
                    
                    st.markdown("---")
                    
                    # Model B: Leave-One-Out Cross-Validation
                    st.subheader("📉 Model B: Leave-One-Out Cross-Validation")
                    
                    
                    if 'model_b_final' not in st.session_state:
                        # LOOCV can be slow, so limit samples if dataset is large
                        if len(X_scaled) > 500:
                            st.warning(f"⚠️ Dataset has {len(X_scaled)} samples. LOOCV may be slow. Using subset of 500 samples.")
                            sample_idx = np.random.choice(len(X_scaled), 500, replace=False)
                            X_loocv = X_scaled[sample_idx]
                            y_loocv = y[sample_idx]
                        else:
                            X_loocv = X_scaled
                            y_loocv = y
                        
                        loo = LeaveOneOut()
                        acc_scores_loo = []
                        y_true_loo = []
                        y_pred_loo = []
                        
                        for train_idx, test_idx in loo.split(X_loocv):
                            X_train, X_test = X_loocv[train_idx], X_loocv[test_idx]
                            y_train, y_test = y_loocv[train_idx], y_loocv[test_idx]
                            
                            model_b = LogisticRegression(max_iter=1000, random_state=42)
                            model_b.fit(X_train, y_train)
                            
                            y_pred = model_b.predict(X_test)
                            y_true_loo.extend(y_test)
                            y_pred_loo.extend(y_pred)
                            acc_scores_loo.append(accuracy_score(y_test, y_pred))
                        
                        # Train final Model B on full dataset
                        model_b_final = LogisticRegression(max_iter=1000, random_state=42)
                        model_b_final.fit(X_scaled, y)
                        
                        # Get predictions for LOOCV subset
                        y_pred_proba_b = model_b_final.predict_proba(X_loocv)
                        
                        # Calculate metrics
                        try:
                            classes_in_test = np.unique(y_loocv)
                            classes_in_model = model_b_final.classes_
                            common_classes = np.intersect1d(classes_in_test, classes_in_model)
                            
                            if len(common_classes) == len(classes_in_model):
                                log_loss_b = log_loss(y_loocv, y_pred_proba_b, labels=classes_in_model)
                            else:
                                log_loss_b = None
                        except:
                            log_loss_b = None
                        
                        if len(np.unique(y)) == 2:
                            try:
                                auc_b = roc_auc_score(y_loocv, y_pred_proba_b[:, 1])
                            except:
                                auc_b = None
                        else:
                            try:
                                auc_b = roc_auc_score(y_loocv, y_pred_proba_b, multi_class='ovr', labels=model_b_final.classes_)
                            except:
                                auc_b = None
                        
                        # Store in session state
                        st.session_state.model_b_final = model_b_final
                        st.session_state.acc_scores_loo = acc_scores_loo
                        st.session_state.y_true_loo = y_true_loo
                        st.session_state.y_pred_loo = y_pred_loo
                        st.session_state.log_loss_b = log_loss_b
                        st.session_state.auc_b = auc_b
                    
                    # Retrieve from session state
                    model_b_final = st.session_state.model_b_final
                    acc_scores_loo = st.session_state.acc_scores_loo
                    y_true_loo = st.session_state.y_true_loo
                    y_pred_loo = st.session_state.y_pred_loo
                    log_loss_b = st.session_state.log_loss_b
                    auc_b = st.session_state.auc_b
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{np.mean(acc_scores_loo):.4f}")
                    
                    if log_loss_b is not None:
                        col2.metric("Log Loss", f"{log_loss_b:.4f}")
                    else:
                        col2.metric("Log Loss", "N/A")
                    
                    if auc_b is not None:
                        col3.metric("AUC-ROC", f"{auc_b:.4f}")
                    else:
                        col3.metric("AUC-ROC", "N/A")
                    
                    # Confusion Matrix
                    cm_b = confusion_matrix(y_true_loo, y_pred_loo)
                    fig_cm_b = px.imshow(cm_b, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                        title="Confusion Matrix - Model B")
                    st.plotly_chart(fig_cm_b, use_container_width=True)
                    
                    st.text("Classification Report - Model B:")
                    st.text(classification_report(y_true_loo, y_pred_loo))
                    
                    st.markdown("---")
                    
                    # Model Comparison
                    st.subheader("🔍 Model Comparison & Selection")
                    
                    comparison_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Log Loss', 'AUC-ROC'],
                        'Model A (K-Fold)': [
                            f"{np.mean(acc_scores_kfold):.4f}",
                            f"{np.mean(log_loss_scores_kfold):.4f}" if log_loss_scores_kfold else 'N/A',
                            f"{np.mean(auc_scores_kfold):.4f}" if auc_scores_kfold else 'N/A'
                        ],
                        'Model B (LOOCV)': [
                            f"{np.mean(acc_scores_loo):.4f}",
                            f"{log_loss_b:.4f}" if log_loss_b is not None else 'N/A',
                            f"{auc_b:.4f}" if auc_b is not None else 'N/A'
                        ]
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    st.info("""
                    **📊 Performance Metrics Interpretation:**
                    
                    - **Classification Accuracy**: Percentage of correct predictions. Higher is better.
                    - **Logarithmic Loss**: Measures the uncertainty of predictions. Lower is better.
                    - **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives.
                    - **Classification Report**: Provides precision, recall, and F1-score for each class.
                    - **AUC-ROC**: Area under the ROC curve. Measures model's ability to distinguish classes. Closer to 1 is better.
                    
                    **🎯 Model Selection Recommendation:**
                    - **Model A (K-Fold)** is recommended for most cases: faster, more practical, less prone to overfitting
                    - **Model B (LOOCV)** provides exhaustive validation but is computationally expensive
                    """)
                    
                    # Model download
                    st.subheader("💾 Download Model")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Save Model A
                        model_a_data = pickle.dumps({
                            'model': model_a_final, 
                            'scaler': scaler, 
                            'features': feature_cols,
                            'label_encoders': label_encoders if 'label_encoders' in locals() else None
                        })
                        st.download_button(
                            label="📥 Download Model A (K-Fold)",
                            data=model_a_data,
                            file_name="model_a_kfold.pkl",
                            mime="application/octet-stream"
                        )
                    
                    with col2:
                        # Save Model B
                        model_b_data = pickle.dumps({
                            'model': model_b_final, 
                            'scaler': scaler, 
                            'features': feature_cols,
                            'label_encoders': label_encoders if 'label_encoders' in locals() else None
                        })
                        st.download_button(
                            label="📥 Download Model B (LOOCV)",
                            data=model_b_data,
                            file_name="model_b_loocv.pkl",
                            mime="application/octet-stream"
                        )
                    
                    # Prediction interface
                    st.subheader("🔮 Make Predictions")
                    
                    selected_model = st.radio("Select Model for Prediction", ["Model A (K-Fold)", "Model B (LOOCV)"])
                    
                    pred_model = model_a_final if "Model A" in selected_model else model_b_final
                    
                    st.write("**Enter feature values:**")
                    st.caption("For categorical features, enter the original text values (e.g., 'Male', 'Female')")
                    
                    input_features = []
                    input_features_display = {}
                    cols = st.columns(3)
                    
                    for i, feature in enumerate(feature_cols):
                        with cols[i % 3]:
                            # Check if feature was encoded
                            if 'label_encoders' in locals() and feature in label_encoders:
                                # Get original categories
                                categories = label_encoders[feature].classes_
                                val = st.selectbox(f"{feature}", categories, key=f"pred_{feature}")
                                # Encode the value
                                encoded_val = label_encoders[feature].transform([val])[0]
                                input_features.append(encoded_val)
                                input_features_display[feature] = val
                            else:
                                # Numeric feature
                                val = st.number_input(f"{feature}", value=0.0, key=f"pred_{feature}")
                                input_features.append(val)
                                input_features_display[feature] = val
                    
                    if st.button("🎯 Predict", key='predict_class'):
                        input_scaled = scaler.transform([input_features])
                        prediction = pred_model.predict(input_scaled)[0]
                        prediction_proba = pred_model.predict_proba(input_scaled)[0]
                        
                        # Show input summary
                        st.write("**Input Summary:**")
                        input_summary_cols = st.columns(2)
                        items = list(input_features_display.items())
                        mid = len(items) // 2
                        
                        with input_summary_cols[0]:
                            for feature, val in items[:mid]:
                                st.write(f"- **{feature}**: {val}")
                        with input_summary_cols[1]:
                            for feature, val in items[mid:]:
                                st.write(f"- **{feature}**: {val}")
                        
                        st.markdown("---")
                        
                        # Show model being used
                        st.info(f"**Using:** {selected_model}")
                        
                        # Prediction result
                        result_col1, result_col2 = st.columns([1, 2])
                        
                        with result_col1:
                            st.metric(
                                label="Predicted Class",
                                value=f"Class {prediction}",
                                delta="Diabetes" if prediction == 1 else "No Diabetes"
                            )
                            st.metric(
                                label="Confidence",
                                value=f"{max(prediction_proba)*100:.2f}%"
                            )
                        
                        with result_col2:
                            st.write("**Prediction Probabilities:**")
                            
                            # Create a nice visualization
                            prob_df = pd.DataFrame({
                                'Class': [f"Class {i}" for i in range(len(prediction_proba))],
                                'Probability': prediction_proba
                            })
                            
                            fig_prob = px.bar(prob_df, x='Class', y='Probability', 
                                             title='Prediction Probability Distribution',
                                             color='Probability',
                                             color_continuous_scale='blues',
                                             text=[f"{p:.1%}" for p in prediction_proba])
                            fig_prob.update_traces(textposition='outside')
                            fig_prob.update_layout(
                                showlegend=False,
                                height=300,
                                yaxis_range=[0, 1]
                            )
                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Detailed probabilities
                        st.write("**Detailed Probabilities:**")
                        prob_cols = st.columns(len(prediction_proba))
                        for i, prob in enumerate(prediction_proba):
                            with prob_cols[i]:
                                st.info(f"**Class {i}**\n\n{prob:.4f}\n\n({prob*100:.2f}%)")
                        
                        # Interpretation
                        st.markdown("---")
                        st.write("**Result Interpretation:**")
                        if prediction == 1:
                            st.warning(f"⚠️ The model predicts **Diabetes** (Class 1) with {prediction_proba[1]*100:.2f}% probability. This patient profile shows characteristics associated with diabetes risk.")
                        else:
                            st.success(f"✅ The model predicts **No Diabetes** (Class 0) with {prediction_proba[0]*100:.2f}% probability. This patient profile shows lower diabetes risk indicators.")

# =============================================================================
# PART 2: REGRESSION TASK
# =============================================================================
with tab2:
    st.header("Part 2: Environmental Data Regression")
    st.markdown("**Using Linear Regression with Different Resampling Techniques**")
    
    uploaded_file_reg = st.file_uploader("Upload your environment dataset (CSV)", type=['csv'], key='reg')
    
    if uploaded_file_reg:
        df_reg = pd.read_csv(uploaded_file_reg)
        
        with st.expander("📊 Dataset Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", df_reg.shape[0])
            col2.metric("Total Features", df_reg.shape[1])
            col3.metric("Memory Usage", f"{df_reg.memory_usage().sum() / 1024:.2f} KB")
            
            st.dataframe(df_reg.head(10), use_container_width=True)
            st.write("**Dataset Info:**")
            st.write(df_reg.describe())
            
            # Show data types
            st.write("**Column Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df_reg.columns,
                'Data Type': df_reg.dtypes.values,
                'Non-Null Count': df_reg.count().values,
                'Unique Values': [df_reg[col].nunique() for col in df_reg.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        st.subheader("⚙️ Configure Model")
        
        st.info("💡 **Tip**: For regression tasks, select a numeric target column (continuous values like temperature, price, etc.)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter to show only numeric columns for target
            numeric_cols = df_reg.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("⚠️ No numeric columns found. Showing all columns, but conversion may be needed.")
                numeric_cols = df_reg.columns.tolist()
            
            target_col_reg = st.selectbox("Select Target Column (must be numeric)", numeric_cols, key='target_reg')
            
        with col2:
            feature_cols_reg = st.multiselect("Select Feature Columns", 
                                             [col for col in df_reg.columns if col != target_col_reg],
                                             default=[col for col in df_reg.columns if col != target_col_reg][:5])
        
        if feature_cols_reg and target_col_reg:
            # Prepare features
            X_reg_df = df_reg[feature_cols_reg].copy()
            
            # Convert non-numeric columns to numeric
            for col in X_reg_df.columns:
                if X_reg_df[col].dtype == 'object':
                    try:
                        # Try converting dates to timestamps
                        X_reg_df[col] = pd.to_datetime(X_reg_df[col]).astype('int64') / 10**9
                    except:
                        # If not dates, use label encoding
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        X_reg_df[col] = le.fit_transform(X_reg_df[col].astype(str))
            
            X_reg = X_reg_df.values
            X_reg = np.nan_to_num(X_reg)
            
            # Prepare target variable
            y_reg_series = df_reg[target_col_reg].copy()
            
            # Check if target is non-numeric
            if y_reg_series.dtype == 'object':
                st.warning("⚠️ Target column contains non-numeric data. Attempting conversion...")
                try:
                    # Try converting to datetime first
                    y_reg_series = pd.to_datetime(y_reg_series).astype('int64') / 10**9
                    st.info("✅ Target converted from date to numeric timestamp")
                except:
                    try:
                        # Try direct conversion to float
                        y_reg_series = pd.to_numeric(y_reg_series, errors='coerce')
                        st.info("✅ Target converted to numeric")
                    except:
                        st.error("❌ Cannot convert target to numeric. Please select a numeric column.")
                        st.stop()
            
            y_reg = y_reg_series.values
            y_reg = np.nan_to_num(y_reg)
            
            # Check if we have valid numeric data
            if not np.isfinite(y_reg).all():
                st.error("❌ Target column contains invalid values after conversion. Please select a different column.")
                st.stop()
            
            scaler_reg = StandardScaler()
            X_reg_scaled = scaler_reg.fit_transform(X_reg)
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            with col2:
                n_repeats = st.slider("Number of Repeated Splits", 3, 20, 10)
            
            if st.button("🚀 Train Models", key='train_reg') or st.session_state.regression_models_trained:
                if st.button("🔄 Reset & Retrain", key='reset_reg'):
                    st.session_state.regression_models_trained = False
                    st.rerun()
                
                if not st.session_state.regression_models_trained:
                    with st.spinner("Training models..."):
                        st.session_state.regression_models_trained = True
                else:
                    st.info("✅ Models already trained. Use prediction section below or click 'Reset & Retrain' to train again.")
                
                if st.session_state.regression_models_trained:
                    
                    # Model A: Train-Test Split
                    st.subheader("📈 Model A: Train-Test Split")
                    
                    if 'model_reg_a' not in st.session_state:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_reg_scaled, y_reg, test_size=test_size, random_state=42
                        )
                        
                        model_reg_a = LinearRegression()
                        model_reg_a.fit(X_train, y_train)
                        y_pred_a = model_reg_a.predict(X_test)
                        
                        mse_a = mean_squared_error(y_test, y_pred_a)
                        mae_a = mean_absolute_error(y_test, y_pred_a)
                        r2_a = r2_score(y_test, y_pred_a)
                        
                        residuals_a = y_test - y_pred_a
                        
                        # Store in session state
                        st.session_state.model_reg_a = model_reg_a
                        st.session_state.y_test_reg_a = y_test
                        st.session_state.y_pred_a = y_pred_a
                        st.session_state.mse_a = mse_a
                        st.session_state.mae_a = mae_a
                        st.session_state.r2_a = r2_a
                        st.session_state.residuals_a = residuals_a
                        st.session_state.scaler_reg = scaler_reg
                        st.session_state.feature_cols_reg = feature_cols_reg
                    
                    # Retrieve from session state
                    model_reg_a = st.session_state.model_reg_a
                    y_test = st.session_state.y_test_reg_a
                    y_pred_a = st.session_state.y_pred_a
                    mse_a = st.session_state.mse_a
                    mae_a = st.session_state.mae_a
                    r2_a = st.session_state.r2_a
                    residuals_a = st.session_state.residuals_a
                    scaler_reg = st.session_state.scaler_reg
                    feature_cols_reg = st.session_state.feature_cols_reg
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{mse_a:.4f}")
                    col2.metric("MAE", f"{mae_a:.4f}")
                    col3.metric("R² Score", f"{r2_a:.4f}")
                    
                    # Prediction vs Actual plot
                    fig_a = go.Figure()
                    fig_a.add_trace(go.Scatter(x=y_test, y=y_pred_a, mode='markers', name='Predictions'))
                    fig_a.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
                    fig_a.update_layout(title="Model A: Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted")
                    st.plotly_chart(fig_a, use_container_width=True)
                    
                    # Residual plot
                    fig_res_a = px.scatter(x=y_pred_a, y=residuals_a, labels={'x': 'Predicted', 'y': 'Residuals'}, 
                                          title="Model A: Residual Plot")
                    fig_res_a.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res_a, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Model B: Repeated Random Train-Test Splits
                    st.subheader("📉 Model B: Repeated Random Train-Test Splits")
                    
                    if 'model_reg_b_final' not in st.session_state:
                        mse_scores_b = []
                        mae_scores_b = []
                        r2_scores_b = []
                        
                        for i in range(n_repeats):
                            X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
                                X_reg_scaled, y_reg, test_size=test_size, random_state=i
                            )
                            
                            model_reg_b = LinearRegression()
                            model_reg_b.fit(X_train_b, y_train_b)
                            y_pred_b = model_reg_b.predict(X_test_b)
                            
                            mse_scores_b.append(mean_squared_error(y_test_b, y_pred_b))
                            mae_scores_b.append(mean_absolute_error(y_test_b, y_pred_b))
                            r2_scores_b.append(r2_score(y_test_b, y_pred_b))
                        
                        # Train final model
                        model_reg_b_final = LinearRegression()
                        model_reg_b_final.fit(X_reg_scaled, y_reg)
                        
                        # Store in session state
                        st.session_state.model_reg_b_final = model_reg_b_final
                        st.session_state.mse_scores_b = mse_scores_b
                        st.session_state.mae_scores_b = mae_scores_b
                        st.session_state.r2_scores_b = r2_scores_b
                    
                    # Retrieve from session state
                    model_reg_b_final = st.session_state.model_reg_b_final
                    mse_scores_b = st.session_state.mse_scores_b
                    mae_scores_b = st.session_state.mae_scores_b
                    r2_scores_b = st.session_state.r2_scores_b
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg MSE", f"{np.mean(mse_scores_b):.4f}", delta=f"±{np.std(mse_scores_b):.4f}")
                    col2.metric("Avg MAE", f"{np.mean(mae_scores_b):.4f}", delta=f"±{np.std(mae_scores_b):.4f}")
                    col3.metric("Avg R² Score", f"{np.mean(r2_scores_b):.4f}", delta=f"±{np.std(r2_scores_b):.4f}")
                    
                    # Score distribution
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Box(y=mse_scores_b, name='MSE'))
                    fig_dist.add_trace(go.Box(y=mae_scores_b, name='MAE'))
                    fig_dist.add_trace(go.Box(y=r2_scores_b, name='R²'))
                    fig_dist.update_layout(title="Model B: Score Distribution Across Splits", yaxis_title="Score Value")
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Model Comparison
                    st.subheader("🔍 Model Comparison & Selection")
                    
                    comparison_df_reg = pd.DataFrame({
                        'Metric': ['MSE', 'MAE', 'R² Score'],
                        'Model A (Single Split)': [
                            f"{mse_a:.4f}",
                            f"{mae_a:.4f}",
                            f"{r2_a:.4f}"
                        ],
                        'Model B (Repeated Splits)': [
                            f"{np.mean(mse_scores_b):.4f} ±{np.std(mse_scores_b):.4f}",
                            f"{np.mean(mae_scores_b):.4f} ±{np.std(mae_scores_b):.4f}",
                            f"{np.mean(r2_scores_b):.4f} ±{np.std(r2_scores_b):.4f}"
                        ]
                    })
                    
                    st.dataframe(comparison_df_reg, use_container_width=True)
                    
                    st.info("""
                    **📊 Performance Metrics Interpretation:**
                    
                    - **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values. Lower is better. Penalizes large errors more.
                    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values. Lower is better. More interpretable than MSE.
                    - **R² Score**: Proportion of variance explained by the model. Range: 0 to 1 (can be negative for very poor models). Closer to 1 is better.
                    
                    **🎯 Model Selection Recommendation:**
                    - **Model B (Repeated Splits)** is recommended: provides more robust performance estimates with confidence intervals
                    - **Model A** is faster but single split may not be representative
                    """)
                    
                    # Model download
                    st.subheader("💾 Download Model")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        model_reg_a_data = pickle.dumps({'model': model_reg_a, 'scaler': scaler_reg, 'features': feature_cols_reg})
                        st.download_button(
                            label="📥 Download Model A (Single Split)",
                            data=model_reg_a_data,
                            file_name="model_reg_a.pkl",
                            mime="application/octet-stream"
                        )
                    
                    with col2:
                        model_reg_b_data = pickle.dumps({'model': model_reg_b_final, 'scaler': scaler_reg, 'features': feature_cols_reg})
                        st.download_button(
                            label="📥 Download Model B (Repeated Splits)",
                            data=model_reg_b_data,
                            file_name="model_reg_b.pkl",
                            mime="application/octet-stream"
                        )
                    
                    # Prediction interface
                    st.subheader("🔮 Make Predictions")
                    
                    selected_model_reg = st.radio("Select Model for Prediction", 
                                                 ["Model A (Single Split)", "Model B (Repeated Splits)"], 
                                                 key='select_reg')
                    
                    pred_model_reg = model_reg_a if "Model A" in selected_model_reg else model_reg_b_final
                    
                    st.write("**Enter feature values:**")
                    input_features_reg = []
                    cols = st.columns(3)
                    for i, feature in enumerate(feature_cols_reg):
                        with cols[i % 3]:
                            val = st.number_input(f"{feature}", value=0.0, key=f"pred_reg_{feature}")
                            input_features_reg.append(val)
                    
                    if st.button("🎯 Predict", key='predict_reg'):
                        input_scaled_reg = scaler_reg.transform([input_features_reg])
                        prediction_reg = pred_model_reg.predict(input_scaled_reg)[0]
                        
                        st.success(f"**Predicted Value:** {prediction_reg:.4f}")
                        
                        # Visualization
                        fig_pred = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction_reg,
                            title={'text': f"Predicted {target_col_reg}"},
                            gauge={'axis': {'range': [None, prediction_reg * 2]}}
                        ))
                        st.plotly_chart(fig_pred, use_container_width=True)

# =============================================================================
# DOCUMENTATION TAB
# =============================================================================
with tab3:
    st.header("📚 Lab Exercise Documentation")
    
    st.markdown("""
    ## 🎯 Objectives
    This lab exercise demonstrates:
    1. Different resampling techniques for model evaluation
    2. Various performance metrics for classification and regression
    3. Model comparison and selection
    4. Interactive prediction using Streamlit
    
    ## 📋 Part 1: Classification Task
    
    ### Model A - K-Fold Cross-Validation
    - Splits data into K equal folds
    - Each fold serves as test set once
    - More efficient than LOOCV
    - Provides good bias-variance tradeoff
    
    ### Model B - Leave-One-Out Cross-Validation (LOOCV)
    - Each sample is test set once
    - N iterations for N samples
    - Maximum use of training data
    - Computationally expensive
    - Low bias but high variance
    
    ### Performance Metrics
    - **Accuracy**: Overall correctness
    - **Log Loss**: Confidence of predictions
    - **Confusion Matrix**: Breakdown of predictions
    - **Classification Report**: Precision, recall, F1-score
    - **AUC-ROC**: Discrimination ability
    
    ## 🌍 Part 2: Regression Task
    
    ### Model A - Train-Test Split
    - Simple random split
    - Fast and straightforward
    - Results depend on split
    
    ### Model B - Repeated Random Splits
    - Multiple random splits
    - More robust estimates
    - Includes confidence intervals
    - Better assessment of variability
    
    ### Performance Metrics
    - **MSE**: Penalizes large errors
    - **MAE**: Average absolute error
    - **R² Score**: Goodness of fit
    
    ## 💡 Tips for Use
    
    1. **Dataset Requirements**:
       - Classification: Binary or multi-class labels
       - Regression: Continuous target variable
       - Clean data with minimal missing values
    
    2. **Feature Selection**:
       - Choose relevant features
       - Remove highly correlated features
       - Scale/normalize if needed
    
    3. **Model Selection**:
       - Consider computational resources
       - Evaluate multiple metrics
       - Look for consistent performance
    
    4. **Making Predictions**:
       - Use the same feature set
       - Ensure correct data types
       - Verify value ranges
    
    ## 🚀 Getting Started
    
    1. Upload your dataset (CSV format)
    2. Select target and feature columns
    3. Configure model parameters
    4. Click "Train Models"
    5. Compare results and select best model
    6. Download model for later use
    7. Make predictions with new data
    
    ## 📝 Submission Checklist
    
    - ✅ Python source code (.py file)
    - ✅ Datasets (CSV files)
    - ✅ Video screen recording (max 5 minutes)
    - ✅ Explanation of features
    - ✅ Sample predictions demonstration
    
    ## 🔗 Useful Resources
    
    - [Kaggle Datasets](https://www.kaggle.com/)
    - [UCI ML Repository](https://archive.ics.uci.edu/datasets)
    - [Google Dataset Search](https://datasetsearch.research.google.com/)
    
    ## ⚠️ Common Issues
    
    - **LOOCV taking too long**: Reduce dataset size or use K-Fold
    - **Poor model performance**: Check feature selection and data quality
    - **Scaling errors**: Ensure consistent preprocessing
    """)
    
    st.success("🎉 Lab Exercise #2 Complete! Good luck with your submission!")