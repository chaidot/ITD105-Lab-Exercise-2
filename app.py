import streamlit as st
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, log_loss, confusion_matrix,
                             classification_report, roc_auc_score,
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); }
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
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
    .main h3:first-of-type {
        text-align: center;
        color: #a5b4fc !important;
        font-weight: 400 !important;
        margin-top: 0 !important;
    }
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
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    [data-testid="stMarkdownContainer"] h2 {
        padding: 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classification_models_trained' not in st.session_state:
    st.session_state.classification_models_trained = False
if 'regression_models_trained' not in st.session_state:
    st.session_state.regression_models_trained = False

# Title
st.title("🔬 ITD105 - Big Data Analytics Lab Exercise #2")
st.markdown("### Resampling Techniques and Performance Metrics")

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Classification Task", "🌍 Regression Task", "📚 Documentation"])

# =============================================================================
# PART 1: CLASSIFICATION TASK
# =============================================================================
with tab1:
    st.header("Part 1: Health Data Classification")
    st.markdown("**Using Logistic Regression with Different Resampling Techniques**")
    
    uploaded_file_class = st.file_uploader("Upload your health dataset (CSV)", type=['csv'], key='class')
    
    if uploaded_file_class:
        # Load data
        df_class = pd.read_csv(uploaded_file_class)

        # Store for prediction defaults
        st.session_state.df_class_for_pred = df_class.copy()

        # Special cleaning for Pima Indians Diabetes dataset
        diabetes_cols_expected = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ]
        is_diabetes_dataset = set(diabetes_cols_expected).issubset(df_class.columns)

        if is_diabetes_dataset:
            cols_with_fake_zeros = [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
            ]
            df_class[cols_with_fake_zeros] = df_class[cols_with_fake_zeros].replace(0, np.nan)
            for col in cols_with_fake_zeros:
                df_class[col].fillna(df_class[col].median(), inplace=True)

            st.info(
                "✅ Detected **diabetes dataset**. Cleaned 0 values in "
                "`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` "
                "(treated as missing and imputed with median)."
            )
        
        # Dataset overview
        with st.expander("📊 Dataset Overview", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", df_class.shape[0])
            c2.metric("Total Features", df_class.shape[1])
            c3.metric("Memory Usage", f"{df_class.memory_usage().sum() / 1024:.2f} KB")
            
            st.dataframe(df_class.head(10), use_container_width=True)
            
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
        
        st.subheader("⚙️ Configure Model")
        st.info("💡 **Note**: Categorical features (like 'Male'/'Female') will be automatically encoded.")
        
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox("Select Target Column", df_class.columns, key='target_class')
        with c2:
            feature_cols = st.multiselect(
                "Select Feature Columns",
                [col for col in df_class.columns if col != target_col],
                default=[col for col in df_class.columns if col != target_col][:5]
            )
        
        if feature_cols and target_col:
            # Prepare features
            X_df = df_class[feature_cols].copy()
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            
            for col in X_df.columns:
                if X_df[col].dtype == 'object':
                    le = LabelEncoder()
                    X_df[col] = le.fit_transform(X_df[col].astype(str))
                    label_encoders[col] = le
            
            X = np.nan_to_num(X_df.values)
            
            # Target
            y_series = df_class[target_col].copy()
            if y_series.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y_series.astype(str))
            else:
                y = y_series.values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_folds = st.slider("Select number of folds for K-Fold CV", 3, 10, 5)
            
            if st.button("🚀 Train Models", key='train_class') or st.session_state.classification_models_trained:
                if st.session_state.classification_models_trained:
                    if st.button("🔄 Reset & Retrain", key='reset_class'):
                        st.session_state.classification_models_trained = False
                        st.session_state.pop('model_a_final', None)
                        st.session_state.pop('model_b_final', None)
                        st.rerun()
                
                if not st.session_state.classification_models_trained:
                    with st.spinner("Training models..."):
                        st.session_state.classification_models_trained = True
                else:
                    st.info("✅ Models already trained. Use prediction section below or click 'Reset & Retrain' to train again.")
                
                if st.session_state.classification_models_trained:
                    # -----------------------
                    # Model A: K-Fold
                    # -----------------------
                    st.subheader("📈 Model A: K-Fold Cross-Validation")

                    if 'model_a_final' not in st.session_state:
                        kfold_metrics = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                        acc_scores_kfold = []
                        log_loss_scores_kfold = []
                        auc_scores_kfold = []

                        all_classes = np.unique(y)

                        for train_idx, test_idx in kfold_metrics.split(X_scaled):
                            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                            y_train, y_test_local = y[train_idx], y[test_idx]
                            
                            model_a_tmp = LogisticRegression(max_iter=1000, random_state=42)
                            model_a_tmp.fit(X_train, y_train)
                            
                            y_pred_local = model_a_tmp.predict(X_test)
                            y_pred_proba_local = model_a_tmp.predict_proba(X_test)
                            
                            acc_scores_kfold.append(accuracy_score(y_test_local, y_pred_local))
                            try:
                                log_loss_scores_kfold.append(
                                    log_loss(y_test_local, y_pred_proba_local, labels=model_a_tmp.classes_)
                                )
                            except:
                                pass
                            
                            if len(all_classes) == 2:
                                try:
                                    auc_scores_kfold.append(
                                        roc_auc_score(y_test_local, y_pred_proba_local[:, 1])
                                    )
                                except:
                                    pass
                        
                        # final model on full data
                        model_a_final = LogisticRegression(max_iter=1000, random_state=42)
                        model_a_final.fit(X_scaled, y)
                        
                        st.session_state.model_a_final = model_a_final
                        st.session_state.acc_scores_kfold = acc_scores_kfold
                        st.session_state.log_loss_scores_kfold = log_loss_scores_kfold
                        st.session_state.auc_scores_kfold = auc_scores_kfold
                        st.session_state.scaler = scaler
                        st.session_state.feature_cols = feature_cols
                        st.session_state.label_encoders = label_encoders
            
                    model_a_final = st.session_state.model_a_final
                    acc_scores_kfold = st.session_state.acc_scores_kfold
                    log_loss_scores_kfold = st.session_state.log_loss_scores_kfold
                    auc_scores_kfold = st.session_state.auc_scores_kfold
                    scaler = st.session_state.scaler
                    feature_cols = st.session_state.feature_cols
                    label_encoders = st.session_state.get('label_encoders', {})
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Accuracy", f"{np.mean(acc_scores_kfold):.4f}")
                    if log_loss_scores_kfold:
                        c2.metric("Avg Log Loss", f"{np.mean(log_loss_scores_kfold):.4f}")
                    if auc_scores_kfold:
                        c3.metric("Avg AUC-ROC", f"{np.mean(auc_scores_kfold):.4f}")
                    
                    # ✅ Recompute CV predictions cleanly for CM & report
                    kfold_cm = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                    y_pred_cv = cross_val_predict(
                        LogisticRegression(max_iter=1000, random_state=42),
                        X_scaled,
                        y,
                        cv=kfold_cm
                    )
                    
                    cm_a = confusion_matrix(y, y_pred_cv)
                    fig_cm_a = px.imshow(
                        cm_a, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        title="Confusion Matrix - Model A (K-Fold, all data)"
                    )
                    st.plotly_chart(fig_cm_a, use_container_width=True)
                    
                    st.text("Classification Report - Model A (K-Fold, all data):")
                    st.text(classification_report(y, y_pred_cv))
                    
                    st.markdown("---")
                    
                    # -----------------------
                    # Model B: LOOCV
                    # -----------------------
                    st.subheader("📉 Model B: Leave-One-Out Cross-Validation")
                    
                    if 'model_b_final' not in st.session_state:
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
                            y_train, y_test_local = y_loocv[train_idx], y_loocv[test_idx]
                            
                            model_b = LogisticRegression(max_iter=1000, random_state=42)
                            model_b.fit(X_train, y_train)
                            
                            y_pred_local = model_b.predict(X_test)
                            y_true_loo.extend(y_test_local)
                            y_pred_loo.extend(y_pred_local)
                            acc_scores_loo.append(accuracy_score(y_test_local, y_pred_local))
                        
                        model_b_final = LogisticRegression(max_iter=1000, random_state=42)
                        model_b_final.fit(X_scaled, y)
                        
                        y_pred_proba_b = model_b_final.predict_proba(X_loocv)
                        try:
                            log_loss_b = log_loss(y_loocv, y_pred_proba_b, labels=model_b_final.classes_)
                        except:
                            log_loss_b = None
                        
                        if len(np.unique(y)) == 2:
                            try:
                                auc_b = roc_auc_score(y_loocv, y_pred_proba_b[:, 1])
                            except:
                                auc_b = None
                        else:
                            try:
                                auc_b = roc_auc_score(
                                    y_loocv, y_pred_proba_b,
                                    multi_class='ovr',
                                    labels=model_b_final.classes_
                                )
                            except:
                                auc_b = None
                        
                        st.session_state.model_b_final = model_b_final
                        st.session_state.acc_scores_loo = acc_scores_loo
                        st.session_state.y_true_loo = y_true_loo
                        st.session_state.y_pred_loo = y_pred_loo
                        st.session_state.log_loss_b = log_loss_b
                        st.session_state.auc_b = auc_b
                    
                    model_b_final = st.session_state.model_b_final
                    acc_scores_loo = st.session_state.acc_scores_loo
                    y_true_loo = st.session_state.y_true_loo
                    y_pred_loo = st.session_state.y_pred_loo
                    log_loss_b = st.session_state.log_loss_b
                    auc_b = st.session_state.auc_b
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{np.mean(acc_scores_loo):.4f}")
                    c2.metric("Log Loss", f"{log_loss_b:.4f}" if log_loss_b is not None else "N/A")
                    c3.metric("AUC-ROC", f"{auc_b:.4f}" if auc_b is not None else "N/A")
                    
                    cm_b = confusion_matrix(y_true_loo, y_pred_loo)
                    fig_cm_b = px.imshow(
                        cm_b, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        title="Confusion Matrix - Model B (LOOCV subset)"
                    )
                    st.plotly_chart(fig_cm_b, use_container_width=True)
                    
                    st.text("Classification Report - Model B:")
                    st.text(classification_report(y_true_loo, y_pred_loo))
                    
                    st.markdown("---")
                    
                    # Comparison table
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
                    **📊 Interpretation:**
                    - **Accuracy**: % of correct predictions (higher is better)
                    - **Log Loss**: confidence quality (lower is better)
                    - **AUC-ROC**: separation between classes (closer to 1 is better)
                    """)
                    
                    # Download models
                    st.subheader("💾 Download Model")
                    c1, c2 = st.columns(2)
                    with c1:
                        model_a_data = pickle.dumps({
                            'model': model_a_final,
                            'scaler': scaler,
                            'features': feature_cols,
                            'label_encoders': label_encoders
                        })
                        st.download_button(
                            "📥 Download Model A (K-Fold)",
                            data=model_a_data,
                            file_name="model_a_kfold.pkl",
                            mime="application/octet-stream"
                        )
                    with c2:
                        model_b_data = pickle.dumps({
                            'model': model_b_final,
                            'scaler': scaler,
                            'features': feature_cols,
                            'label_encoders': label_encoders
                        })
                        st.download_button(
                            "📥 Download Model B (LOOCV)",
                            data=model_b_data,
                            file_name="model_b_loocv.pkl",
                            mime="application/octet-stream"
                        )
                    
                    # -----------------------
                    # Prediction interface
                    # -----------------------
                    st.subheader("🔮 Make Predictions")
                    selected_model = st.radio(
                        "Select Model for Prediction",
                        ["Model A (K-Fold)", "Model B (LOOCV)"]
                    )
                    pred_model = model_a_final if "Model A" in selected_model else model_b_final
                    
                    st.write("**Enter feature values:**")
                    st.caption("For categorical features, choose from the dropdown.")
                    
                    input_features = []
                    input_features_display = {}
                    cols = st.columns(3)

                    # ✅ Use dataset statistics for defaults
                    df_pred_source = st.session_state.get("df_class_for_pred", None)
                    
                    for i, feature in enumerate(feature_cols):
                        with cols[i % 3]:
                            if feature in label_encoders:
                                categories = label_encoders[feature].classes_
                                default_index = 0
                                if df_pred_source is not None and feature in df_pred_source.columns:
                                    try:
                                        mode_val = df_pred_source[feature].mode().iloc[0]
                                        if mode_val in categories:
                                            default_index = list(categories).index(mode_val)
                                    except Exception:
                                        default_index = 0
                                
                                val = st.selectbox(
                                    f"{feature}",
                                    categories,
                                    index=default_index,
                                    key=f"pred_{feature}"
                                )
                                encoded_val = label_encoders[feature].transform([val])[0]
                                input_features.append(encoded_val)
                                input_features_display[feature] = val
                            else:
                                if df_pred_source is not None and feature in df_pred_source.columns:
                                    try:
                                        default_val = float(df_pred_source[feature].mean())
                                    except Exception:
                                        default_val = 0.0
                                else:
                                    default_val = 0.0
                                
                                val = st.number_input(
                                    f"{feature}",
                                    value=default_val,
                                    key=f"pred_{feature}"
                                )
                                input_features.append(val)
                                input_features_display[feature] = val
                    
                    if st.button("🎯 Predict", key='predict_class'):
                        input_scaled = scaler.transform([input_features])
                        prediction = pred_model.predict(input_scaled)[0]
                        prediction_proba = pred_model.predict_proba(input_scaled)[0]
                        
                        st.write("**Input Summary:**")
                        s1, s2 = st.columns(2)
                        items = list(input_features_display.items())
                        mid = len(items) // 2
                        with s1:
                            for f, v in items[:mid]:
                                st.write(f"- **{f}**: {v}")
                        with s2:
                            for f, v in items[mid:]:
                                st.write(f"- **{f}**: {v}")
                        
                        st.markdown("---")
                        st.info(f"**Using:** {selected_model}")
                        
                        r1, r2 = st.columns([1, 2])
                        with r1:
                            main_label = "Class 1" if prediction == 1 else "Class 0"
                            if is_diabetes_dataset and target_col == "Outcome":
                                main_label = "Diabetes" if prediction == 1 else "No Diabetes"
                            
                            st.metric("Predicted Class", main_label)
                            st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")
                        with r2:
                            prob_df = pd.DataFrame({
                                'Class': [f"{c}" for c in range(len(prediction_proba))],
                                'Probability': prediction_proba
                            })
                            fig_prob = px.bar(
                                prob_df, x='Class', y='Probability',
                                title='Prediction Probability Distribution',
                                color='Probability',
                                color_continuous_scale='blues',
                                text=[f"{p:.1%}" for p in prediction_proba]
                            )
                            fig_prob.update_traces(textposition='outside')
                            fig_prob.update_layout(showlegend=False, height=300, yaxis_range=[0, 1])
                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Detailed probs
                        st.write("**Detailed Probabilities:**")
                        pc = st.columns(len(prediction_proba))
                        for i, prob in enumerate(prediction_proba):
                            with pc[i]:
                                st.info(f"**Class {i}**\n\n{prob:.4f}\n\n({prob*100:.2f}%)")
                        
                        st.markdown("---")
                        st.write("**Result Interpretation:**")
                        
                        if is_diabetes_dataset and target_col == "Outcome" and len(prediction_proba) == 2:
                            prob_diabetes = float(prediction_proba[1])
                            if prob_diabetes >= 0.7:
                                level = "High diabetes risk"
                                msg = "Profile strongly suggests presence of diabetes."
                            elif prob_diabetes >= 0.4:
                                level = "Moderate diabetes risk"
                                msg = "Profile shows several risk indicators; follow-up is recommended."
                            else:
                                level = "Low diabetes risk"
                                msg = "Profile is closer to non-diabetic examples in the dataset."
                            
                            if prediction == 1:
                                st.warning(
                                    f"⚠️ The model predicts **Diabetes (class 1)** "
                                    f"with probability **{prob_diabetes*100:.2f}%**.\n\n"
                                    f"**Risk level:** {level}\n\n{msg}"
                                )
                            else:
                                st.success(
                                    f"✅ The model predicts **No Diabetes (class 0)**.\n\n"
                                    f"Estimated probability of diabetes is "
                                    f"**{prob_diabetes*100:.2f}%**.\n\n"
                                    f"**Risk level:** {level}\n\n{msg}"
                                )
                        else:
                            if prediction == 1:
                                st.warning(
                                    "⚠️ The model predicts **class 1** for this sample. "
                                    "Interpret the meaning of class 1 based on your dataset."
                                )
                            else:
                                st.success(
                                    "✅ The model predicts **class 0** for this sample. "
                                    "Interpret the meaning of class 0 based on your dataset."
                                )

# =============================================================================
# PART 2: REGRESSION TASK
# =============================================================================
with tab2:
    st.header("Part 2: Environmental Data Regression")
    st.markdown("**Using Linear Regression with Different Resampling Techniques**")
    
    uploaded_file_reg = st.file_uploader(
        "Upload your environment dataset (CSV)", type=['csv'], key='reg'
    )
    
    if uploaded_file_reg:
        # Load and basic cleaning
        df_reg = pd.read_csv(uploaded_file_reg)
        
        # Treat -200 in numeric columns as missing (AirQuality style)
        num_cols_all = df_reg.select_dtypes(include=[np.number]).columns
        df_reg[num_cols_all] = df_reg[num_cols_all].replace(-200, np.nan)
        
        df_reg_display = df_reg.copy()
        
        with st.expander("📊 Dataset Overview", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", df_reg_display.shape[0])
            c2.metric("Total Features", df_reg_display.shape[1])
            c3.metric("Memory Usage", f"{df_reg_display.memory_usage().sum() / 1024:.2f} KB")
            
            st.dataframe(df_reg_display.head(10), use_container_width=True)
            st.write("**Dataset Info:**")
            st.write(df_reg_display.describe())
            
            st.write("**Column Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df_reg_display.columns,
                'Data Type': df_reg_display.dtypes.values,
                'Non-Null Count': df_reg_display.count().values,
                'Unique Values': [df_reg_display[col].nunique() for col in df_reg_display.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        st.subheader("⚙️ Configure Model")
        st.info("💡 Select a numeric target, e.g. **CO(GT)** or another pollution/temperature column.")
        
        c1, c2 = st.columns(2)
        with c1:
            numeric_cols = df_reg.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("⚠️ No numeric columns found. Showing all columns; conversion may be needed.")
                numeric_cols = df_reg.columns.tolist()
            target_col_reg = st.selectbox(
                "Select Target Column (must be numeric)",
                numeric_cols,
                key='target_reg'
            )
        with c2:
            feature_cols_reg = st.multiselect(
                "Select Feature Columns",
                [col for col in df_reg.columns if col != target_col_reg],
                default=[col for col in df_reg.columns if col != target_col_reg][:5]
            )
        
        if feature_cols_reg and target_col_reg:
            # Build modelling DataFrame after knowing the target
            df_model = df_reg.copy()
            df_model = df_model.dropna(subset=[target_col_reg])               # drop missing target
            df_model = df_model.dropna(subset=feature_cols_reg, how="all")   # drop rows with all features NaN
            
            if df_model.empty:
                st.error("❌ After cleaning missing values, no data remains. Choose different target/features.")
                st.stop()
            
            st.session_state.df_reg_model = df_model.copy()
            
            # Feature preprocessing
            X_reg_df = df_model[feature_cols_reg].copy()
            from sklearn.preprocessing import LabelEncoder
            reg_label_encoders = {}
            reg_datetime_cols = []
            
            for col in X_reg_df.columns:
                if X_reg_df[col].dtype == "object":
                    try:
                        parsed = pd.to_datetime(X_reg_df[col])
                        if parsed.notna().mean() > 0.8:
                            X_reg_df[col] = parsed.astype("int64") / 10**9
                            reg_datetime_cols.append(col)
                            continue
                    except Exception:
                        pass
                    le = LabelEncoder()
                    X_reg_df[col] = le.fit_transform(X_reg_df[col].astype(str))
                    reg_label_encoders[col] = le
            
            num_cols_feats = X_reg_df.select_dtypes(include=[np.number]).columns
            X_reg_df[num_cols_feats] = X_reg_df[num_cols_feats].fillna(X_reg_df[num_cols_feats].mean())
            
            X_reg = np.nan_to_num(X_reg_df.values)
            
            # Target
            y_reg_series = df_model[target_col_reg].copy()
            if y_reg_series.dtype == "object":
                st.warning("⚠️ Target column contains non-numeric data. Attempting conversion...")
                try:
                    y_reg_series = pd.to_datetime(y_reg_series).astype("int64") / 10**9
                    st.info("✅ Target converted from date to numeric timestamp")
                except Exception:
                    try:
                        y_reg_series = pd.to_numeric(y_reg_series, errors="coerce")
                        st.info("✅ Target converted to numeric")
                    except Exception:
                        st.error("❌ Cannot convert target to numeric. Please select a numeric column.")
                        st.stop()
            
            y_reg = np.nan_to_num(y_reg_series.values)
            if not np.isfinite(y_reg).all():
                st.error("❌ Target column contains invalid values after conversion. Please select a different column.")
                st.stop()
            
            st.session_state.reg_label_encoders = reg_label_encoders
            st.session_state.reg_datetime_cols = reg_datetime_cols
            
            scaler_reg = StandardScaler()
            X_reg_scaled = scaler_reg.fit_transform(X_reg)
            
            c1, c2 = st.columns(2)
            with c1:
                test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            with c2:
                n_repeats = st.slider("Number of Repeated Splits", 3, 20, 10)
            
            if st.button("🚀 Train Models", key='train_reg') or st.session_state.regression_models_trained:
                if st.session_state.regression_models_trained:
                    if st.button("🔄 Reset & Retrain", key='reset_reg'):
                        st.session_state.regression_models_trained = False
                        st.session_state.pop('model_reg_a', None)
                        st.session_state.pop('model_reg_b_final', None)
                        st.rerun()
                
                if not st.session_state.regression_models_trained:
                    with st.spinner("Training models..."):
                        st.session_state.regression_models_trained = True
                else:
                    st.info("✅ Models already trained. Use prediction section below or click 'Reset & Retrain' to train again.")
                
                if st.session_state.regression_models_trained:
                    # -------------------
                    # Model A: single split
                    # -------------------
                    st.subheader("📈 Model A: Train-Test Split")
                    if 'model_reg_a' not in st.session_state:
                        X_train, X_test, y_train, y_test_local = train_test_split(
                            X_reg_scaled, y_reg, test_size=test_size, random_state=42
                        )
                        model_reg_a = LinearRegression()
                        model_reg_a.fit(X_train, y_train)
                        y_pred_a = model_reg_a.predict(X_test)
                        mse_a = mean_squared_error(y_test_local, y_pred_a)
                        mae_a = mean_absolute_error(y_test_local, y_pred_a)
                        r2_a = r2_score(y_test_local, y_pred_a)
                        residuals_a = y_test_local - y_pred_a
                        
                        st.session_state.model_reg_a = model_reg_a
                        st.session_state.y_test_reg_a = y_test_local
                        st.session_state.y_pred_a = y_pred_a
                        st.session_state.mse_a = mse_a
                        st.session_state.mae_a = mae_a
                        st.session_state.r2_a = r2_a
                        st.session_state.residuals_a = residuals_a
                        st.session_state.scaler_reg = scaler_reg
                        st.session_state.feature_cols_reg = feature_cols_reg
                    
                    model_reg_a = st.session_state.model_reg_a
                    y_test_local = st.session_state.y_test_reg_a
                    y_pred_a = st.session_state.y_pred_a
                    mse_a = st.session_state.mse_a
                    mae_a = st.session_state.mae_a
                    r2_a = st.session_state.r2_a
                    residuals_a = st.session_state.residuals_a
                    scaler_reg = st.session_state.scaler_reg
                    feature_cols_reg = st.session_state.feature_cols_reg
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MSE", f"{mse_a:.4f}")
                    c2.metric("MAE", f"{mae_a:.4f}")
                    c3.metric("R² Score", f"{r2_a:.4f}")
                    
                    fig_a = go.Figure()
                    fig_a.add_trace(go.Scatter(x=y_test_local, y=y_pred_a, mode='markers', name='Predictions'))
                    fig_a.add_trace(go.Scatter(
                        x=y_test_local, y=y_test_local,
                        mode='lines', name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_a.update_layout(
                        title="Model A: Predicted vs Actual",
                        xaxis_title="Actual",
                        yaxis_title="Predicted"
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
                    
                    fig_res_a = px.scatter(
                        x=y_pred_a, y=residuals_a,
                        labels={'x': 'Predicted', 'y': 'Residuals'},
                        title="Model A: Residual Plot"
                    )
                    fig_res_a.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res_a, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # -------------------
                    # Model B: repeated splits
                    # -------------------
                    st.subheader("📉 Model B: Repeated Random Train-Test Splits")
                    if 'model_reg_b_final' not in st.session_state:
                        mse_scores_b, mae_scores_b, r2_scores_b = [], [], []
                        
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
                        
                        model_reg_b_final = LinearRegression()
                        model_reg_b_final.fit(X_reg_scaled, y_reg)
                        
                        st.session_state.model_reg_b_final = model_reg_b_final
                        st.session_state.mse_scores_b = mse_scores_b
                        st.session_state.mae_scores_b = mae_scores_b
                        st.session_state.r2_scores_b = r2_scores_b
                    
                    model_reg_b_final = st.session_state.model_reg_b_final
                    mse_scores_b = st.session_state.mse_scores_b
                    mae_scores_b = st.session_state.mae_scores_b
                    r2_scores_b = st.session_state.r2_scores_b
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg MSE", f"{np.mean(mse_scores_b):.4f}", delta=f"±{np.std(mse_scores_b):.4f}")
                    c2.metric("Avg MAE", f"{np.mean(mae_scores_b):.4f}", delta=f"±{np.std(mae_scores_b):.4f}")
                    c3.metric("Avg R² Score", f"{np.mean(r2_scores_b):.4f}", delta=f"±{np.std(r2_scores_b):.4f}")
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Box(y=mse_scores_b, name='MSE'))
                    fig_dist.add_trace(go.Box(y=mae_scores_b, name='MAE'))
                    fig_dist.add_trace(go.Box(y=r2_scores_b, name='R²'))
                    fig_dist.update_layout(
                        title="Model B: Score Distribution Across Splits",
                        yaxis_title="Score Value"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.markdown("---")
                    
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
                    **📊 Regression metrics:**
                    - **MSE / MAE** → lower is better
                    - **R²** → closer to 1 means better fit
                    """)
                    
                    # Download models
                    st.subheader("💾 Download Model")
                    c1, c2 = st.columns(2)
                    with c1:
                        model_reg_a_data = pickle.dumps({
                            'model': model_reg_a,
                            'scaler': scaler_reg,
                            'features': feature_cols_reg
                        })
                        st.download_button(
                            "📥 Download Model A (Single Split)",
                            data=model_reg_a_data,
                            file_name="model_reg_a.pkl",
                            mime="application/octet-stream"
                        )
                    with c2:
                        model_reg_b_data = pickle.dumps({
                            'model': model_reg_b_final,
                            'scaler': scaler_reg,
                            'features': feature_cols_reg
                        })
                        st.download_button(
                            "📥 Download Model B (Repeated Splits)",
                            data=model_reg_b_data,
                            file_name="model_reg_b.pkl",
                            mime="application/octet-stream"
                        )
                    
                    # -------------------
                    # Prediction interface (regression)
                    # -------------------
                    st.subheader("🔮 Make Predictions")
                    reg_label_encoders = st.session_state.get("reg_label_encoders", {})
                    reg_datetime_cols = st.session_state.get("reg_datetime_cols", [])
                    df_reg_model = st.session_state.get("df_reg_model", df_model)
                    
                    selected_model_reg = st.radio(
                        "Select Model for Prediction",
                        ["Model A (Single Split)", "Model B (Repeated Splits)"],
                        key='select_reg'
                    )
                    pred_model_reg = model_reg_a if "Model A" in selected_model_reg else model_reg_b_final
                    
                    st.write("**Enter feature values:**")
                    input_features_reg = []
                    cols = st.columns(3)
                    
                    for i, feature in enumerate(feature_cols_reg):
                        with cols[i % 3]:
                            if feature in reg_datetime_cols:
                                try:
                                    example_value = str(df_reg_model[feature].dropna().iloc[0])
                                except Exception:
                                    example_value = "e.g. 10/03/2004 18.00.00"
                                
                                user_text = st.text_input(
                                    f"{feature} (same format as in CSV)",
                                    value=example_value,
                                    key=f"pred_reg_{feature}"
                                )
                                try:
                                    parsed = pd.to_datetime(user_text)
                                    val = parsed.value / 10**9
                                except Exception:
                                    st.error(f"❌ Could not parse '{feature}'. Please match the format in your CSV.")
                                    val = 0.0
                            
                            elif feature in reg_label_encoders:
                                categories = reg_label_encoders[feature].classes_
                                selected_cat = st.selectbox(
                                    feature,
                                    options=categories,
                                    key=f"pred_reg_{feature}"
                                )
                                val = reg_label_encoders[feature].transform([selected_cat])[0]
                            
                            else:
                                if pd.api.types.is_numeric_dtype(df_reg_model[feature]):
                                    default_val = float(df_reg_model[feature].mean())
                                else:
                                    default_val = 0.0
                                val = st.number_input(
                                    feature,
                                    value=default_val,
                                    key=f"pred_reg_{feature}"
                                )
                            
                            input_features_reg.append(val)
                    
                    if st.button("🎯 Predict", key='predict_reg'):
                        input_scaled_reg = scaler_reg.transform([input_features_reg])
                        prediction_raw = pred_model_reg.predict(input_scaled_reg)[0]
                        prediction_reg = float(max(prediction_raw, 0.0))  # no negative pollutants
                        
                        st.success(f"**Predicted {target_col_reg}:** {prediction_reg:.4f}")
                        
                        fig_pred = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction_reg,
                            title={'text': f"Predicted {target_col_reg}"},
                            gauge={
                                'axis': {
                                    'range': [
                                        float(min(np.min(y_reg), prediction_reg, 0.0)),
                                        float(max(np.max(y_reg), prediction_reg)) * 1.1
                                    ]
                                }
                            }
                        ))
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Interpretation
                        try:
                            y_min = float(np.nanmin(y_reg))
                            y_max = float(np.nanmax(y_reg))
                            y_mean = float(np.nanmean(y_reg))
                            q1, q3 = np.percentile(y_reg, [25, 75])
                            
                            if prediction_reg <= q1:
                                level = "Low"
                                qualitative = "relatively **low** compared to past observations."
                            elif prediction_reg >= q3:
                                level = "High"
                                qualitative = "relatively **high** compared to past observations."
                            else:
                                level = "Moderate"
                                qualitative = "within the **typical/average** range of past observations."
                            
                            st.markdown("---")
                            st.subheader("🧾 Interpretation (Environmental Context)")
                            st.markdown(f"""
- **Predicted `{target_col_reg}`:** `{prediction_reg:.4f}`
- **Training data range:** `{y_min:.4f}` to `{y_max:.4f}`
- **Training mean:** `{y_mean:.4f}`
- **Relative level:** **{level}**

For this environmental variable (**{target_col_reg}**), the predicted value is {qualitative}

**Reading this for air-quality or environmental data:**

- If **{target_col_reg}** is a *pollutant* (e.g., CO(GT), NO₂, PM, etc.):
  - **Low** → cleaner / better air conditions  
  - **Moderate** → typical background level in your dataset  
  - **High** → more polluted / potentially unhealthy conditions  

- If **{target_col_reg}** is a *weather / physical variable* (e.g., temperature, humidity):
  - **Low / High** → near the extremes seen in your data  
  - **Moderate** → close to usual conditions  

Always interpret the value in the context of what **{target_col_reg}** actually measures and the units/thresholds defined in your course or dataset.
""")
                        except Exception:
                            st.warning("Could not compute detailed interpretation, but the numeric prediction above is still valid.")

# =============================================================================
# DOCUMENTATION TAB
# =============================================================================
with tab3:
    st.header("📚 Lab Exercise Documentation")
    st.markdown("""
    ## 🎯 Objectives
    - Demonstrate resampling techniques (K-Fold, LOOCV, repeated train-test splits)
    - Compare classification and regression performance metrics
    - Provide interactive prediction interfaces

    ## 📋 Part 1: Classification
    - Logistic Regression with:
      - **Model A:** K-Fold Cross-Validation
      - **Model B:** Leave-One-Out Cross-Validation
    - Metrics: Accuracy, Log Loss, Confusion Matrix, AUC-ROC

    ## 🌍 Part 2: Regression
    - Linear Regression with:
      - **Model A:** Single Train-Test Split
      - **Model B:** Repeated Random Splits
    - Metrics: MSE, MAE, R² Score

    ## 📝 Submission Checklist
    - ✅ Source code (.py)
    - ✅ Datasets (CSV)
    - ✅ Short demo video
    - ✅ Explanation of selected features & metrics
    - ✅ Sample predictions & interpretations
    """)
    st.success("🎉 Lab Exercise #2 Complete! Good luck with your submission! 🫶")
