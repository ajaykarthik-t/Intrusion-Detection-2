import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64
import os
import time
import joblib
from io import BytesIO
from datetime import datetime

# ML libraries
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, auc, 
                           precision_recall_curve)
from sklearn.feature_selection import SelectKBest, f_classif

# Initialize the app state
def initialize_session_state():
    """Initialize all session state variables to ensure they exist"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'categorical_cols' not in st.session_state:
        st.session_state.categorical_cols = None
    if 'numerical_cols' not in st.session_state:
        st.session_state.numerical_cols = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []

# --------------------------
# DATA GENERATION & HANDLING
# --------------------------

@st.cache_data
def generate_synthetic_data(n_samples=1000, n_classes=2, class_imbalance=0.2, random_seed=42):
    """
    Generate synthetic network intrusion data with customizable parameters
    
    Parameters:
    -----------
    n_samples: int
        Number of samples to generate
    n_classes: int
        Number of target classes (2-5)
    class_imbalance: float
        For binary classification, determines ratio of attack samples (0.0-0.5)
    random_seed: int
        Random seed for reproducibility
        
    Returns:
    --------
    DataFrame with synthetic network data
    """
    # Validate inputs
    if n_classes < 2 or n_classes > 5:
        raise ValueError("Number of classes must be between 2 and 5")
    
    if class_imbalance < 0.0 or class_imbalance > 0.5:
        raise ValueError("Class imbalance must be between 0.0 and 0.5")
    
    # Define class weights for different scenarios
    if n_classes == 2:
        # Binary classification: Normal vs Attack
        weights = [1 - class_imbalance, class_imbalance]
    elif n_classes == 3:
        # 3 classes: Normal, DOS, Probe
        weights = [0.7, 0.2, 0.1]
    elif n_classes == 4:
        # 4 classes: Normal, DOS, Probe, R2L
        weights = [0.7, 0.15, 0.1, 0.05]
    else:
        # 5 classes: Normal, DOS, Probe, R2L, U2R (U2R is typically very rare)
        weights = [0.7, 0.15, 0.08, 0.05, 0.02]
    
    # Generate synthetic data with sklearn's make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,       # Fixed number of features for simplicity
        n_informative=8,     # Most features are informative
        n_redundant=2,       # Some features are redundant
        n_classes=n_classes,
        weights=weights,     # Class imbalance as specified
        random_state=random_seed
    )
    
    # Create DataFrame with named features
    feature_names = [
        'duration',          # Connection duration in seconds
        'protocol_type',     # Type of protocol (TCP, UDP, ICMP)
        'service',           # Service on destination (HTTP, FTP, etc.)
        'flag',              # Status flag of the connection
        'src_bytes',         # Bytes sent from source to destination
        'dst_bytes',         # Bytes sent from destination to source
        'land',              # 1 if connection is from/to same host/port; 0 otherwise
        'wrong_fragment',    # Number of wrong fragments
        'urgent',            # Number of urgent packets
        'count'              # Number of connections to the same host in past 2 seconds
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Map class labels to meaningful names
    if n_classes == 2:
        class_names = ['Normal', 'Attack']
    else:
        class_names = ['Normal', 'DOS', 'Probe', 'R2L', 'U2R'][:n_classes]
    
    df['class'] = [class_names[i] for i in y]
    
    # Convert numerical values to meaningful categorical data for selected columns
    # Protocol type: tcp, udp, icmp
    df['protocol_type'] = df['protocol_type'].apply(
        lambda x: ['tcp', 'udp', 'icmp'][int((x + 3) * 10) % 3]
    )
    
    # Service: http, ftp, smtp, ssh, dns
    service_options = ['http', 'ftp', 'smtp', 'ssh', 'dns', 'pop3', 'telnet', 'imap', 'sql']
    df['service'] = df['service'].apply(
        lambda x: service_options[int((x + 3) * 10) % len(service_options)]
    )
    
    # Flag: SF (normal), S0 (rejected), REJ (rejected), RSTO, RSTR
    flag_options = ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2']
    df['flag'] = df['flag'].apply(
        lambda x: flag_options[int((x + 3) * 10) % len(flag_options)]
    )
    
    # Make some realistic correlations:
    
    # 1. Normal traffic tends to have longer duration and complete connections
    mask_normal = df['class'] == 'Normal'
    df.loc[mask_normal, 'duration'] = df.loc[mask_normal, 'duration'].abs() + 1
    df.loc[mask_normal, 'flag'] = df.loc[mask_normal, 'flag'].apply(
        lambda x: 'SF' if np.random.rand() < 0.8 else x
    )
    
    # 2. DOS attacks have many connections with little data transfer
    if 'DOS' in class_names:
        mask_dos = df['class'] == 'DOS'
        df.loc[mask_dos, 'count'] = df.loc[mask_dos, 'count'].abs() + 10
        df.loc[mask_dos, 'src_bytes'] = df.loc[mask_dos, 'src_bytes'].abs() * 0.2
        df.loc[mask_dos, 'flag'] = df.loc[mask_dos, 'flag'].apply(
            lambda x: 'S0' if np.random.rand() < 0.7 else 'REJ'
        )
    
    # 3. Probe attacks often use ICMP and have very short duration
    if 'Probe' in class_names:
        mask_probe = df['class'] == 'Probe'
        df.loc[mask_probe, 'duration'] = df.loc[mask_probe, 'duration'].abs() * 0.1
        df.loc[mask_probe, 'protocol_type'] = df.loc[mask_probe, 'protocol_type'].apply(
            lambda x: 'icmp' if np.random.rand() < 0.6 else x
        )
    
    # 4. R2L attacks often happen over services like FTP, with normal-looking connections
    if 'R2L' in class_names:
        mask_r2l = df['class'] == 'R2L'
        df.loc[mask_r2l, 'service'] = df.loc[mask_r2l, 'service'].apply(
            lambda x: 'ftp' if np.random.rand() < 0.5 else ('smtp' if np.random.rand() < 0.3 else x)
        )
        df.loc[mask_r2l, 'flag'] = df.loc[mask_r2l, 'flag'].apply(
            lambda x: 'SF' if np.random.rand() < 0.7 else x
        )
    
    # 5. U2R attacks often have normal connection characteristics but may have unusual byte counts
    if 'U2R' in class_names:
        mask_u2r = df['class'] == 'U2R'
        df.loc[mask_u2r, 'flag'] = 'SF'  # Usually normal connections
        df.loc[mask_u2r, 'dst_bytes'] = df.loc[mask_u2r, 'dst_bytes'].abs() * 5  # Unusual data transfer
    
    # Clean up numerical columns to ensure they're appropriate
    # Ensure bytes are non-negative
    df['src_bytes'] = df['src_bytes'].abs()
    df['dst_bytes'] = df['dst_bytes'].abs()
    
    # Ensure land is binary
    df['land'] = (df['land'] > 0.5).astype(float)
    
    # Ensure wrong_fragment and urgent are non-negative and mostly zero
    df['wrong_fragment'] = (df['wrong_fragment'] > 1.0).astype(float) * df['wrong_fragment'].abs() * 0.2
    df['urgent'] = (df['urgent'] > 1.5).astype(float) * df['urgent'].abs() * 0.1
    
    # Ensure count is positive
    df['count'] = df['count'].abs() + 1
    
    # Round numerical values for readability
    for col in ['duration', 'src_bytes', 'dst_bytes', 'count']:
        df[col] = df[col].round(2)
    
    # Add a timestamp column for realism
    now = datetime.now()
    df['timestamp'] = [now - pd.Timedelta(seconds=int(x*100)) for x in np.random.rand(len(df))]
    
    # Add source and destination IP addresses
    df['src_ip'] = [f"192.168.{np.random.randint(1, 254)}.{np.random.randint(1, 254)}" for _ in range(len(df))]
    df['dst_ip'] = [f"10.0.{np.random.randint(1, 254)}.{np.random.randint(1, 254)}" for _ in range(len(df))]
    
    return df

def save_dataset(df, filename="intrusion_dataset.csv"):
    """Save dataset to a local file with error handling"""
    try:
        df.to_csv(filename, index=False)
        return True, f"Dataset successfully saved to {filename}"
    except Exception as e:
        return False, f"Error saving dataset: {str(e)}"

def load_dataset(file_buffer):
    """Load dataset from uploaded file"""
    try:
        # Check file extension
        if file_buffer.name.endswith('.csv'):
            df = pd.read_csv(file_buffer)
        elif file_buffer.name.endswith('.xlsx') or file_buffer.name.endswith('.xls'):
            df = pd.read_excel(file_buffer)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        
        # Validate that the dataset has expected columns
        required_columns = ['protocol_type', 'service', 'flag', 'class']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Dataset is missing required columns: {', '.join(missing_columns)}"
        
        # Identify categorical and numerical columns
        categorical_cols = ['protocol_type', 'service', 'flag']
        numerical_cols = [col for col in df.columns if col not in categorical_cols and col != 'class']
        
        st.session_state.categorical_cols = categorical_cols
        st.session_state.numerical_cols = numerical_cols
        
        return df, "Dataset loaded successfully!"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def get_download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for a file"""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="{download_filename}" class="download-button">{download_link_text}</a>'
    elif isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}" class="download-button">{download_link_text}</a>'
    return href

# ------------------------------
# DATA PREPROCESSING & MODELING
# ------------------------------

def perform_eda(df):
    """Perform exploratory data analysis on the dataset"""
    st.subheader("Dataset Overview")
    
    # Basic info
    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("First 5 rows:")
    st.dataframe(df.head())
    
    # Class distribution
    st.subheader("Class Distribution")
    class_counts = df['class'].value_counts()
    
    # Show as table and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(pd.DataFrame({
            'Class': class_counts.index,
            'Count': class_counts.values,
            'Percentage': (class_counts.values / len(df) * 100).round(2)
        }))
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.barplot(x=class_counts.index, y=class_counts.values)
        ax.set_title('Distribution of Classes')
        ax.set_ylabel('Count')
        ax.set_xlabel('Class')
        
        # Add count labels on top of bars
        for i, count in enumerate(class_counts.values):
            ax.text(i, count + 5, str(count), ha='center')
        
        st.pyplot(fig)
    
    # Feature distributions
    with st.expander("Explore Feature Distributions"):
        # Let user select features to explore
        numerical_cols = st.session_state.numerical_cols
        
        if len(numerical_cols) > 0:
            selected_features = st.multiselect(
                "Select features to explore:",
                numerical_cols,
                default=numerical_cols[:min(3, len(numerical_cols))]
            )
            
            if selected_features:
                st.subheader("Feature Distributions")
                
                for feature in selected_features:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Create distribution plot, colored by class
                    sns.histplot(data=df, x=feature, hue='class', kde=True, element="step", common_norm=False, ax=ax)
                    ax.set_title(f'Distribution of {feature} by Class')
                    st.pyplot(fig)
        
        # Show categorical feature distributions
        categorical_cols = st.session_state.categorical_cols
        if categorical_cols:
            st.subheader("Categorical Feature Distributions")
            
            for cat_col in categorical_cols:
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Count plot for categorical variables
                cat_counts = df.groupby([cat_col, 'class']).size().unstack().fillna(0)
                cat_counts.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title(f'{cat_col} by Class')
                ax.set_ylabel('Count')
                
                st.pyplot(fig)
    
    # Correlation analysis
    with st.expander("Correlation Analysis"):
        st.subheader("Correlation Between Features")
        corr = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        st.markdown("""
        **Interpreting the Correlation Matrix:**
        * Values close to 1 indicate strong positive correlation 
        * Values close to -1 indicate strong negative correlation
        * Values close to 0 indicate little to no linear correlation
        """)
    
    # Feature importance analysis (if we have a model)
    if st.session_state.model is not None and hasattr(st.session_state.model, 'feature_importances_'):
        with st.expander("Feature Importance Analysis"):
            st.subheader("Feature Importance from Current Model")
            
            feature_imp = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
    
    # Additional statistics for numerical features
    with st.expander("Detailed Statistics"):
        st.subheader("Numerical Feature Statistics")
        st.dataframe(df[numerical_cols].describe())
        
        # Show value counts for categorical features
        st.subheader("Categorical Feature Value Counts")
        for col in categorical_cols:
            st.write(f"### {col}")
            st.dataframe(df[col].value_counts().reset_index().rename(
                columns={'index': col, col: 'Count'}
            ))

def build_preprocessing_pipeline(df, categorical_cols, numerical_cols, feature_selection=False, k_features=None):
    """
    Create a preprocessing pipeline with categorical and numerical transformers
    
    Parameters:
    -----------
    df: DataFrame
        The dataset to process
    categorical_cols: list
        List of categorical column names
    numerical_cols: list
        List of numerical column names
    feature_selection: bool
        Whether to perform feature selection
    k_features: int or None
        Number of features to select if feature_selection is True
        
    Returns:
    --------
    A sklearn ColumnTransformer pipeline
    """
    # Categorical features transformer - One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Numerical features transformer - Standard Scaling
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Build final pipeline with optional feature selection
    if feature_selection and k_features:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selector', SelectKBest(f_classif, k=k_features)),
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
    
    return pipeline

def prepare_data_for_training(df, test_size=0.2, random_state=42):
    """
    Prepare data for model training with preprocessing pipeline
    
    Parameters:
    -----------
    df: DataFrame
        The dataset to process
    test_size: float
        Percentage of data to use for testing
    random_state: int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, preprocessing_pipeline
    """
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    st.session_state.feature_names = feature_names
    
    # Get categorical and numerical columns
    categorical_cols = st.session_state.categorical_cols
    numerical_cols = st.session_state.numerical_cols
    
    # Create label encoder for the target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    st.session_state.label_encoder = le
    st.session_state.class_names = le.classes_
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Build preprocessing pipeline
    feature_selection = st.session_state.get('feature_selection', False)
    k_features = st.session_state.get('k_features', None)
    
    preprocessing_pipeline = build_preprocessing_pipeline(
        df, categorical_cols, numerical_cols, feature_selection, k_features
    )
    
    return X_train, X_test, y_train, y_test, preprocessing_pipeline

def create_model(model_type, hyperparameters=None):
    """
    Create a machine learning model based on model_type
    
    Parameters:
    -----------
    model_type: str
        The type of model to create
    hyperparameters: dict
        Hyperparameters for the model
        
    Returns:
    --------
    A sklearn estimator
    """
    if hyperparameters is None:
        hyperparameters = {}
    
    if model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', None),
            min_samples_split=hyperparameters.get('min_samples_split', 2),
            random_state=42
        )
    elif model_type == "Logistic Regression":
        return LogisticRegression(
            C=hyperparameters.get('C', 1.0),
            max_iter=hyperparameters.get('max_iter', 1000),
            random_state=42
        )
    elif model_type == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            learning_rate=hyperparameters.get('learning_rate', 0.1),
            max_depth=hyperparameters.get('max_depth', 3),
            random_state=42
        )
    elif model_type == "Support Vector Machine":
        return SVC(
            C=hyperparameters.get('C', 1.0),
            kernel=hyperparameters.get('kernel', 'rbf'),
            probability=True,
            random_state=42
        )
    elif model_type == "Neural Network":
        return MLPClassifier(
            hidden_layer_sizes=hyperparameters.get('hidden_layer_sizes', (100,)),
            max_iter=hyperparameters.get('max_iter', 1000),
            alpha=hyperparameters.get('alpha', 0.0001),
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model_with_pipeline(pipeline, model, X_train, y_train, X_test, y_test):
    """
    Train a model with preprocessing pipeline and evaluate it
    
    Parameters:
    -----------
    pipeline: sklearn Pipeline
        Preprocessing pipeline
    model: sklearn estimator
        ML model to train
    X_train, y_train, X_test, y_test: data splits
        Training and testing data
        
    Returns:
    --------
    trained_pipeline, metrics_dict
    """
    # Create full pipeline with preprocessing and model
    full_pipeline = Pipeline(steps=[
        ('preprocessing', pipeline),
        ('model', model)
    ])
    
    # Record training start time
    start_time = time.time()
    
    # Fit pipeline to training data
    full_pipeline.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = full_pipeline.predict(X_test)
    
    # Get prediction probabilities if available
    if hasattr(full_pipeline, "predict_proba"):
        y_proba = full_pipeline.predict_proba(X_test)
    else:
        y_proba = None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'training_time': training_time,
        'model_type': type(model).__name__
    }
    
    # Store training record
    training_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': type(model).__name__,
        'metrics': metrics,
        'parameters': model.get_params()
    }
    
    st.session_state.training_history.append(training_record)
    
    return full_pipeline, metrics, y_pred, y_proba

def tune_hyperparameters(pipeline, model_type, X_train, y_train, param_grid):
    """Tune hyperparameters using GridSearchCV"""
    
    # Create base model
    base_model = create_model(model_type)
    
    # Create full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessing', pipeline),
        ('model', base_model)
    ])
    
    # Set up parameter grid for GridSearchCV
    # We need to prefix parameter names with 'model__'
    prefixed_param_grid = {'model__' + k: v for k, v in param_grid.items()}
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        full_pipeline, 
        prefixed_param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit GridSearchCV
    with st.spinner("Tuning hyperparameters... This may take a while."):
        grid_search.fit(X_train, y_train)
    
    # Get best parameters (removing 'model__' prefix)
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    
    # Create model with best parameters
    best_model = create_model(model_type, best_params)
    
    return best_model, best_params, grid_search.best_score_

def evaluate_model(pipeline, X_test, y_test, class_names):
    """Create detailed evaluation plots for model performance"""
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    
    # Get prediction probabilities if available
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)
    else:
        y_proba = None
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # 2. ROC Curve and Precision-Recall Curve (for each class if multiclass)
    fpr = {}
    tpr = {}
    roc_auc = {}
    precision = {}
    recall = {}
    pr_auc = {}
    
    n_classes = len(class_names)
    
    if y_proba is not None:
        # One-hot encode true labels for ROC curve calculation
        y_test_bin = np.zeros((len(y_test), n_classes))
        for i in range(len(y_test)):
            y_test_bin[i, y_test[i]] = 1
        
        # Calculate ROC and PR curves for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
        
        # Plot ROC Curves
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        for i in range(n_classes):
            ax2.plot(fpr[i], tpr[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax2.legend(loc="lower right")
        
        # Plot Precision-Recall Curves
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        
        for i in range(n_classes):
            ax3.plot(recall[i], precision[i], lw=2,
                    label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})')
        
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend(loc="lower left")
        
        return fig1, fig2, fig3
    
    # If no probabilities available, just return confusion matrix
    return fig1, None, None

def explain_prediction(model, input_data, class_names, feature_names, categorical_cols, numerical_cols):
    """Provide explanation for a prediction"""
    
    # Ensure input data is in the right format
    if isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        input_df = pd.DataFrame([input_data])
    
    # Get the prediction and probabilities
    pred = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]
    
    # Convert prediction to class name
    pred_class = class_names[pred]
    
    # Get top features contributing to the prediction
    explanation = {}
    
    # For tree-based models, we can get feature importances
    if hasattr(model, 'feature_importances_'):
        # Extract the model from the pipeline
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            model_step = model.named_steps['model']
            if hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
                
                # Get feature names from the preprocessor
                if hasattr(model, 'named_steps') and 'preprocessing' in model.named_steps:
                    preprocessor = model.named_steps['preprocessing']
                    if hasattr(preprocessor, 'transformers_'):
                        # Get transformed feature names
                        all_transformed_features = []
                        
                        # Extract numerical features (they keep their names)
                        all_transformed_features.extend(numerical_cols)
                        
                        # Extract categorical features (they become one-hot encoded)
                        for cat_col in categorical_cols:
                            # Get unique values for this categorical column
                            unique_vals = input_df[cat_col].unique()
                            for val in unique_vals:
                                all_transformed_features.append(f"{cat_col}_{val}")
                        
                        # Create a mapping of feature importances
                        feature_imp = {
                            feature: imp 
                            for feature, imp in zip(all_transformed_features[:len(importances)], importances)
                        }
                        
                        # Sort by importance and get top features
                        top_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:5]
                        explanation['top_features'] = top_features
    
    # For all models, we can check if high/low values contributed
    # For numerical features, compare to the average value
    for col in numerical_cols:
        if col in input_df.columns:
            val = input_df[col].values[0]
            
            # We'd need training data averages to compare properly
            # This is a simplified version
            if val > 10:  # Arbitrary threshold - replace with actual data analysis
                explanation[f'high_{col}'] = val
            elif val < 0.1:  # Arbitrary threshold - replace with actual data analysis
                explanation[f'low_{col}'] = val
    
    # For categorical features, note unusual values
    for col in categorical_cols:
        if col in input_df.columns:
            val = input_df[col].values[0]
            
            # Check for suspicious values based on column
            if col == 'flag' and val in ['S0', 'REJ']:
                explanation[f'unusual_{col}'] = val
            elif col == 'protocol_type' and val == 'icmp':
                if pred_class != 'Normal':
                    explanation[f'unusual_{col}'] = val
    
    # Return the prediction details and explanation
    return {
        'prediction': pred_class,
        'confidence': probas[pred],
        'probabilities': {class_name: prob for class_name, prob in zip(class_names, probas)},
        'explanation': explanation
    }

# -------------------
# ALERT HANDLING
# -------------------

def record_alert(prediction, input_data, timestamp=None):
    """
    Record a new alert in the system
    
    Parameters:
    -----------
    prediction: dict
        The prediction result including class and confidence
    input_data: dict
        The input data that triggered the alert
    timestamp: datetime (optional)
        The timestamp of the alert, defaults to current time
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create alert record
    alert = {
        'timestamp': timestamp,
        'prediction': prediction['prediction'],
        'confidence': prediction['confidence'],
        'data': input_data,
        'status': 'New',
        'id': len(st.session_state.alerts) + 1
    }
    
    # Add alert to session state
    st.session_state.alerts.append(alert)
    
    return alert

def get_alert_summary():
    """Get a summary of recorded alerts"""
    alerts = st.session_state.alerts
    
    if not alerts:
        return "No alerts recorded"
    
    # Count alerts by type
    alert_counts = {}
    for alert in alerts:
        alert_type = alert['prediction']
        if alert_type in alert_counts:
            alert_counts[alert_type] += 1
        else:
            alert_counts[alert_type] = 1
    
    # Format summary
    summary = f"Total alerts: {len(alerts)}\n"
    for alert_type, count in alert_counts.items():
        summary += f"- {alert_type}: {count}\n"
    
    return summary

# -------------------
# UI COMPONENTS
# -------------------

def display_sidebar():
    """Display sidebar navigation and controls"""
    st.sidebar.title("🛡️ Intrusion Detection")
    
    # Sidebar navigation
    navigation = st.sidebar.radio(
        "Navigation",
        ["About", "Dataset", "Exploration", "Model Training", "Evaluation", "Live Detection", "Alerts", "Settings"]
    )
    
    # Add sidebar status indicators if data and model exist
    if st.session_state.data is not None:
        data_status = "✅ Dataset Loaded"
        data_info = f"({st.session_state.data.shape[0]} records, {len(st.session_state.data['class'].unique())} classes)"
        st.sidebar.success(f"{data_status} {data_info}")
    else:
        st.sidebar.warning("❌ No Dataset Loaded")
    
    if st.session_state.model is not None:
        # Get the model type name
        if hasattr(st.session_state.model, 'named_steps') and 'model' in st.session_state.model.named_steps:
            model_type = type(st.session_state.model.named_steps['model']).__name__
        else:
            model_type = type(st.session_state.model).__name__
        
        # Get performance if available
        if st.session_state.model_metrics:
            accuracy = st.session_state.model_metrics.get('accuracy', 0)
            model_info = f"(Type: {model_type}, Accuracy: {accuracy:.2f})"
        else:
            model_info = f"(Type: {model_type})"
        
        st.sidebar.success(f"✅ Model Trained {model_info}")
    else:
        st.sidebar.warning("❌ No Model Trained")
    
    # Alert count display if we have any
    if st.session_state.alerts:
        new_alerts = sum(1 for alert in st.session_state.alerts if alert['status'] == 'New')
        if new_alerts > 0:
            st.sidebar.warning(f"⚠️ {new_alerts} New Alert{'s' if new_alerts > 1 else ''}")
    
    # System settings
    with st.sidebar.expander("System Settings"):
        # Theme selection
        theme = st.selectbox(
            "UI Theme",
            ["Light", "Dark", "Auto"],
            index=2
        )
        
        # Advanced options toggle
        advanced_mode = st.checkbox("Advanced Mode", value=False)
        if advanced_mode:
            st.session_state.advanced_mode = True
        else:
            st.session_state.advanced_mode = False
        
        # Clear current session
        if st.button("Clear Session Data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.experimental_rerun()
    
    # About/help
    with st.sidebar.expander("Help"):
        st.markdown("""
        **Quick Guide:**
        1. Load or generate a dataset
        2. Explore the data
        3. Train a model
        4. Evaluate performance
        5. Use model for detection
        
        For more help, see the About section.
        """)
    
    return navigation

def display_about_page():
    """Display the about page with information about the application"""
    st.title("🛡️ AI Intrusion Detection System")
    
    # Main info sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## About This Application
        
        This is an enhanced network intrusion detection system that uses machine learning to identify potentially malicious network traffic. The system can analyze network traffic patterns and classify them as normal or as specific types of attacks.
        
        ### Key Features:
        
        - **Multi-class detection**: Identifies various attack types (DoS, Probe, R2L, U2R)
        - **Interactive visualization**: Explore data and model performance visually
        - **Multiple ML models**: Choose from various algorithms for optimal detection
        - **Real-time evaluation**: Test detection with custom input data
        - **Alert management**: Track and respond to detected intrusions
        """)
    
    with col2:
        # Display some stats/key metrics
        st.markdown("### System Status")
        
        # Dataset status
        if st.session_state.data is not None:
            st.success(f"Dataset: {st.session_state.data.shape[0]} records loaded")
        else:
            st.warning("No dataset loaded")
        
        # Model status
        if st.session_state.model is not None:
            if st.session_state.model_metrics:
                accuracy = st.session_state.model_metrics.get('accuracy', 0)
                st.success(f"Model: Trained (Accuracy: {accuracy:.2f})")
            else:
                st.success("Model: Trained")
        else:
            st.warning("No model trained")
        
        # Alert status
        if st.session_state.alerts:
            alert_count = len(st.session_state.alerts)
            st.info(f"Alerts: {alert_count} recorded")
        else:
            st.info("Alerts: None recorded")
    
    # About IDS
    with st.expander("What is an Intrusion Detection System?"):
        st.markdown("""
        An **Intrusion Detection System (IDS)** is a security technology that monitors network traffic for suspicious activity and policy violations. It watches for malicious activities or security policy violations and produces reports to a management console.
        
        ### Types of IDS:
        
        - **Network-based (NIDS)**: Monitors network traffic for suspicious activity by analyzing protocol activity
        - **Host-based (HIDS)**: Monitors the characteristics of a single host for suspicious activity
        - **Signature-based**: Compares known threat signatures to observed events to identify intrusions
        - **Anomaly-based**: Creates a baseline of normal activity and flags deviations from this baseline
        
        ### Common Network Attacks:
        
        - **Denial of Service (DoS)**: Floods systems, servers, or networks with traffic to exhaust resources
        - **Probe/Scan**: Attempts to gather information about network resources for future attacks
        - **Remote to Local (R2L)**: Unauthorized access from a remote machine to a local machine
        - **User to Root (U2R)**: Attempts to gain root/admin privileges by a normal user
        """)
    
    # About AI in IDS
    with st.expander("How AI Improves Intrusion Detection"):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Benefits of AI in Intrusion Detection:
            
            - **Pattern Recognition**: AI can identify complex patterns that would be difficult for humans to detect
            - **Adaptability**: Machine learning models can adapt to new types of threats
            - **Reduced False Positives**: Advanced algorithms can better distinguish between normal and malicious activity
            - **Automated Response**: AI systems can automatically respond to certain types of attacks
            - **Scalability**: Handle large volumes of network traffic data efficiently
            
            ### Machine Learning for IDS:
            
            Different algorithms are suitable for different aspects of intrusion detection:
            
            - **Random Forests**: Good for identifying known attack patterns
            - **Neural Networks**: Excel at finding complex patterns in network behavior
            - **Support Vector Machines**: Effective for anomaly detection
            - **Gradient Boosting**: Strong performance in classifying attack types
            """)
        
        with col2:
            # Add a simple diagram or illustration
            st.image("https://via.placeholder.com/400x300?text=AI+IDS+Diagram", 
                    caption="Simplified AI-based IDS Architecture")
    
    # App usage instructions
    with st.expander("How to Use This Application"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Dataset Section**:
            - Generate synthetic data or upload your own dataset
            - Review basic statistics of the data
        
        2. **Exploration Section**:
            - Analyze the dataset with visualizations
            - Understand feature distributions and correlations
        
        3. **Model Training Section**:
            - Select a machine learning algorithm
            - Configure model parameters
            - Train the model on your dataset
        
        4. **Evaluation Section**:
            - Review model performance metrics
            - Examine confusion matrix and ROC curves
            - Identify strengths and weaknesses of the model
        
        5. **Live Detection Section**:
            - Test the model with sample or custom data
            - See detailed explanations of predictions
        
        6. **Alerts Section**:
            - View all detected intrusions
            - Manage and respond to alerts
        """)
    
    # Footer with info
    st.markdown("---")
    st.markdown("""
    **Note**: This application is for educational and demonstration purposes. For production use, additional security measures and optimizations would be necessary.
    
    Created with Streamlit and scikit-learn. Enhanced version with improved UI/UX, expanded functionality, and better code organization.
    """)

def display_dataset_page():
    """Display dataset generation, upload and basic exploration page"""
    st.title("Dataset Management")
    
    # Create tabs for different dataset options
    dataset_tab = st.tabs(["Generate Data", "Upload Data", "Dataset Info"])
    
    # Tab 1: Generate synthetic data
    with dataset_tab[0]:
        st.header("Generate Synthetic Network Data")
        
        # Data generation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.slider("Number of samples:", 100, 10000, 1000, 100)
            n_classes = st.slider("Number of classes:", 2, 5, 2, 1)
            
            # Show class names based on selected number
            class_names = {
                2: ["Normal", "Attack"],
                3: ["Normal", "DOS", "Probe"],
                4: ["Normal", "DOS", "Probe", "R2L"],
                5: ["Normal", "DOS", "Probe", "R2L", "U2R"]
            }
            
            st.write(f"Classes: {', '.join(class_names[n_classes])}")
        
        with col2:
            class_imbalance = st.slider(
                "Attack ratio:", 0.1, 0.5, 0.2, 0.05, 
                help="Proportion of attack samples (higher = more balanced dataset)"
            )
            
            random_seed = st.number_input("Random seed:", 0, 999, 42)
            
            advanced_params = st.checkbox("Show advanced parameters")
            if advanced_params:
                # Additional parameters could be added here
                st.warning("Advanced parameters not yet implemented")
        
        # Generate data button
        if st.button("Generate Data", key="gen_data_btn"):
            with st.spinner("Generating data..."):
                try:
                    df = generate_synthetic_data(
                        n_samples=n_samples, 
                        n_classes=n_classes,
                        class_imbalance=class_imbalance,
                        random_seed=random_seed
                    )
                    
                    st.session_state.data = df
                    
                    # Identify categorical and numerical columns
                    categorical_cols = ['protocol_type', 'service', 'flag']
                    numerical_cols = [col for col in df.columns if col not in categorical_cols 
                                     and col != 'class' and col != 'timestamp'
                                     and col != 'src_ip' and col != 'dst_ip']
                    
                    st.session_state.categorical_cols = categorical_cols
                    st.session_state.numerical_cols = numerical_cols
                    
                    st.success(f"Generated {n_samples} samples with {n_classes} classes!")
                    
                    # Show preview of the data
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    # Download options
                    st.subheader("Download Dataset")
                    filename = st.text_input("Filename:", "intrusion_dataset.csv")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save Locally"):
                            success, message = save_dataset(df, filename)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    
                    with col2:
                        # Add download link
                        st.markdown(
                            get_download_link(df, filename, "Download CSV"),
                            unsafe_allow_html=True
                        )
                    
                except Exception as e:
                    st.error(f"Error generating data: {str(e)}")
        
        # Show guidance
        st.markdown("""
        **About Synthetic Data:**
        
        This generator creates realistic network traffic data, including:
        - Normal traffic with various protocols and services
        - Different attack types with realistic signatures
        - Meaningful relationships between features
        
        For production use, consider using real network traffic data with proper labeling.
        """)
    
    # Tab 2: Upload existing dataset
    with dataset_tab[1]:
        st.header("Upload Your Own Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file:",
            type=["csv", "xlsx", "xls"],
            help="File should contain network traffic data with at least 'protocol_type', 'service', 'flag', and 'class' columns"
        )
        
        if uploaded_file is not None:
            # Load the dataset
            df, message = load_dataset(uploaded_file)
            
            if df is not None:
                st.success(message)
                st.session_state.data = df
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Show basic stats
                st.subheader("Dataset Summary")
                st.write(f"Total records: {df.shape[0]}")
                st.write(f"Features: {df.shape[1]}")
                
                # Class distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                df['class'].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)
            else:
                st.error(message)
        
        # Template download option
        st.markdown("### Need a template?")
        if st.button("Download Template CSV"):
            # Create a simple template dataframe
            template_df = pd.DataFrame({
                'duration': [2.0, 0.5, 10.0],
                'protocol_type': ['tcp', 'icmp', 'udp'],
                'service': ['http', 'dns', 'ftp'],
                'flag': ['SF', 'S0', 'REJ'],
                'src_bytes': [1250, 50, 300],
                'dst_bytes': [8750, 0, 12000],
                'land': [0.0, 0.0, 0.0],
                'wrong_fragment': [0.0, 0.0, 0.0],
                'urgent': [0.0, 0.0, 0.0],
                'count': [4, 80, 1],
                'class': ['Normal', 'DOS', 'Probe']
            })
            
            # Add download link
            st.markdown(
                get_download_link(template_df, "template.csv", "Download Template CSV"),
                unsafe_allow_html=True
            )
            
            # Show required format info
            st.info("""
            **Required Format:**
            
            The template includes the minimum required columns. Your dataset should have:
            - Categorical columns: 'protocol_type', 'service', 'flag'
            - Target column: 'class' (with attack type labels)
            - Numerical features like 'duration', 'src_bytes', etc.
            """)
    
    # Tab 3: Dataset Information
    with dataset_tab[2]:
        st.header("Current Dataset Information")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Basic info
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", f"{df.shape[0]:,}")
            with col2:
                st.metric("Features", f"{df.shape[1]-1:,}")  # Excluding class column
            with col3:
                st.metric("Classes", f"{df['class'].nunique():,}")
            
            # Class distribution
            st.subheader("Class Distribution")
            class_counts = df['class'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            class_counts['Percentage'] = (class_counts['Count'] / df.shape[0] * 100).round(2)
            
            # Display as table and chart
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(class_counts)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = sns.barplot(x='Class', y='Count', data=class_counts, ax=ax)
                ax.set_title('Class Distribution')
                
                # Add count labels on top of the bars
                for i, bar in enumerate(bars.patches):
                    bars.annotate(format(bar.get_height(), '.0f'),
                                (bar.get_x() + bar.get_width() / 2,
                                 bar.get_height()), ha='center', va='center',
                                 size=10, xytext=(0, 8),
                                 textcoords='offset points')
                
                st.pyplot(fig)
            
            # Feature information
            st.subheader("Feature Information")
            
            # Get categorical and numerical columns
            categorical_cols = st.session_state.categorical_cols
            numerical_cols = st.session_state.numerical_cols
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Categorical Features:")
                for col in categorical_cols:
                    st.write(f"- {col} ({df[col].nunique()} unique values)")
            
            with col2:
                st.write("Numerical Features:")
                for col in numerical_cols:
                    st.write(f"- {col} (range: {df[col].min():.2f} to {df[col].max():.2f})")
            
            # Preview of data
            with st.expander("Data Preview"):
                st.dataframe(df.head(10))
            
            # Data types and statistics
            with st.expander("Dataset Statistics"):
                st.write("Data Types:")
                st.write(df.dtypes)
                
                st.write("Numerical Statistics:")
                st.write(df[numerical_cols].describe())
            
            # Missing values check
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.warning("Missing Values Detected:")
                st.write(missing_values[missing_values > 0])
            else:
                st.success("No missing values in the dataset")
            
            # Options for dataset manipulation
            st.subheader("Dataset Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Download Dataset"):
                    st.markdown(
                        get_download_link(df, "intrusion_dataset.csv", "Download CSV"),
                        unsafe_allow_html=True
                    )
            
            with col2:
                if st.button("Clear Dataset"):
                    st.session_state.data = None
                    st.experimental_rerun()
            
            with col3:
                if st.button("Sample Dataset"):
                    sample_size = min(1000, df.shape[0])
                    st.session_state.data = df.sample(sample_size, random_state=42)
                    st.experimental_rerun()
        else:
            st.warning("No dataset loaded. Please generate or upload a dataset first.")
            
            # Quick actions
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Sample Dataset"):
                    with st.spinner("Generating sample data..."):
                        df = generate_synthetic_data(n_samples=500, n_classes=3)
                        st.session_state.data = df
                        
                        # Identify categorical and numerical columns
                        categorical_cols = ['protocol_type', 'service', 'flag']
                        numerical_cols = [col for col in df.columns if col not in categorical_cols 
                                         and col != 'class' and col != 'timestamp'
                                         and col != 'src_ip' and col != 'dst_ip']
                        
                        st.session_state.categorical_cols = categorical_cols
                        st.session_state.numerical_cols = numerical_cols
                        
                        st.experimental_rerun()
            
            with col2:
                st.write("Or upload a dataset in the 'Upload Data' tab")

def display_exploration_page():
    """Display data exploration and visualization page"""
    st.title("Data Exploration & Analysis")
    
    if st.session_state.data is None:
        st.warning("Please load a dataset first.")
        if st.button("Go to Dataset Page"):
            st.session_state.page = "Dataset"
            st.experimental_rerun()
        return
    
    # Get the dataset
    df = st.session_state.data
    
    # Perform EDA
    perform_eda(df)
    
    # Advanced Analysis
    with st.expander("Advanced Analysis"):
        st.subheader("Statistical Tests & Advanced Metrics")
        
        # Detect outliers in numerical features
        st.write("#### Outlier Detection")
        
        numerical_cols = st.session_state.numerical_cols
        selected_col = st.selectbox("Select feature for outlier analysis:", numerical_cols)
        
        if selected_col:
            # Calculate IQR
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            
            # Display outlier information
            st.write(f"Feature: {selected_col}")
            st.write(f"IQR: {IQR:.4f} (Q1: {Q1:.4f}, Q3: {Q3:.4f})")
            st.write(f"Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
            st.write(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}% of data)")
            
            # Plot histogram with outlier bounds
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[selected_col], bins=30, kde=True, ax=ax)
            ax.axvline(lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
            ax.axvline(upper_bound, color='r', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
            ax.set_title(f'Distribution of {selected_col} with Outlier Bounds')
            ax.legend()
            st.pyplot(fig)
            
            # Show outliers by class
            if len(outliers) > 0:
                st.write("#### Outliers by Class")
                class_counts = outliers['class'].value_counts().reset_index()
                class_counts.columns = ['Class', 'Count']
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Class', y='Count', data=class_counts, ax=ax)
                ax.set_title(f'Distribution of Outliers in {selected_col} by Class')
                st.pyplot(fig)
        
        # Feature relationships
        st.write("#### Feature Relationships")
        
        # Let user select features for scatter plot
        feature_x = st.selectbox("X-axis feature:", numerical_cols, key='feat_x')
        feature_y = st.selectbox("Y-axis feature:", 
                                [f for f in numerical_cols if f != feature_x], 
                                key='feat_y')
        
        if feature_x and feature_y:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='class', alpha=0.7, ax=ax)
            ax.set_title(f'Relationship between {feature_x} and {feature_y}')
            
            # Add best fit line for normal traffic
            normal_df = df[df['class'] == 'Normal']
            if len(normal_df) > 1:  # Need at least 2 points for regression
                sns.regplot(data=normal_df, x=feature_x, y=feature_y, 
                           scatter=False, ax=ax, line_kws={"color": "blue", "alpha": 0.5})
            
            # Adjust legend if too many classes
            if df['class'].nunique() > 5:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            st.pyplot(fig)
    
    # Export analysis report
    st.subheader("Export Analysis Report")
    if st.button("Generate Analysis Report"):
        # Create a BytesIO object to write to
        buffer = BytesIO()
        
        # Create a simple HTML report
        html_content = f"""
        <html>
        <head>
            <title>Network Traffic Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Network Traffic Analysis Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Dataset Summary</h2>
                <p>Total Records: {df.shape[0]}</p>
                <p>Features: {df.shape[1]}</p>
                <p>Classes: {df['class'].nunique()}</p>
            </div>
            
            <h2>Class Distribution</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        # Add class distribution
        for idx, row in df['class'].value_counts().reset_index().iterrows():
            html_content += f"""
                <tr>
                    <td>{row['index']}</td>
                    <td>{row['class']}</td>
                    <td>{row['class'] / len(df) * 100:.2f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Feature Statistics</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                </tr>
        """
        
        # Add feature statistics
        numerical_cols = st.session_state.numerical_cols
        for col in numerical_cols:
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{df[col].min():.2f}</td>
                    <td>{df[col].max():.2f}</td>
                    <td>{df[col].mean():.2f}</td>
                    <td>{df[col].std():.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Correlation Analysis</h2>
            <p>Notable correlations between features:</p>
            <ul>
        """
        
        # Add top correlations
        corr = df[numerical_cols].corr().abs().unstack().sort_values(ascending=False)
        corr = corr[corr < 1].reset_index()  # Remove self-correlations
        corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
        
        for idx, row in corr.head(5).iterrows():
            html_content += f"""
                <li>{row['Feature 1']} and {row['Feature 2']}: {row['Correlation']:.2f}</li>
            """
        
        html_content += """
            </ul>
            
            <h2>Conclusion</h2>
            <p>This report provides a basic analysis of network traffic data. Further investigation may be required for specific patterns or anomalies.</p>
        </body>
        </html>
        """
        
        # Encode the HTML content and create a download link
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="traffic_analysis_report.html" class="download-button">Download Analysis Report</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_model_training_page():
    """Display model training page with options for different algorithms"""
    st.title("Model Training")
    
    if st.session_state.data is None:
        st.warning("Please load a dataset first.")
        if st.button("Go to Dataset Page"):
            st.session_state.page = "Dataset"
            st.experimental_rerun()
        return
    
    # Get the dataset
    df = st.session_state.data
    
    # Display training options in tabs
    training_tabs = st.tabs(["Basic Training", "Advanced Options", "Training Results"])
    
    # Tab 1: Basic Training
    with training_tabs[0]:
        st.header("Train Intrusion Detection Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            model_type = st.selectbox(
                "Select Model Type:",
                ["Random Forest", "Logistic Regression", "Gradient Boosting", 
                 "Support Vector Machine", "Neural Network"]
            )
            
            # Test size selection
            test_size = st.slider("Test Set Size:", 0.1, 0.5, 0.2, 0.05)
            
            # Simple explanation of selected model
            model_descriptions = {
                "Random Forest": """
                **Random Forest** creates multiple decision trees and merges their predictions. 
                It's good for handling complex data with many features and can identify important features automatically.
                """,
                "Logistic Regression": """
                **Logistic Regression** is a simple but effective algorithm for classification.
                It works well for linearly separable data and provides easily interpretable results.
                """,
                "Gradient Boosting": """
                **Gradient Boosting** builds trees sequentially, with each tree correcting the errors of previous ones.
                It often achieves high accuracy but may be prone to overfitting.
                """,
                "Support Vector Machine": """
                **Support Vector Machine** finds the optimal boundary between classes.
                It works well with clear margins of separation and is effective in high-dimensional spaces.
                """,
                "Neural Network": """
                **Neural Network** can learn complex patterns in data through multiple layers.
                It's powerful but may require more data and tuning to perform optimally.
                """
            }
            
            st.info(model_descriptions[model_type])
        
        with col2:
            # Model-specific parameters
            st.subheader("Model Parameters")
            
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of trees:", 10, 500, 100, 10)
                max_depth = st.slider("Maximum tree depth:", 2, 30, 10, 1)
                
                hyperparameters = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }
            
            elif model_type == "Logistic Regression":
                C = st.slider("Regularization strength:", 0.01, 10.0, 1.0, 0.01)
                max_iter = st.slider("Maximum iterations:", 100, 2000, 1000, 100)
                
                hyperparameters = {
                    'C': C,
                    'max_iter': max_iter
                }
            
            elif model_type == "Gradient Boosting":
                n_estimators = st.slider("Number of boosting stages:", 10, 500, 100, 10)
                learning_rate = st.slider("Learning rate:", 0.01, 0.5, 0.1, 0.01)
                max_depth = st.slider("Maximum tree depth:", 2, 10, 3, 1)
                
                hyperparameters = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth
                }
            
            elif model_type == "Support Vector Machine":
                C = st.slider("Regularization parameter:", 0.1, 10.0, 1.0, 0.1)
                kernel = st.selectbox("Kernel type:", ["rbf", "linear", "poly", "sigmoid"])
                
                hyperparameters = {
                    'C': C,
                    'kernel': kernel
                }
            
            elif model_type == "Neural Network":
                hidden_layer_sizes = st.selectbox(
                    "Hidden layer structure:", 
                    [(100,), (50, 50), (100, 50), (100, 100), (200, 100, 50)]
                )
                alpha = st.slider("L2 regularization:", 0.0001, 0.01, 0.0001, 0.0001)
                
                hyperparameters = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'alpha': alpha,
                    'max_iter': 1000
                }
        
        # Training button
        if st.button("Train Model"):
            with st.spinner("Preprocessing data and training model..."):
                try:
                    # Prepare data
                    X_train, X_test, y_train, y_test, preprocessing_pipeline = prepare_data_for_training(
                        df, test_size=test_size, random_state=42
                    )
                    
                    # Create and train model
                    model = create_model(model_type, hyperparameters)
                    
                    # Train model
                    full_pipeline, metrics, y_pred, y_proba = train_model_with_pipeline(
                        preprocessing_pipeline, model, X_train, y_train, X_test, y_test
                    )
                    
                    # Store results in session state
                    st.session_state.model = full_pipeline
                    st.session_state.model_metrics = metrics
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    # Success message
                    st.success(f"Model trained successfully! Accuracy: {metrics['accuracy']:.4f}")
                    
                    # Basic performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")
                    
                    # Show confusion matrix
                    st.subheader("Confusion Matrix")
                    
                    # Get class names
                    class_names = st.session_state.class_names
                    
                    # Calculate confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=class_names, yticklabels=class_names, ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    st.exception(e)
    
    # Tab 2: Advanced Options
    with training_tabs[1]:
        st.header("Advanced Training Options")
        
        # Feature selection
        st.subheader("Feature Selection")
        feature_selection = st.checkbox("Enable feature selection")
        
        if feature_selection:
            numerical_cols = st.session_state.numerical_cols
            categorical_cols = st.session_state.categorical_cols
            
            # Total number of features after one-hot encoding
            total_features = len(numerical_cols)
            if categorical_cols:
                # Count unique values in each categorical column
                for col in categorical_cols:
                    if df is not None:
                        total_features += df[col].nunique()
            
            k_features = st.slider(
                "Number of features to select:",
                min_value=1,
                max_value=total_features,
                value=min(10, total_features),
                help="Select top K features based on ANOVA F-value"
            )
            
            st.session_state.feature_selection = feature_selection
            st.session_state.k_features = k_features
        else:
            st.session_state.feature_selection = False
            st.session_state.k_features = None
        
        # Cross-validation options
        st.subheader("Cross-Validation")
        use_cv = st.checkbox("Use cross-validation")
        
        if use_cv:
            cv_folds = st.slider("Number of folds:", 2, 10, 5, 1)
            st.session_state.use_cv = True
            st.session_state.cv_folds = cv_folds
        else:
            st.session_state.use_cv = False
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Tuning")
        use_hyperparam_tuning = st.checkbox("Tune hyperparameters")
        
        if use_hyperparam_tuning:
            st.info("This will use Grid Search to find optimal parameters. It may take some time.")
            
            # Model selection for tuning
            tuning_model = st.selectbox(
                "Select model for tuning:",
                ["Random Forest", "Logistic Regression", "Gradient Boosting", "Support Vector Machine"]
            )
            
            # Define parameter grids based on model type
            if tuning_model == "Random Forest":
                param_grid = {
                    'n_estimators': st.multiselect("Number of trees:", [50, 100, 200, 300], default=[100, 200]),
                    'max_depth': st.multiselect("Maximum depth:", [None, 10, 20, 30], default=[None, 10]),
                    'min_samples_split': st.multiselect("Min samples to split:", [2, 5, 10], default=[2])
                }
            
            elif tuning_model == "Logistic Regression":
                param_grid = {
                    'C': st.multiselect("Regularization strength:", [0.01, 0.1, 1.0, 10.0], default=[0.1, 1.0]),
                    'solver': st.multiselect("Solver:", ['liblinear', 'lbfgs', 'newton-cg'], default=['liblinear'])
                }
            
            elif tuning_model == "Gradient Boosting":
                param_grid = {
                    'n_estimators': st.multiselect("Number of estimators:", [50, 100, 200], default=[100]),
                    'learning_rate': st.multiselect("Learning rate:", [0.01, 0.1, 0.2], default=[0.1]),
                    'max_depth': st.multiselect("Maximum depth:", [3, 5, 7], default=[3])
                }
            
            elif tuning_model == "Support Vector Machine":
                param_grid = {
                    'C': st.multiselect("Regularization parameter:", [0.1, 1.0, 10.0], default=[1.0]),
                    'kernel': st.multiselect("Kernel:", ['linear', 'rbf', 'poly'], default=['rbf'])
                }
            
            # Check if parameter grid is not empty (all selections have defaults)
            all_params_selected = all(len(v) > 0 for v in param_grid.values())
            
            if all_params_selected:
                # Button to start hyperparameter tuning
                if st.button("Start Hyperparameter Tuning"):
                    with st.spinner("Tuning hyperparameters... This may take a while."):
                        try:
                            # Prepare data
                            X_train, X_test, y_train, y_test, preprocessing_pipeline = prepare_data_for_training(
                                df, test_size=0.2, random_state=42
                            )
                            
                            # Tune hyperparameters
                            best_model, best_params, best_score = tune_hyperparameters(
                                preprocessing_pipeline, tuning_model, X_train, y_train, param_grid
                            )
                            
                            # Train model with best parameters
                            full_pipeline, metrics, y_pred, y_proba = train_model_with_pipeline(
                                preprocessing_pipeline, best_model, X_train, y_train, X_test, y_test
                            )
                            
                            # Store results in session state
                            st.session_state.model = full_pipeline
                            st.session_state.model_metrics = metrics
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.best_params = best_params
                            
                            # Success message
                            st.success(f"Hyperparameter tuning completed! Best score: {best_score:.4f}")
                            
                            # Show best parameters
                            st.subheader("Best Parameters")
                            st.json(best_params)
                            
                            # Show performance metrics
                            st.subheader("Model Performance")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            with col2:
                                st.metric("Precision", f"{metrics['precision']:.4f}")
                            with col3:
                                st.metric("Recall", f"{metrics['recall']:.4f}")
                            with col4:
                                st.metric("F1 Score", f"{metrics['f1']:.4f}")
                            
                        except Exception as e:
                            st.error(f"Error during hyperparameter tuning: {str(e)}")
                            st.exception(e)
            else:
                st.warning("Please select at least one value for each parameter.")
        
        # Class imbalance handling
        st.subheader("Class Imbalance Handling")
        handle_imbalance = st.checkbox("Handle class imbalance")
        
        if handle_imbalance:
            imbalance_method = st.selectbox(
                "Imbalance handling method:",
                ["Class weights", "Oversampling (SMOTE)", "Undersampling"]
            )
            
            # Add information about each method
            if imbalance_method == "Class weights":
                st.info("Assigns higher weights to minority classes during training.")
            elif imbalance_method == "Oversampling (SMOTE)":
                st.info("Generates synthetic samples for minority classes.")
                st.warning("SMOTE implementation requires the 'imbalanced-learn' package, which is not included in this demo.")
            elif imbalance_method == "Undersampling":
                st.info("Reduces the number of samples from majority classes.")
                st.warning("This may discard valuable information from majority classes.")
            
            st.session_state.handle_imbalance = True
            st.session_state.imbalance_method = imbalance_method
        else:
            st.session_state.handle_imbalance = False
    
    # Tab 3: Training Results
    with training_tabs[2]:
        st.header("Training Results History")
        
        if not st.session_state.training_history:
            st.info("No training history available. Train a model first.")
        else:
            # Display training history
            history = st.session_state.training_history
            
            # Create a summary table
            summary_data = []
            for i, record in enumerate(history):
                summary_data.append({
                    'Run': i + 1,
                    'Timestamp': record['timestamp'],
                    'Model': record['model_type'],
                    'Accuracy': record['metrics']['accuracy'],
                    'F1 Score': record['metrics']['f1'],
                    'Training Time': f"{record['metrics']['training_time']:.2f}s"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            # Plot performance comparison
            st.subheader("Performance Comparison")
            
            # Prepare data for plotting
            model_names = [record['model_type'] for record in history]
            accuracies = [record['metrics']['accuracy'] for record in history]
            f1_scores = [record['metrics']['f1'] for record in history]
            precisions = [record['metrics']['precision'] for record in history]
            recalls = [record['metrics']['recall'] for record in history]
            
            # Create plot data
            plot_data = pd.DataFrame({
                'Model': model_names,
                'Run': [f"Run {i+1}" for i in range(len(model_names))],
                'Accuracy': accuracies,
                'F1 Score': f1_scores,
                'Precision': precisions,
                'Recall': recalls
            })
            
            # Create plot
            metric_to_plot = st.selectbox(
                "Select metric to visualize:",
                ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Run', y=metric_to_plot, hue='Model', data=plot_data, ax=ax)
            ax.set_title(f'{metric_to_plot} Comparison Across Runs')
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            
            # Option to view detailed record
            st.subheader("Detailed Run Information")
            run_to_view = st.selectbox(
                "Select run to view details:",
                [f"Run {i+1}" for i in range(len(history))]
            )
            
            if run_to_view:
                run_index = int(run_to_view.split(" ")[1]) - 1
                record = history[run_index]
                
                # Display detailed information
                st.write(f"**Model Type:** {record['model_type']}")
                st.write(f"**Training Timestamp:** {record['timestamp']}")
                
                # Metrics
                st.write("**Performance Metrics:**")
                metrics_df = pd.DataFrame({
                    'Metric': list(record['metrics'].keys()),
                    'Value': list(record['metrics'].values())
                })
                st.dataframe(metrics_df)
                
                # Parameters
                st.write("**Model Parameters:**")
                st.json(record['parameters'])
            
            # Option to export training history
            if st.button("Export Training History"):
                # Convert history to DataFrame
                export_data = []
                for i, record in enumerate(history):
                    export_record = {
                        'Run': i + 1,
                        'Timestamp': record['timestamp'],
                        'Model': record['model_type']
                    }
                    
                    # Add metrics
                    for metric, value in record['metrics'].items():
                        export_record[f'Metric_{metric}'] = value
                    
                    # Add parameters (flattened)
                    for param, value in record['parameters'].items():
                        export_record[f'Param_{param}'] = str(value)
                    
                    export_data.append(export_record)
                
                export_df = pd.DataFrame(export_data)
                
                # Create download link
                st.markdown(
                    get_download_link(export_df, "training_history.csv", "Download CSV"),
                    unsafe_allow_html=True
                )

def display_evaluation_page():
    """Display model evaluation and performance analysis page"""
    st.title("Model Evaluation & Analysis")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
        if st.button("Go to Model Training Page"):
            st.session_state.page = "Model Training"
            st.experimental_rerun()
        return
    
    # Get model and test data
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    class_names = st.session_state.class_names
    
    # Create tabs for different evaluation aspects
    eval_tabs = st.tabs(["Performance Metrics", "ROC & PR Curves", "Feature Analysis", "Error Analysis"])
    
    # Tab 1: Performance Metrics
    with eval_tabs[0]:
        st.header("Model Performance Metrics")
        
        # Display overall metrics
        metrics = st.session_state.model_metrics
        
        # Create metrics cards in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")
        
        # Additional metrics and description
        st.markdown("""
        **Understanding these metrics:**
        
        - **Accuracy**: Overall correctness of the model (correct predictions / total predictions)
        - **Precision**: Ability to avoid false positives (true positives / (true positives + false positives))
        - **Recall**: Ability to find all positive cases (true positives / (true positives + false negatives))
        - **F1 Score**: Harmonic mean of precision and recall, balancing both metrics
        """)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        # Make predictions on test data
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        
        # Calculate classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        
        # Round values for display
        report_df = report_df.round(3)
        
        # Display the report
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        # Per-class metrics visualization
        st.subheader("Per-Class Performance")
        
        # Extract per-class metrics
        per_class_metrics = pd.DataFrame({
            'Class': class_names,
            'Precision': [report[c]['precision'] for c in class_names],
            'Recall': [report[c]['recall'] for c in class_names],
            'F1-Score': [report[c]['f1-score'] for c in class_names],
            'Support': [report[c]['support'] for c in class_names]
        })
        
        # Create a bar chart for per-class metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Melt the DataFrame for easier plotting with seaborn
        plot_data = per_class_metrics.melt(
            id_vars=['Class', 'Support'],
            value_vars=['Precision', 'Recall', 'F1-Score'],
            var_name='Metric',
            value_name='Value'
        )
        
        # Create the bar chart
        sns.barplot(x='Class', y='Value', hue='Metric', data=plot_data, ax=ax)
        ax.set_title('Performance Metrics by Class')
        ax.set_ylim(0, 1)
        
        # Annotate with support counts
        for i, cls in enumerate(class_names):
            support = per_class_metrics.loc[per_class_metrics['Class'] == cls, 'Support'].values[0]
            ax.text(i, 0.05, f'n={support}', ha='center', color='white', fontweight='bold')
        
        st.pyplot(fig)
    
    # Tab 2: ROC & PR Curves
    with eval_tabs[1]:
        st.header("ROC and Precision-Recall Curves")
        
        # Generate predictions and probabilities
        y_pred = model.predict(X_test)
        
        # Check if model can provide probability estimates
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            
            # Create ROC curves for each class
            st.subheader("ROC Curves")
            
            # One-hot encode true labels for ROC curve calculation
            n_classes = len(class_names)
            y_test_bin = np.zeros((len(y_test), n_classes))
            for i in range(len(y_test)):
                y_test_bin[i, y_test[i]] = 1
            
            # Calculate ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Add each class's ROC curve
            for i in range(n_classes):
                plt.plot(
                    fpr[i], tpr[i],
                    lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
                )
            
            # Add diagonal line for reference (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
            # Set plot details
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            
            # Display the plot
            st.pyplot(fig)
            
            # Create precision-recall curves for each class
            st.subheader("Precision-Recall Curves")
            
            # Calculate precision-recall curve for each class
            precision = {}
            recall = {}
            pr_auc = {}
            
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                pr_auc[i] = auc(recall[i], precision[i])
            
            # Plot precision-recall curves
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Add each class's PR curve
            for i in range(n_classes):
                plt.plot(
                    recall[i], precision[i],
                    lw=2,
                    label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})'
                )
            
            # Set plot details
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend(loc="best")
            
            # Display the plot
            st.pyplot(fig)
            
            # Threshold analysis
            with st.expander("Threshold Analysis"):
                st.info("""
                Adjusting the classification threshold allows you to balance precision and recall based on your needs:
                - Lower threshold: Higher recall (more attacks detected) but lower precision (more false alarms)
                - Higher threshold: Higher precision (fewer false alarms) but lower recall (might miss attacks)
                """)
                
                # Let user select a class for threshold analysis
                selected_class = st.selectbox(
                    "Select class for threshold analysis:",
                    class_names
                )
                
                if selected_class:
                    class_idx = list(class_names).index(selected_class)
                    
                    # Create a DataFrame with different thresholds
                    thresholds = np.linspace(0, 1, 100)
                    threshold_data = []
                    
                    for threshold in thresholds:
                        # Create binary predictions for this threshold
                        y_pred_bin = (y_proba[:, class_idx] >= threshold).astype(int)
                        
                        # Calculate metrics
                        tn, fp, fn, tp = confusion_matrix(
                            y_test_bin[:, class_idx], y_pred_bin, labels=[0, 1]
                        ).ravel()
                        
                        # Calculate performance metrics
                        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
                        
                        threshold_data.append({
                            'Threshold': threshold,
                            'Precision': precision_val,
                            'Recall': recall_val,
                            'F1-Score': f1,
                            'True Positives': tp,
                            'False Positives': fp,
                            'True Negatives': tn,
                            'False Negatives': fn
                        })
                    
                    # Create DataFrame
                    threshold_df = pd.DataFrame(threshold_data)
                    
                    # Plot threshold vs metrics
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(threshold_df['Threshold'], threshold_df['Precision'], 'b-', label='Precision')
                    ax.plot(threshold_df['Threshold'], threshold_df['Recall'], 'r-', label='Recall')
                    ax.plot(threshold_df['Threshold'], threshold_df['F1-Score'], 'g-', label='F1-Score')
                    
                    ax.set_xlabel('Threshold')
                    ax.set_ylabel('Score')
                    ax.set_title(f'Performance Metrics vs. Threshold for {selected_class}')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                    
                    # Find optimal threshold for F1 score
                    best_idx = threshold_df['F1-Score'].idxmax()
                    best_threshold = threshold_df.loc[best_idx, 'Threshold']
                    best_f1 = threshold_df.loc[best_idx, 'F1-Score']
                    
                    st.info(f"Optimal threshold for maximum F1-Score: {best_threshold:.2f} (F1 = {best_f1:.4f})")
                    
                    # Allow user to select a specific threshold
                    custom_threshold = st.slider(
                        "Select custom threshold:",
                        min_value=0.0,
                        max_value=1.0,
                        value=best_threshold,
                        step=0.01
                    )
                    
                    # Display metrics for selected threshold
                    selected_row = threshold_df.iloc[(threshold_df['Threshold'] - custom_threshold).abs().argsort()[:1]]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{selected_row['Precision'].values[0]:.4f}")
                    with col2:
                        st.metric("Recall", f"{selected_row['Recall'].values[0]:.4f}")
                    with col3:
                        st.metric("F1-Score", f"{selected_row['F1-Score'].values[0]:.4f}")
                    
                    # Confusion matrix at selected threshold
                    st.subheader(f"Confusion Matrix at Threshold = {custom_threshold:.2f}")
                    
                    # Create a mini confusion matrix for this class vs others
                    cm_data = np.array([
                        [selected_row['True Negatives'].values[0], selected_row['False Positives'].values[0]],
                        [selected_row['False Negatives'].values[0], selected_row['True Positives'].values[0]]
                    ])
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(
                        cm_data, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=['Not ' + selected_class, selected_class],
                        yticklabels=['Not ' + selected_class, selected_class],
                        ax=ax
                    )
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    
                    st.pyplot(fig)
        else:
            st.warning("This model doesn't provide probability estimates, so ROC and PR curves cannot be generated.")
    
    # Tab 3: Feature Analysis
    with eval_tabs[2]:
        st.header("Feature Importance Analysis")
        
        # Check if model provides feature importances
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            model_step = model.named_steps['model']
            
            if hasattr(model_step, 'feature_importances_'):
                # Get feature importances
                feature_importance = model_step.feature_importances_
                
                # Get feature names
                if hasattr(model, 'named_steps') and 'preprocessing' in model.named_steps:
                    preprocessor = model.named_steps['preprocessing']
                    
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            # This should work for newer sklearn versions
                            feature_names = preprocessor.get_feature_names_out()
                        except:
                            # Fallback for older versions or if above fails
                            if hasattr(st.session_state, 'feature_names'):
                                # Use original feature names as fallback
                                feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
                            else:
                                feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
                    else:
                        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
                else:
                    feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
                
                # Create a DataFrame with feature importances
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(feature_importance)],
                    'Importance': feature_importance
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Display the top features
                st.subheader("Top Features by Importance")
                
                # Limit to top 20 features for visual clarity
                top_n = min(20, len(importance_df))
                top_features = importance_df.head(top_n)
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create horizontal bar chart
                sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                ax.set_title(f'Top {top_n} Features by Importance')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                
                st.pyplot(fig)
                
                # Display feature importance table
                st.dataframe(importance_df)
                
                # Feature correlation with target
                st.subheader("Feature Correlation with Target")
                
                st.info("""
                This analysis shows how individual features correlate with different classes.
                Strong positive correlations indicate that higher values of a feature are associated with a particular class.
                """)
                
                # We need to go back to the original data for this
                if st.session_state.X_train is not None and st.session_state.y_train is not None:
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    
                    # Check if X_train is DataFrame (not numpy array)
                    if isinstance(X_train, pd.DataFrame):
                        numerical_cols = st.session_state.numerical_cols
                        
                        # For each class, calculate correlation with numerical features
                        corr_data = []
                        
                        for i, cls in enumerate(class_names):
                            # Create binary target for this class
                            y_binary = (y_train == i).astype(int)
                            
                            # Calculate correlation for each numerical feature
                            for col in numerical_cols:
                                if col in X_train.columns:
                                    corr = np.corrcoef(X_train[col], y_binary)[0, 1]
                                    corr_data.append({
                                        'Feature': col,
                                        'Class': cls,
                                        'Correlation': corr
                                    })
                        
                        if corr_data:
                            # Create DataFrame
                            corr_df = pd.DataFrame(corr_data)
                            
                            # Create heatmap of correlations
                            pivot_df = corr_df.pivot(index='Feature', columns='Class', values='Correlation')
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, ax=ax)
                            ax.set_title('Feature Correlation with Classes')
                            
                            st.pyplot(fig)
                        else:
                            st.warning("Could not calculate feature correlations with the available data.")
                    else:
                        st.warning("Feature correlation analysis requires the original DataFrame format.")
            elif hasattr(model_step, 'coef_'):
                # For linear models like Logistic Regression
                coefs = model_step.coef_
                
                # Get feature names (similar to above)
                if hasattr(model, 'named_steps') and 'preprocessing' in model.named_steps:
                    preprocessor = model.named_steps['preprocessing']
                    
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names = preprocessor.get_feature_names_out()
                        except:
                            if hasattr(st.session_state, 'feature_names'):
                                feature_names = [f"Feature {i}" for i in range(coefs.shape[1])]
                            else:
                                feature_names = [f"Feature {i}" for i in range(coefs.shape[1])]
                    else:
                        feature_names = [f"Feature {i}" for i in range(coefs.shape[1])]
                else:
                    feature_names = [f"Feature {i}" for i in range(coefs.shape[1])]
                
                # For multiclass, we have coefficients for each class
                if len(coefs.shape) > 1 and coefs.shape[0] > 1:
                    st.subheader("Feature Coefficients by Class")
                    
                    # Create a DataFrame for each class
                    for i, cls in enumerate(class_names):
                        if i < coefs.shape[0]:  # Make sure we have coefficients for this class
                            # Create DataFrame for this class
                            coef_df = pd.DataFrame({
                                'Feature': feature_names[:len(coefs[i])],
                                'Coefficient': coefs[i]
                            })
                            
                            # Sort by absolute coefficient value
                            coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
                            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)
                            
                            # Display top coefficients for this class
                            st.write(f"#### Class: {cls}")
                            
                            # Plot top coefficients
                            top_n = min(15, len(coef_df))
                            top_coefs = coef_df.head(top_n)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = sns.barplot(x='Coefficient', y='Feature', data=top_coefs, ax=ax)
                            
                            # Color bars based on coefficient sign
                            for bar, coef in zip(bars.patches, top_coefs['Coefficient']):
                                if coef < 0:
                                    bar.set_color('r')
                                else:
                                    bar.set_color('b')
                            
                            ax.set_title(f'Top {top_n} Features for {cls} Class')
                            ax.axvline(x=0, color='k', linestyle='--')
                            
                            st.pyplot(fig)
                else:
                    # Binary classification case
                    coef_df = pd.DataFrame({
                        'Feature': feature_names[:len(coefs[0])],
                        'Coefficient': coefs[0]
                    })
                    
                    # Sort by absolute coefficient value
                    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
                    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)
                    
                    # Display top coefficients
                    st.subheader("Top Features by Coefficient")
                    
                    # Plot top coefficients
                    top_n = min(20, len(coef_df))
                    top_coefs = coef_df.head(top_n)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = sns.barplot(x='Coefficient', y='Feature', data=top_coefs, ax=ax)
                    
                    # Color bars based on coefficient sign
                    for bar, coef in zip(bars.patches, top_coefs['Coefficient']):
                        if coef < 0:
                            bar.set_color('r')
                        else:
                            bar.set_color('b')
                    
                    ax.set_title(f'Top {top_n} Features by Coefficient Magnitude')
                    ax.axvline(x=0, color='k', linestyle='--')
                    
                    st.pyplot(fig)
            else:
                st.info("This model type doesn't provide direct feature importance or coefficient information.")
                
                # If we have a permutation-based feature importance option
                st.subheader("Calculate Permutation Feature Importance")
                
                if st.button("Calculate Permutation Importance (may take time)"):
                    with st.spinner("Calculating permutation feature importance..."):
                        try:
                            from sklearn.inspection import permutation_importance
                            
                            # Calculate permutation importance
                            r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                            
                            # Use feature names if available
                            if hasattr(st.session_state, 'feature_names'):
                                feature_names = st.session_state.feature_names
                            else:
                                feature_names = [f"Feature {i}" for i in range(len(r.importances_mean))]
                            
                            # Create DataFrame
                            perm_importance_df = pd.DataFrame({
                                'Feature': feature_names[:len(r.importances_mean)],
                                'Importance': r.importances_mean,
                                'Std': r.importances_std
                            })
                            
                            # Sort by importance
                            perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)
                            
                            # Plot permutation importance
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Limit to top 20 features
                            top_n = min(20, len(perm_importance_df))
                            top_features = perm_importance_df.head(top_n)
                            
                            # Create horizontal bar chart with error bars
                            bars = sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                            
                            # Add error bars
                            for i, importance in enumerate(top_features['Importance']):
                                std = top_features['Std'].iloc[i]
                                ax.errorbar(importance, i, xerr=std, fmt='k.')
                            
                            ax.set_title('Permutation Feature Importance')
                            ax.set_xlabel('Mean Importance')
                            
                            st.pyplot(fig)
                            
                            # Display table
                            st.dataframe(perm_importance_df)
                        except Exception as e:
                            st.error(f"Error calculating permutation importance: {str(e)}")
        else:
            st.warning("Feature importance analysis not available for this model type.")
    
    # Tab 4: Error Analysis
    with eval_tabs[3]:
        st.header("Error Analysis")
        
        # Get predictions on test data
        y_pred = model.predict(X_test)
        
        # Find misclassified samples
        misclassified_indices = np.where(y_pred != y_test)[0]
        
        if len(misclassified_indices) > 0:
            # Create a DataFrame with misclassified samples
            misclassified_df = pd.DataFrame({
                'Index': misclassified_indices,
                'True Class': [class_names[y_test[i]] for i in misclassified_indices],
                'Predicted Class': [class_names[y_pred[i]] for i in misclassified_indices]
            })
            
            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                
                # Add predicted probability
                misclassified_df['Probability'] = [
                    y_proba[i, y_pred[i]] for i in misclassified_indices
                ]
                
                # Sort by predicted probability (confidence in wrong prediction)
                misclassified_df = misclassified_df.sort_values('Probability', ascending=False)
            
            # Display misclassification summary
            st.subheader("Misclassification Summary")
            
            # Number of misclassifications
            st.write(f"Total misclassified samples: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")
            
            # Misclassification matrix (which classes get confused for which)
            st.write("#### Class Confusion Patterns")
            
            # Create a cross-tabulation of true vs predicted classes for misclassifications
            misclass_crosstab = pd.crosstab(
                misclassified_df['True Class'], 
                misclassified_df['Predicted Class'],
                rownames=['True Class'],
                colnames=['Predicted Class']
            )
            
            # Display as a heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(misclass_crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title('Misclassification Patterns')
            
            st.pyplot(fig)
            
            # Common error cases
            st.write("#### Most Common Errors")
            
            # Count and display most common misclassification types
            error_counts = misclassified_df.groupby(['True Class', 'Predicted Class']).size().reset_index()
            error_counts.columns = ['True Class', 'Predicted Class', 'Count']
            error_counts = error_counts.sort_values('Count', ascending=False)
            
            st.dataframe(error_counts)
            
            # Sample of misclassified instances
            st.write("#### Sample Misclassified Instances")
            
            # Show a sample of misclassified instances
            st.dataframe(misclassified_df.head(10))
            
            # Detailed analysis of most confident mistakes
            st.subheader("Analysis of Most Confident Mistakes")
            
            # Let user select how many top mistakes to view
            n_mistakes = st.slider("Number of mistakes to analyze:", 1, min(10, len(misclassified_df)), 3)
            
            # Get original feature values for misclassified samples if possible
            if isinstance(st.session_state.X_test, pd.DataFrame):
                X_test_df = st.session_state.X_test
                
                for i in range(min(n_mistakes, len(misclassified_df))):
                    idx = misclassified_df['Index'].iloc[i]
                    true_class = misclassified_df['True Class'].iloc[i]
                    pred_class = misclassified_df['Predicted Class'].iloc[i]
                    
                    st.write(f"#### Mistake #{i+1}: {true_class} classified as {pred_class}")
                    
                    if 'Probability' in misclassified_df.columns:
                        prob = misclassified_df['Probability'].iloc[i]
                        st.write(f"Confidence in wrong prediction: {prob:.2f}")
                    
                    # Display feature values for this sample
                    sample_data = X_test_df.iloc[idx].to_frame().T
                    st.dataframe(sample_data)
                    
                    # Plot feature values against class averages if numerical
                    numerical_cols = st.session_state.numerical_cols
                    if numerical_cols:
                        # Select a subset of numerical features for visualization
                        selected_features = numerical_cols[:min(5, len(numerical_cols))]
                        
                        # Create a figure to plot the comparison
                        fig, axs = plt.subplots(len(selected_features), 1, figsize=(10, 3*len(selected_features)))
                        
                        # If only one feature, make axs a list for consistent indexing
                        if len(selected_features) == 1:
                            axs = [axs]
                        
                        for j, feature in enumerate(selected_features):
                            if feature in sample_data.columns:
                                # Calculate average values for each class
                                avg_by_class = {}
                                for k, cls in enumerate(class_names):
                                    class_mask = y_test == k
                                    avg_by_class[cls] = X_test_df.loc[class_mask, feature].mean()
                                
                                # Plot as bar chart
                                ax = axs[j]
                                
                                # Prepare data for plotting
                                bar_data = {
                                    'Class': list(avg_by_class.keys()) + ['This Sample'],
                                    'Value': list(avg_by_class.values()) + [sample_data[feature].values[0]]
                                }
                                
                                # Create DataFrame
                                bar_df = pd.DataFrame(bar_data)
                                
                                # Plot bars
                                bars = sns.barplot(x='Class', y='Value', data=bar_df, ax=ax)
                                
                                # Highlight the sample's value and the true/predicted classes
                                for k, bar in enumerate(bars.patches):
                                    if bar_df['Class'].iloc[k] == 'This Sample':
                                        bar.set_color('gold')
                                    elif bar_df['Class'].iloc[k] == true_class:
                                        bar.set_color('green')
                                    elif bar_df['Class'].iloc[k] == pred_class:
                                        bar.set_color('red')
                                
                                ax.set_title(f'Feature: {feature}')
                                ax.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.warning("Detailed feature analysis not available for non-DataFrame test data.")
        else:
            st.success("No misclassifications found in the test set!")

def display_live_detection_page():
    """Display live detection and prediction page"""
    st.title("Live Intrusion Detection")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
        if st.button("Go to Model Training Page"):
            st.session_state.page = "Model Training"
            st.experimental_rerun()
        return
    
    # Get model and classes
    model = st.session_state.model
    class_names = st.session_state.class_names
    
    # Create detection options
    detection_tabs = st.tabs(["Sample Data", "Custom Input", "Batch Detection"])
    
    # Tab 1: Sample Data
    with detection_tabs[0]:
        st.header("Detection with Sample Data")
        
        # Create predefined examples for different scenarios
        predefined_examples = {
            "Normal HTTP Traffic": {
                'duration': 2.0,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'SF',
                'src_bytes': 1250,
                'dst_bytes': 8750,
                'land': 0.0,
                'wrong_fragment': 0.0,
                'urgent': 0.0,
                'count': 4
            },
            "Normal SSH Traffic": {
                'duration': 5.12,
                'protocol_type': 'tcp',
                'service': 'ssh',
                'flag': 'SF',
                'src_bytes': 800,
                'dst_bytes': 3500,
                'land': 0.0,
                'wrong_fragment': 0.0,
                'urgent': 0.0,
                'count': 1
            },
            "DOS Attack Example": {
                'duration': 0.5,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'S0',
                'src_bytes': 100,
                'dst_bytes': 0,
                'land': 0.0,
                'wrong_fragment': 0.0,
                'urgent': 0.0,
                'count': 80
            },
            "Probe Attack Example": {
                'duration': 0.01,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'S0',
                'src_bytes': 50,
                'dst_bytes': 0,
                'land': 0.0,
                'wrong_fragment': 0.0, 
                'urgent': 0.0,
                'count': 2
            },
            "R2L Attack Example": {
                'duration': 10.0,
                'protocol_type': 'tcp',
                'service': 'ftp',
                'flag': 'SF',
                'src_bytes': 300,
                'dst_bytes': 12000,
                'land': 0.0,
                'wrong_fragment': 0.0,
                'urgent': 0.0,
                'count': 1
            }
        }
        
        # Add source and destination IPs to examples
        for example in predefined_examples.values():
            example['src_ip'] = '192.168.1.100'
            example['dst_ip'] = '10.0.0.2'
        
        # Sample selection
        selected_example = st.selectbox(
            "Choose a predefined example:",
            list(predefined_examples.keys())
        )
        
        # Display selected example details
        example_data = predefined_examples[selected_example]
        
        # Show example details in a nice format
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Connection Details")
            st.write(f"**Protocol:** {example_data['protocol_type']}")
            st.write(f"**Service:** {example_data['service']}")
            st.write(f"**Flag:** {example_data['flag']}")
            st.write(f"**Duration:** {example_data['duration']} seconds")
            st.write(f"**Connection count:** {example_data['count']}")
        
        with col2:
            st.subheader("Traffic Data")
            st.write(f"**Source IP:** {example_data['src_ip']}")
            st.write(f"**Destination IP:** {example_data['dst_ip']}")
            st.write(f"**Source bytes:** {example_data['src_bytes']}")
            st.write(f"**Destination bytes:** {example_data['dst_bytes']}")
            st.write(f"**Land:** {example_data['land']}")
            st.write(f"**Wrong fragment:** {example_data['wrong_fragment']}")
            st.write(f"**Urgent:** {example_data['urgent']}")
        
        # Detect button
        if st.button("Analyze Traffic", key="sample_detect_btn"):
            with st.spinner("Analyzing traffic pattern..."):
                # Prepare input for prediction
                input_df = pd.DataFrame([example_data])
                
                # Make prediction
                try:
                    # Get categorical and numerical columns
                    categorical_cols = st.session_state.categorical_cols
                    numerical_cols = st.session_state.numerical_cols
                    
                    # Remove IP addresses and any other columns not used in training
                    input_features = input_df.drop(['src_ip', 'dst_ip'], axis=1, errors='ignore')
                    
                    # Make prediction
                    prediction_result = explain_prediction(
                        model, 
                        input_features, 
                        class_names, 
                        st.session_state.feature_names, 
                        categorical_cols, 
                        numerical_cols
                    )
                    
                    # Display prediction result with styling
                    pred_class = prediction_result['prediction']
                    confidence = prediction_result['confidence']
                    
                    # Determine if it's an attack
                    is_attack = pred_class != 'Normal'
                    
                    if is_attack:
                        # Record the alert
                        alert = record_alert(prediction_result, example_data)
                        
                        st.error(f"⚠️ **ALERT: {pred_class} attack detected!**")
                        st.write(f"Confidence: {confidence:.2%}")
                        
                        # Alert ID for reference
                        st.info(f"Alert #{alert['id']} has been recorded. See the Alerts page for details.")
                    else:
                        st.success("✅ **Normal traffic detected**")
                        st.write(f"Confidence: {confidence:.2%}")
                    
                    # Show all class probabilities
                    st.subheader("Detection Confidence by Class")
                    
                    # Convert probabilities to DataFrame
                    probs_df = pd.DataFrame({
                        'Class': list(prediction_result['probabilities'].keys()),
                        'Probability': list(prediction_result['probabilities'].values())
                    }).sort_values('Probability', ascending=False)
                    
                    # Plot probabilities
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = sns.barplot(x='Class', y='Probability', data=probs_df, ax=ax)
                    
                    # Color the bars based on predicted class
                    for i, bar in enumerate(bars.patches):
                        if probs_df.iloc[i]['Class'] == pred_class:
                            bar.set_color('red' if is_attack else 'green')
                    
                    ax.set_title('Prediction Confidence by Class')
                    ax.set_ylim(0, 1)
                    for i, p in enumerate(ax.patches):
                        ax.annotate(f"{p.get_height():.2%}", 
                                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                                   ha = 'center', va = 'bottom')
                    
                    st.pyplot(fig)
                    
                    # Show explanation for the prediction
                    st.subheader("Explanation")
                    
                    if 'explanation' in prediction_result and prediction_result['explanation']:
                        explanation = prediction_result['explanation']
                        
                        # Format the explanation for display
                        if 'top_features' in explanation:
                            st.write("**Top contributing features:**")
                            for feature, importance in explanation['top_features']:
                                st.write(f"- {feature}: {importance:.4f}")
                        
                        # Suspicious values
                        suspicious_features = []
                        for key, value in explanation.items():
                            if key != 'top_features':
                                if key.startswith('high_'):
                                    feature = key.replace('high_', '')
                                    suspicious_features.append(f"High {feature}: {value}")
                                elif key.startswith('low_'):
                                    feature = key.replace('low_', '')
                                    suspicious_features.append(f"Low {feature}: {value}")
                                elif key.startswith('unusual_'):
                                    feature = key.replace('unusual_', '')
                                    suspicious_features.append(f"Unusual {feature}: {value}")
                        
                        if suspicious_features:
                            st.write("**Suspicious values:**")
                            for feature in suspicious_features:
                                st.write(f"- {feature}")
                        
                        # Flow diagram for attack path
                        if is_attack:
                            st.subheader("Attack Pattern Analysis")
                            
                            # Create a custom attack path visualization based on attack type
                            if pred_class == 'DOS':
                                st.write("""
                                **DOS Attack Pattern Detected:**
                                
                                This appears to be a Denial of Service attack with typical characteristics:
                                - High connection count (80)
                                - Low duration per connection
                                - Little to no data transfer
                                - Rejected connection flags (S0)
                                
                                Recommendation: Block source IP and monitor for distributed attack patterns.
                                """)
                            elif pred_class == 'Probe':
                                st.write("""
                                **Probe Attack Pattern Detected:**
                                
                                This appears to be a network probe with typical characteristics:
                                - Very short connection duration
                                - Minimal data transfer
                                - Multiple connection attempts
                                
                                Recommendation: Monitor source IP for further suspicious activity.
                                """)
                            elif pred_class == 'R2L':
                                st.write("""
                                **Remote to Local (R2L) Attack Pattern Detected:**
                                
                                This appears to be an attempt to gain unauthorized access:
                                - Normal-looking connection
                                - Suspicious data transfer pattern (small to source, large from destination)
                                - Sensitive service used (FTP)
                                
                                Recommendation: Investigate user authentication and file access logs.
                                """)
                            elif 'U2R' in pred_class:
                                st.write("""
                                **User to Root (U2R) Attack Pattern Detected:**
                                
                                This appears to be a privilege escalation attempt:
                                - Normal-looking connection
                                - Suspicious data size
                                - Potential command execution
                                
                                Recommendation: Immediate investigation of user activities and system logs.
                                """)
                    else:
                        st.write("No detailed explanation available for this prediction.")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
    
    # Tab 2: Custom Input
    with detection_tabs[1]:
        st.header("Custom Traffic Analysis")
        
        # Get categorical and numerical features
        categorical_cols = st.session_state.categorical_cols
        numerical_cols = st.session_state.numerical_cols
        
        if not categorical_cols or not numerical_cols:
            st.warning("Feature information not available. Please retrain the model.")
            return
        
        st.info("Enter details about the network traffic you want to analyze:")
        
        # Create input form with multiple columns for better layout
        col1, col2 = st.columns(2)
        
        # Dictionary to store input values
        input_data = {}
        
        with col1:
            st.subheader("Basic Information")
            
            # Protocol type
            input_data['protocol_type'] = st.selectbox(
                "Protocol Type:",
                ["tcp", "udp", "icmp"]
            )
            
            # Service
            input_data['service'] = st.selectbox(
                "Service:",
                ["http", "ftp", "smtp", "ssh", "dns", "pop3", "telnet", "imap", "sql"]
            )
            
            # Flag
            input_data['flag'] = st.selectbox(
                "Connection Flag:",
                ["SF", "S0", "REJ", "RSTO", "RSTR", "SH", "S1", "S2"]
            )
            
            # Duration
            input_data['duration'] = st.number_input(
                "Connection Duration (seconds):",
                min_value=0.0,
                max_value=1000.0,
                value=2.0,
                step=0.1
            )
            
            # Connection count
            input_data['count'] = st.number_input(
                "Connection Count:",
                min_value=1,
                max_value=1000,
                value=1,
                step=1,
                help="Number of connections to the same host in past 2 seconds"
            )
        
        with col2:
            st.subheader("Traffic Details")
            
            # Data transfer
            input_data['src_bytes'] = st.number_input(
                "Source Bytes:",
                min_value=0,
                max_value=100000,
                value=1000,
                step=100,
                help="Bytes sent from source to destination"
            )
            
            input_data['dst_bytes'] = st.number_input(
                "Destination Bytes:",
                min_value=0,
                max_value=100000,
                value=2000,
                step=100,
                help="Bytes sent from destination to source"
            )
            
            # Other flags
            input_data['land'] = st.selectbox(
                "Land Flag:",
                [0.0, 1.0],
                format_func=lambda x: "Yes" if x == 1.0 else "No",
                help="1 if connection is from/to same host/port; 0 otherwise"
            )
            
            input_data['wrong_fragment'] = st.number_input(
                "Wrong Fragment:",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                help="Number of wrong fragments"
            )
            
            input_data['urgent'] = st.number_input(
                "Urgent Packets:",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                help="Number of urgent packets"
            )
        
        # Add IP addresses for realism
        input_data['src_ip'] = st.text_input("Source IP:", "192.168.1.100")
        input_data['dst_ip'] = st.text_input("Destination IP:", "10.0.0.1")
        
        # Analyze button
        if st.button("Analyze Traffic", key="custom_detect_btn"):
            with st.spinner("Analyzing custom traffic pattern..."):
                try:
                    # Prepare input for prediction
                    input_df = pd.DataFrame([input_data])
                    
                    # Remove IP addresses for prediction
                    input_features = input_df.drop(['src_ip', 'dst_ip'], axis=1, errors='ignore')
                    
                    # Make prediction
                    prediction_result = explain_prediction(
                        model, 
                        input_features, 
                        class_names, 
                        st.session_state.feature_names, 
                        categorical_cols, 
                        numerical_cols
                    )
                    
                    # Display prediction result
                    pred_class = prediction_result['prediction']
                    confidence = prediction_result['confidence']
                    
                    # Determine if it's an attack
                    is_attack = pred_class != 'Normal'
                    
                    if is_attack:
                        # Record the alert
                        alert = record_alert(prediction_result, input_data)
                        
                        # Create a styled alert box
                        st.markdown(f"""
                        <div style="background-color: #FFCCCC; padding: 20px; border-radius: 10px; border-left: 6px solid #CC0000;">
                            <h3 style="color: #CC0000;">⚠️ ALERT: {pred_class} attack detected!</h3>
                            <p>Confidence: {confidence:.2%}</p>
                            <p>Alert #{alert['id']} has been recorded.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Create a styled success box
                        st.markdown(f"""
                        <div style="background-color: #CCFFCC; padding: 20px; border-radius: 10px; border-left: 6px solid #007700;">
                            <h3 style="color: #007700;">✅ Normal traffic detected</h3>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detection details
                    with st.expander("Detection Details", expanded=True):
                        # Show all class probabilities
                        st.subheader("Detection Confidence by Class")
                        
                        # Convert probabilities to DataFrame
                        probs_df = pd.DataFrame({
                            'Class': list(prediction_result['probabilities'].keys()),
                            'Probability': list(prediction_result['probabilities'].values())
                        }).sort_values('Probability', ascending=False)
                        
                        # Plot probabilities
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = sns.barplot(x='Class', y='Probability', data=probs_df, ax=ax)
                        
                        # Color the bars based on predicted class
                        for i, bar in enumerate(bars.patches):
                            if probs_df.iloc[i]['Class'] == pred_class:
                                bar.set_color('red' if is_attack else 'green')
                        
                        ax.set_title('Prediction Confidence by Class')
                        ax.set_ylim(0, 1)
                        for i, p in enumerate(ax.patches):
                            ax.annotate(f"{p.get_height():.2%}", 
                                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                                       ha = 'center', va = 'bottom')
                        
                        st.pyplot(fig)
                        
                        # Show explanation for the prediction
                        st.subheader("Explanation")
                        
                        if 'explanation' in prediction_result and prediction_result['explanation']:
                            explanation = prediction_result['explanation']
                            
                            # Format the explanation for display
                            if 'top_features' in explanation:
                                st.write("**Top contributing features:**")
                                for feature, importance in explanation['top_features']:
                                    st.write(f"- {feature}: {importance:.4f}")
                            
                            # Suspicious values
                            suspicious_features = []
                            for key, value in explanation.items():
                                if key != 'top_features':
                                    if key.startswith('high_'):
                                        feature = key.replace('high_', '')
                                        suspicious_features.append(f"High {feature}: {value}")
                                    elif key.startswith('low_'):
                                        feature = key.replace('low_', '')
                                        suspicious_features.append(f"Low {feature}: {value}")
                                    elif key.startswith('unusual_'):
                                        feature = key.replace('unusual_', '')
                                        suspicious_features.append(f"Unusual {feature}: {value}")
                            
                            if suspicious_features:
                                st.write("**Suspicious values:**")
                                for feature in suspicious_features:
                                    st.write(f"- {feature}")
                        else:
                            st.write("No detailed explanation available for this prediction.")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
    
    # Tab 3: Batch Detection
    with detection_tabs[2]:
        st.header("Batch Detection")
        
        st.info("""
        Upload a CSV file with multiple traffic records for batch analysis.
        The file should contain the same features used for training the model.
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload traffic data CSV:",
            type=["csv"],
            help="CSV file containing network traffic data"
        )
        
        if uploaded_file is not None:
            try:
                # Load the data
                batch_df = pd.read_csv(uploaded_file)
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(batch_df.head())
                
                # Check for required columns
                categorical_cols = st.session_state.categorical_cols
                numerical_cols = st.session_state.numerical_cols
                
                missing_cols = []
                for col in categorical_cols + numerical_cols:
                    if col not in batch_df.columns:
                        missing_cols.append(col)
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Process batch button
                    if st.button("Process Batch"):
                        with st.spinner("Analyzing batch data..."):
                            try:
                                # Prepare input for prediction
                                # Remove IP addresses and any other columns not used in training
                                batch_features = batch_df.drop(['src_ip', 'dst_ip', 'timestamp', 'class'], 
                                                             axis=1, errors='ignore')
                                
                                # Make predictions
                                batch_predictions = model.predict(batch_features)
                                
                                # Get prediction probabilities if available
                                if hasattr(model, "predict_proba"):
                                    batch_probas = model.predict_proba(batch_features)
                                    
                                    # Add the highest probability for each prediction
                                    batch_confidence = np.max(batch_probas, axis=1)
                                else:
                                    batch_confidence = np.ones(len(batch_predictions))
                                
                                # Convert predictions to class names
                                batch_pred_classes = [class_names[p] for p in batch_predictions]
                                
                                # Add predictions to the original DataFrame
                                result_df = batch_df.copy()
                                result_df['predicted_class'] = batch_pred_classes
                                result_df['confidence'] = batch_confidence
                                
                                # Identify potential attacks
                                result_df['is_attack'] = result_df['predicted_class'] != 'Normal'
                                
                                # Show results summary
                                st.subheader("Detection Results")
                                
                                # Count by prediction class
                                pred_counts = result_df['predicted_class'].value_counts()
                                
                                # Display counts
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Total Records", len(result_df))
                                    st.metric("Alerts", sum(result_df['is_attack']))
                                
                                with col2:
                                    # Calculate average confidence
                                    avg_conf = result_df['confidence'].mean()
                                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                                    
                                    # High confidence attacks
                                    high_conf_attacks = sum((result_df['is_attack']) & (result_df['confidence'] > 0.9))
                                    st.metric("High Confidence Alerts", high_conf_attacks)
                                
                                # Plot prediction distribution
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.countplot(x='predicted_class', data=result_df, ax=ax)
                                ax.set_title('Detection Results')
                                ax.set_xlabel('Predicted Class')
                                ax.set_ylabel('Count')
                                
                                # Add count labels on top of bars
                                for i, p in enumerate(ax.patches):
                                    ax.annotate(f"{p.get_height()}", 
                                               (p.get_x() + p.get_width() / 2., p.get_height() + 0.5), 
                                               ha = 'center')
                                
                                st.pyplot(fig)
                                
                                # Show detailed results
                                st.subheader("Detailed Results")
                                
                                # Filter options
                                show_option = st.radio(
                                    "Display:",
                                    ["All Records", "Attacks Only", "Normal Only"]
                                )
                                
                                if show_option == "Attacks Only":
                                    filtered_df = result_df[result_df['is_attack']]
                                elif show_option == "Normal Only":
                                    filtered_df = result_df[~result_df['is_attack']]
                                else:
                                    filtered_df = result_df
                                
                                # Sort options
                                sort_by = st.selectbox(
                                    "Sort by:",
                                    ["confidence", "predicted_class"]
                                )
                                
                                # Display sorted results
                                sorted_df = filtered_df.sort_values(sort_by, ascending=False)
                                st.dataframe(sorted_df)
                                
                                # Add download option for results
                                csv = sorted_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Results CSV",
                                    data=csv,
                                    file_name="batch_detection_results.csv",
                                    mime="text/csv"
                                )
                                
                                # Record alerts
                                if sum(result_df['is_attack']) > 0:
                                    if st.button("Record All Alerts"):
                                        attack_records = result_df[result_df['is_attack']]
                                        
                                        for _, row in attack_records.iterrows():
                                            # Create prediction result dict
                                            pred_result = {
                                                'prediction': row['predicted_class'],
                                                'confidence': row['confidence']
                                            }
                                            
                                            # Record the alert
                                            record_alert(pred_result, row.to_dict())
                                        
                                        st.success(f"Recorded {len(attack_records)} alerts!")
                                        
                            except Exception as e:
                                st.error(f"Error during batch processing: {str(e)}")
                                st.exception(e)
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Template for batch upload
        with st.expander("Batch Upload Template"):
            st.write("""
            Your CSV file should include the following columns:
            - protocol_type
            - service
            - flag
            - duration
            - src_bytes
            - dst_bytes
            - land
            - wrong_fragment
            - urgent
            - count
            
            Optional columns:
            - src_ip
            - dst_ip
            - timestamp
            """)
            
            # Provide a template
            template_data = {
                'duration': [2.0, 0.5, 10.0],
                'protocol_type': ['tcp', 'icmp', 'udp'],
                'service': ['http', 'dns', 'ftp'],
                'flag': ['SF', 'S0', 'REJ'],
                'src_bytes': [1250, 50, 300],
                'dst_bytes': [8750, 0, 12000],
                'land': [0.0, 0.0, 0.0],
                'wrong_fragment': [0.0, 0.0, 0.0],
                'urgent': [0.0, 0.0, 0.0],
                'count': [4, 80, 1],
                'src_ip': ['192.168.1.100', '10.0.0.5', '172.16.10.1'],
                'dst_ip': ['10.0.0.2', '192.168.1.1', '8.8.8.8']
            }
            
            template_df = pd.DataFrame(template_data)
            
            # Show template
            st.dataframe(template_df)
            
            # Provide download link for template
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="Download Template CSV",
                data=csv,
                file_name="batch_template.csv",
                mime="text/csv"
            )

def display_alerts_page():
    """Display alerts management dashboard"""
    st.title("Intrusion Alerts Dashboard")
    
    # Initialize alerts if not present
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    # Get alerts
    alerts = st.session_state.alerts
    
    if not alerts:
        st.info("No alerts recorded yet. Use the Live Detection page to generate alerts.")
        
        # Option to generate sample alerts
        if st.button("Generate Sample Alerts"):
            # Create some sample alerts
            sample_alerts = [
                {
                    'timestamp': datetime.now() - pd.Timedelta(minutes=5),
                    'prediction': 'DOS',
                    'confidence': 0.95,
                    'data': {
                        'protocol_type': 'tcp',
                        'service': 'http',
                        'flag': 'S0',
                        'duration': 0.5,
                        'src_bytes': 100,
                        'dst_bytes': 0,
                        'count': 80,
                        'src_ip': '192.168.1.100',
                        'dst_ip': '10.0.0.2'
                    },
                    'status': 'New',
                    'id': 1
                },
                {
                    'timestamp': datetime.now() - pd.Timedelta(minutes=15),
                    'prediction': 'Probe',
                    'confidence': 0.88,
                    'data': {
                        'protocol_type': 'tcp',
                        'service': 'http',
                        'flag': 'S0',
                        'duration': 0.01,
                        'src_bytes': 50,
                        'dst_bytes': 0,
                        'count': 2,
                        'src_ip': '192.168.1.101',
                        'dst_ip': '10.0.0.2'
                    },
                    'status': 'New',
                    'id': 2
                },
                {
                    'timestamp': datetime.now() - pd.Timedelta(hours=1),
                    'prediction': 'R2L',
                    'confidence': 0.76,
                    'data': {
                        'protocol_type': 'tcp',
                        'service': 'ftp',
                        'flag': 'SF',
                        'duration': 10.0,
                        'src_bytes': 300,
                        'dst_bytes': 12000,
                        'count': 1,
                        'src_ip': '192.168.1.102',
                        'dst_ip': '10.0.0.3'
                    },
                    'status': 'New',
                    'id': 3
                }
            ]
            
            st.session_state.alerts.extend(sample_alerts)
            st.success("Sample alerts generated!")
            st.experimental_rerun()
        
        return
    
    # Dashboard overview
    st.subheader("Alerts Overview")
    
    # Convert alerts to DataFrame for easier manipulation
    alerts_df = pd.DataFrame(alerts)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alerts))
    
    with col2:
        new_alerts = sum(1 for alert in alerts if alert['status'] == 'New')
        st.metric("New Alerts", new_alerts)
    
    with col3:
        # Number of unique source IPs
        if 'data' in alerts_df.columns:
            unique_ips = len(set(a['data'].get('src_ip', '') for a in alerts if 'data' in a and 'src_ip' in a['data']))
            st.metric("Unique Source IPs", unique_ips)
        else:
            st.metric("Unique Source IPs", "N/A")
    
    with col4:
        # Most common attack type
        if len(alerts) > 0:
            attack_counts = alerts_df['prediction'].value_counts()
            most_common = attack_counts.index[0]
            st.metric("Most Common Attack", most_common)
        else:
            st.metric("Most Common Attack", "N/A")
    
    # Alert visualization
    st.subheader("Alert Visualization")
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["Attack Types", "Timeline", "Source IPs"])
    
    # Tab 1: Attack Types
    with viz_tabs[0]:
        # Count attacks by type
        attack_counts = alerts_df['prediction'].value_counts().reset_index()
        attack_counts.columns = ['Attack Type', 'Count']
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x='Attack Type', y='Count', data=attack_counts, ax=ax)
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{int(p.get_height())}", 
                       (p.get_x() + p.get_width() / 2., p.get_height() + 0.1), 
                       ha = 'center')
        
        ax.set_title('Alerts by Attack Type')
        
        st.pyplot(fig)
    
    # Tab 2: Timeline
    with viz_tabs[1]:
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in alerts_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(alerts_df['timestamp']):
                alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            
            # Group by hour and attack type
            alerts_df['hour'] = alerts_df['timestamp'].dt.floor('H')
            timeline_data = alerts_df.groupby(['hour', 'prediction']).size().reset_index()
            timeline_data.columns = ['Hour', 'Attack Type', 'Count']
            
            # Create timeline visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create a pivot table for easier plotting
            pivot_data = timeline_data.pivot(index='Hour', columns='Attack Type', values='Count').fillna(0)
            
            # Plot stacked bar chart
            pivot_data.plot(kind='bar', stacked=True, ax=ax)
            
            ax.set_title('Alerts Timeline')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Number of Alerts')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
        else:
            st.warning("Timestamp information not available for timeline visualization")
    
    # Tab 3: Source IPs
    with viz_tabs[2]:
        # Extract source IPs from data
        if 'data' in alerts_df.columns:
            # Get source IPs
            src_ips = [a['data'].get('src_ip', 'Unknown') for a in alerts if 'data' in a and 'src_ip' in a['data']]
            
            # Count alerts by source IP
            ip_counts = pd.Series(src_ips).value_counts().reset_index()
            ip_counts.columns = ['Source IP', 'Count']
            
            # Only show top 10 IPs
            top_ips = ip_counts.head(10)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='Source IP', y='Count', data=top_ips, ax=ax)
            
            # Add count labels
            for i, p in enumerate(ax.patches):
                ax.annotate(f"{int(p.get_height())}", 
                           (p.get_x() + p.get_width() / 2., p.get_height() + 0.1), 
                           ha = 'center')
            
            ax.set_title('Top 10 Source IPs by Alert Count')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
        else:
            st.warning("Source IP information not available for visualization")
    
    # Alerts table with filters
    st.subheader("Alerts Management")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by status
        status_filter = st.selectbox(
            "Filter by Status:",
            ["All", "New", "Investigating", "Resolved", "False Positive"]
        )
    
    with col2:
        # Filter by attack type
        attack_types = ["All"] + list(alerts_df['prediction'].unique())
        attack_filter = st.selectbox(
            "Filter by Attack Type:",
            attack_types
        )
    
    with col3:
        # Filter by source IP
        if 'data' in alerts_df.columns:
            src_ips = ["All"] + list(set(a['data'].get('src_ip', '') for a in alerts if 'data' in a and 'src_ip' in a['data']))
            ip_filter = st.selectbox(
                "Filter by Source IP:",
                src_ips
            )
        else:
            ip_filter = "All"
    
    # Apply filters to alerts
    filtered_alerts = alerts
    
    if status_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a['status'] == status_filter]
    
    if attack_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a['prediction'] == attack_filter]
    
    if ip_filter != "All" and 'data' in alerts_df.columns:
        filtered_alerts = [a for a in filtered_alerts if 'data' in a and a['data'].get('src_ip', '') == ip_filter]
    
    # Show alert table
    if filtered_alerts:
        # Create a table for display
        table_data = []
        for alert in filtered_alerts:
            row = {
                'ID': alert['id'],
                'Timestamp': alert['timestamp'],
                'Type': alert['prediction'],
                'Confidence': f"{alert['confidence']:.2%}",
                'Status': alert['status']
            }
            
            # Add source IP if available
            if 'data' in alert and 'src_ip' in alert['data']:
                row['Source IP'] = alert['data']['src_ip']
            else:
                row['Source IP'] = "N/A"
            
            table_data.append(row)
        
        # Convert to DataFrame
        table_df = pd.DataFrame(table_data)
        
        # Display the table
        st.dataframe(table_df)
        
        # Alert details
        st.subheader("Alert Details")
        
        # Let user select an alert to view
        selected_id = st.number_input(
            "Select Alert ID to view details:",
            min_value=min(a['id'] for a in filtered_alerts),
            max_value=max(a['id'] for a in filtered_alerts),
            value=min(a['id'] for a in filtered_alerts)
        )
        
        # Find the selected alert
        selected_alert = next((a for a in alerts if a['id'] == selected_id), None)
        
        if selected_alert:
            # Display alert details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Alert ID:** {selected_alert['id']}")
                st.write(f"**Timestamp:** {selected_alert['timestamp']}")
                st.write(f"**Attack Type:** {selected_alert['prediction']}")
                st.write(f"**Confidence:** {selected_alert['confidence']:.2%}")
            
            with col2:
                st.write(f"**Status:** {selected_alert['status']}")
                
                # Status update
                new_status = st.selectbox(
                    "Update Status:",
                    ["New", "Investigating", "Resolved", "False Positive"],
                    index=["New", "Investigating", "Resolved", "False Positive"].index(selected_alert['status'])
                )
                
                if new_status != selected_alert['status']:
                    if st.button("Update Status"):
                        selected_alert['status'] = new_status
                        st.success(f"Status updated to {new_status}")
                        st.experimental_rerun()
            
            # Show alert data
            st.subheader("Traffic Details")
            
            if 'data' in selected_alert:
                # Display connection details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Connection Information:**")
                    if 'protocol_type' in selected_alert['data']:
                        st.write(f"Protocol: {selected_alert['data']['protocol_type']}")
                    if 'service' in selected_alert['data']:
                        st.write(f"Service: {selected_alert['data']['service']}")
                    if 'flag' in selected_alert['data']:
                        st.write(f"Flag: {selected_alert['data']['flag']}")
                    if 'duration' in selected_alert['data']:
                        st.write(f"Duration: {selected_alert['data']['duration']} seconds")
                    if 'count' in selected_alert['data']:
                        st.write(f"Connection count: {selected_alert['data']['count']}")
                
                with col2:
                    st.write("**Traffic Data:**")
                    if 'src_ip' in selected_alert['data']:
                        st.write(f"Source IP: {selected_alert['data']['src_ip']}")
                    if 'dst_ip' in selected_alert['data']:
                        st.write(f"Destination IP: {selected_alert['data']['dst_ip']}")
                    if 'src_bytes' in selected_alert['data']:
                        st.write(f"Source bytes: {selected_alert['data']['src_bytes']}")
                    if 'dst_bytes' in selected_alert['data']:
                        st.write(f"Destination bytes: {selected_alert['data']['dst_bytes']}")
                
                # Add actions
                st.subheader("Alert Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Block Source IP"):
                        st.info(f"Action simulated: Blocked {selected_alert['data'].get('src_ip', 'unknown IP')}")
                
                with col2:
                    if st.button("Export Alert Details"):
                        # Convert alert to JSON
                        alert_json = json.dumps(selected_alert, default=str, indent=2)
                        
                        # Create download button
                        st.download_button(
                            label="Download Alert JSON",
                            data=alert_json,
                            file_name=f"alert_{selected_alert['id']}.json",
                            mime="application/json"
                        )
                
                with col3:
                    if st.button("Delete Alert"):
                        # Remove the alert from the list
                        st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != selected_alert['id']]
                        st.success("Alert deleted!")
                        st.experimental_rerun()
            else:
                st.warning("No detailed data available for this alert.")
        else:
            st.error("Alert not found!")
    else:
        st.info("No alerts match the selected filters.")
    
    # Bulk actions
    st.subheader("Bulk Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Mark All as Read"):
            # Update all new alerts to investigating
            for alert in st.session_state.alerts:
                if alert['status'] == 'New':
                    alert['status'] = 'Investigating'
            
            st.success("All new alerts marked as Investigating")
            st.experimental_rerun()
    
    with col2:
        if st.button("Export All Alerts"):
            # Convert all alerts to CSV
            alerts_export = []
            
            for alert in alerts:
                export_row = {
                    'id': alert['id'],
                    'timestamp': alert['timestamp'],
                    'prediction': alert['prediction'],
                    'confidence': alert['confidence'],
                    'status': alert['status']
                }
                
                # Add data fields
                if 'data' in alert:
                    for key, value in alert['data'].items():
                        export_row[f'data_{key}'] = value
                
                alerts_export.append(export_row)
            
            # Convert to DataFrame
            export_df = pd.DataFrame(alerts_export)
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="Download All Alerts CSV",
                data=csv,
                file_name="all_alerts.csv",
                mime="text/csv"
            )

def display_settings_page():
    """Display application settings and configuration page"""
    st.title("Settings & Configuration")
    
    # Create tabs for different settings categories
    settings_tabs = st.tabs(["Application Settings", "Model Settings", "System Information", "About"])
    
    # Tab 1: Application Settings
    with settings_tabs[0]:
        st.header("Application Settings")
        
        # UI Preferences
        st.subheader("UI Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Theme setting
            theme = st.selectbox(
                "Interface Theme:",
                ["Light", "Dark", "System Default"],
                index=2
            )
            
            # Chart style
            chart_style = st.selectbox(
                "Chart Style:",
                ["Default", "Whitegrid", "Darkgrid", "Ticks"],
                index=0,
                help="Visual style for charts and plots"
            )
            
            # Apply chart style
            if chart_style == "Whitegrid":
                sns.set_style("whitegrid")
            elif chart_style == "Darkgrid":
                sns.set_style("darkgrid")
            elif chart_style == "Ticks":
                sns.set_style("ticks")
            else:
                sns.set_style("darkgrid")  # Default
        
        with col2:
            # Table display options
            show_dataframe_styling = st.checkbox(
                "Enhanced DataFrame Styling",
                value=True,
                help="Apply enhanced styling to data tables (may affect performance)"
            )
            
            # Animation option
            enable_animations = st.checkbox(
                "Enable Animations",
                value=True,
                help="Enable UI animations and transitions"
            )
            
            # Developer mode
            developer_mode = st.checkbox(
                "Developer Mode",
                value=False,
                help="Show additional technical information and options"
            )
        
        # Alert Settings
        st.subheader("Alert Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert threshold
            alert_threshold = st.slider(
                "Alert Confidence Threshold:",
                min_value=0.5,
                max_value=0.99,
                value=0.7,
                step=0.01,
                help="Minimum confidence level to generate an alert"
            )
            
            # Auto-refresh
            enable_auto_refresh = st.checkbox(
                "Enable Auto-Refresh",
                value=False,
                help="Automatically refresh the alerts dashboard"
            )
            
            if enable_auto_refresh:
                refresh_interval = st.number_input(
                    "Refresh Interval (seconds):",
                    min_value=5,
                    max_value=300,
                    value=60,
                    step=5
                )
        
        with col2:
            # Alert retention
            alert_retention = st.number_input(
                "Alert Retention (days):",
                min_value=1,
                max_value=365,
                value=30,
                step=1,
                help="Number of days to keep alerts before automatic deletion"
            )
            
            # Alert export format
            alert_export_format = st.selectbox(
                "Default Export Format:",
                ["CSV", "JSON", "Excel"],
                index=0
            )
        
        # Save settings button
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
            
            # In a real app, would save to session_state or config file
            st.session_state.settings = {
                'theme': theme,
                'chart_style': chart_style,
                'show_dataframe_styling': show_dataframe_styling,
                'enable_animations': enable_animations,
                'developer_mode': developer_mode,
                'alert_threshold': alert_threshold,
                'enable_auto_refresh': enable_auto_refresh,
                'refresh_interval': refresh_interval if enable_auto_refresh else None,
                'alert_retention': alert_retention,
                'alert_export_format': alert_export_format
            }
    
    # Tab 2: Model Settings
    with settings_tabs[1]:
        st.header("Model Settings")
        
        if st.session_state.model is None:
            st.warning("No model trained. Train a model first to access model settings.")
        else:
            # Get model information
            if hasattr(st.session_state.model, 'named_steps') and 'model' in st.session_state.model.named_steps:
                model_step = st.session_state.model.named_steps['model']
                model_type = type(model_step).__name__
            else:
                model_type = type(st.session_state.model).__name__
            
            st.subheader(f"Current Model: {model_type}")
            
            # Model parameters
            if hasattr(st.session_state.model, 'named_steps') and 'model' in st.session_state.model.named_steps:
                model_params = st.session_state.model.named_steps['model'].get_params()
                
                st.write("**Model Parameters:**")
                st.json(model_params)
            
            # Model export
            st.subheader("Model Export")
            
            export_format = st.radio(
                "Export Format:",
                ["Pickle", "Joblib"],
                horizontal=True
            )
            
            if st.button("Export Model"):
                with st.spinner("Preparing model for export..."):
                    try:
                        if export_format == "Pickle":
                            # Serialize with pickle
                            model_serialized = pickle.dumps(st.session_state.model)
                            
                            # Create download button
                            st.download_button(
                                label="Download Model (pickle)",
                                data=model_serialized,
                                file_name="intrusion_detection_model.pkl",
                                mime="application/octet-stream"
                            )
                        else:  # Joblib
                            # Create a BytesIO object
                            bio = BytesIO()
                            
                            # Save model to BytesIO
                            joblib.dump(st.session_state.model, bio)
                            
                            # Create download button
                            st.download_button(
                                label="Download Model (joblib)",
                                data=bio.getvalue(),
                                file_name="intrusion_detection_model.joblib",
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"Error exporting model: {str(e)}")
            
            # Model retraining settings
            st.subheader("Retraining Settings")
            
            # Automatic retraining
            enable_auto_retrain = st.checkbox(
                "Enable Automatic Retraining",
                value=False,
                help="Automatically retrain the model periodically or when performance degrades"
            )
            
            if enable_auto_retrain:
                col1, col2 = st.columns(2)
                
                with col1:
                    retrain_freq = st.selectbox(
                        "Retraining Frequency:",
                        ["Daily", "Weekly", "Monthly", "On Performance Drop"],
                        index=1
                    )
                
                with col2:
                    # Performance threshold
                    perf_threshold = st.slider(
                        "Performance Threshold:",
                        min_value=0.7,
                        max_value=0.99,
                        value=0.8,
                        step=0.01,
                        help="Retrain when model performance drops below this threshold"
                    )
            
            # Model deployment
            st.subheader("Model Deployment")
            
            deployment_mode = st.selectbox(
                "Deployment Mode:",
                ["Development", "Testing", "Production"],
                index=0
            )
            
            if deployment_mode == "Production":
                st.warning("""
                Production mode applies additional security and performance optimizations.
                Ensure you have thoroughly tested the model before enabling production mode.
                """)
    
    # Tab 3: System Information
    with settings_tabs[2]:
        st.header("System Information")
        
        # System metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Application Status")
            
            # Memory usage
            import psutil
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
            
            # Session info
            st.metric("Session Duration", "00:30:15")  # Placeholder
            
            # Data stats
            if st.session_state.data is not None:
                st.metric("Dataset Size", f"{st.session_state.data.shape[0]} records")
            else:
                st.metric("Dataset Size", "No data loaded")
        
        with col2:
            st.subheader("System Resources")
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            st.metric("CPU Usage", f"{cpu_percent}%")
            
            # Memory stats
            vm = psutil.virtual_memory()
            st.metric("System Memory", f"{vm.percent}% used")
            
            # Disk space
            disk = psutil.disk_usage('/')
            st.metric("Disk Space", f"{disk.percent}% used")
        
        # Environment information
        st.subheader("Environment Information")
        
        # Python version
        import sys
        st.write(f"**Python Version:** {sys.version.split(' ')[0]}")
        
        # Streamlit version
        st.write(f"**Streamlit Version:** {st.__version__}")
        
        # Pandas version
        st.write(f"**Pandas Version:** {pd.__version__}")
        
        # Scikit-learn version
        import sklearn
        st.write(f"**Scikit-learn Version:** {sklearn.__version__}")
        
        # System platform
        st.write(f"**Platform:** {sys.platform}")
        
        # View logs button
        if st.button("View Application Logs"):
            st.info("Logs would be displayed here in a real application.")
            
            # Placeholder for logs
            st.code("""
            2023-07-10 15:32:45 INFO: Application started
            2023-07-10 15:33:12 INFO: Model loaded successfully
            2023-07-10 15:34:01 WARNING: High memory usage detected
            2023-07-10 15:35:22 INFO: Dataset processed successfully
            """)
    
    # Tab 4: About
    with settings_tabs[3]:
        st.header("About")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # App icon/logo would go here
            st.image("https://via.placeholder.com/200x200?text=IDS", width=200)
        
        with col2:
            st.subheader("AI Intrusion Detection System")
            st.write("Version 2.0")
            st.write("© 2023 AI Security Solutions")
            
            st.write("""
            This application provides machine learning-based network intrusion detection capabilities, 
            helping security teams identify and respond to potential threats in real-time.
            """)
            
            st.write("**Features:**")
            st.write("- Multi-class attack detection")
            st.write("- Interactive visualizations")
            st.write("- Real-time monitoring")
            st.write("- Alert management")
            st.write("- Model customization")
        
        # Documentation link
        st.subheader("Documentation")
        
        st.markdown("""
        For complete documentation, please visit the [User Guide](https://example.com/docs) (placeholder link).
        
        **Quick Links:**
        - [Getting Started](https://example.com/docs/getting-started)
        - [Model Training Guide](https://example.com/docs/model-training)
        - [Alert Response Procedures](https://example.com/docs/alerts)
        - [API Documentation](https://example.com/docs/api)
        """)
        
        # Credits and acknowledgments
        st.subheader("Credits")
        
        st.write("""
        This application is built using:
        
        - Streamlit
        - Pandas
        - Scikit-learn
        - Matplotlib
        - Seaborn
        
        Special thanks to the open-source community for making this project possible.
        """)
        
        # License information
        st.subheader("License")
        
        st.write("""
        This software is released under the MIT License.
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        """)

# -----------------
# MAIN APPLICATION
# -----------------

def main():
    """Main entry point for the application"""
    
    # Set page configuration
    st.set_page_config(
        page_title="AI Intrusion Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
        }
        .error-card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            background-color: #FFEBEE;
            border-left: 4px solid #F44336;
        }
        .info-card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            background-color: #E3F2FD;
            border-left: 4px solid #2196F3;
        }
        .download-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #4CAF50;
            color: white;
            text-align: center;
            text-decoration: none;
            border-radius: 0.25rem;
            transition: background-color 0.3s;
        }
        .download-button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar navigation
    page = display_sidebar()
    
    # Display the selected page
    if page == "About":
        display_about_page()
    elif page == "Dataset":
        display_dataset_page()
    elif page == "Exploration":
        display_exploration_page()
    elif page == "Model Training":
        display_model_training_page()
    elif page == "Evaluation":
        display_evaluation_page()
    elif page == "Live Detection":
        display_live_detection_page()
    elif page == "Alerts":
        display_alerts_page()
    elif page == "Settings":
        display_settings_page()
    else:
        display_about_page()

if __name__ == "__main__":
    main()