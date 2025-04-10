import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import base64
import os

# Set page config
st.set_page_config(
    page_title=" Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Main title
st.title("🛡️  AI Intrusion Detection System")

# Function to generate synthetic intrusion data
def generate_synthetic_data(n_samples=1000, n_classes=2):
    """Generate synthetic network intrusion data"""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,  # Fixed number of features for simplicity
        n_informative=8,
        n_redundant=2,
        n_classes=n_classes,
        weights=[0.8, 0.2] if n_classes == 2 else None,  # 80% normal, 20% attack for binary classification
        random_state=42
    )
    
    # Create DataFrame with named features
    feature_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
                   'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'count']
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add class column with actual class names
    if n_classes == 2:
        class_names = ['Normal', 'Attack']
    else:
        class_names = ['Normal', 'DOS', 'Probe', 'R2L', 'U2R'][:n_classes]
    
    df['class'] = [class_names[i] for i in y]
    
    # Make protocol_type, service, and flag categorical
    df['protocol_type'] = df['protocol_type'].apply(lambda x: ['tcp', 'udp', 'icmp'][int((x + 3) * 10) % 3])
    df['service'] = df['service'].apply(lambda x: ['http', 'ftp', 'smtp', 'ssh', 'dns'][int((x + 3) * 10) % 5])
    df['flag'] = df['flag'].apply(lambda x: ['SF', 'S0', 'REJ', 'RSTO', 'RSTR'][int((x + 3) * 10) % 5])
    
    return df

# Function to save a dataset locally
def save_dataset_locally(df, filename="intrusion_dataset.csv"):
    """Save dataset to a local file"""
    try:
        df.to_csv(filename, index=False)
        return True, f"Dataset successfully saved to {filename}"
    except Exception as e:
        return False, f"Error saving dataset: {str(e)}"

# Function to create a download link
def download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for a file"""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    elif isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    return href

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section", 
    ["About", "Dataset", "Model Training", "Prediction", "Download"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'categorical_cols' not in st.session_state:
    st.session_state.categorical_cols = None
if 'numerical_cols' not in st.session_state:
    st.session_state.numerical_cols = None

# About page
if app_mode == "About":
    st.header("About This Application")
    
    st.markdown("""
    ### What is an Intrusion Detection System?
    
    An Intrusion Detection System (IDS) is a network security technology that monitors network traffic for suspicious activity and issues alerts when such activity is discovered. 
    
    ### How AI Improves Intrusion Detection
    
    AI and machine learning allow intrusion detection systems to:
    - Learn patterns of normal network behavior
    - Identify anomalies that may indicate an attack
    - Adapt to new types of threats
    - Reduce false positives
    
    ### How to Use This App
    
    1. Start in the **Dataset** section to generate or upload data
    2. Move to **Model Training** to train a machine learning model
    3. Test your model in the **Prediction** section
    4. Save your model and data in the **Download** section
    
    This simplified application focuses on the core workflow of building an AI-based intrusion detection system.
    """)

# Dataset page
elif app_mode == "Dataset":
    st.header("Dataset Creation & Exploration")
    
    # Create tabs for generating or uploading data
    data_tab = st.radio("Select data source:", ["Generate Synthetic Data", "Upload CSV"])
    
    if data_tab == "Generate Synthetic Data":
        st.subheader("Generate Synthetic Network Data")
        
        # Simple options for data generation
        n_samples = st.slider("Number of samples:", 100, 5000, 1000, 100)
        n_classes = st.slider("Number of classes:", 2, 5, 2, 1)
        
        class_names = {
            2: ["Normal", "Attack"],
            3: ["Normal", "DOS", "Probe"],
            4: ["Normal", "DOS", "Probe", "R2L"],
            5: ["Normal", "DOS", "Probe", "R2L", "U2R"]
        }
        
        st.write(f"Classes: {', '.join(class_names[n_classes])}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Data"):
                with st.spinner("Generating data..."):
                    df = generate_synthetic_data(n_samples, n_classes)
                    st.session_state.data = df
                    
                    # Identify categorical and numerical columns
                    categorical_cols = ['protocol_type', 'service', 'flag']
                    numerical_cols = [col for col in df.columns if col not in categorical_cols and col != 'class']
                    
                    st.session_state.categorical_cols = categorical_cols
                    st.session_state.numerical_cols = numerical_cols
                    
                    st.success(f"Generated {n_samples} samples with {n_classes} classes!")
                    st.dataframe(df.head())
                    
                    # Show class distribution
                    st.subheader("Class Distribution")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df['class'].value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
        
        with col2:
            if st.session_state.data is not None:
                # Add option to save dataset locally
                st.subheader("Save Generated Dataset")
                save_filename = st.text_input("Filename:", "intrusion_dataset.csv")
                
                if st.button("Save Dataset Locally"):
                    success, message = save_dataset_locally(st.session_state.data, save_filename)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                
                # Add download link
                st.markdown("### Download Dataset")
                st.markdown(
                    download_link(st.session_state.data, 'intrusion_dataset.csv', 'Download Dataset as CSV'),
                    unsafe_allow_html=True
                )
    
    else:  # Upload CSV
        st.subheader("Upload Your Dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check for class column
                if 'class' not in df.columns and 'label' not in df.columns:
                    st.warning("Your dataset doesn't have a column named 'class' or 'label'.")
                    class_col = st.selectbox("Select the column containing the target labels:", df.columns)
                    df = df.rename(columns={class_col: 'class'})
                elif 'label' in df.columns:
                    df = df.rename(columns={'label': 'class'})
                
                # Identify categorical and numerical columns
                categorical_cols = []
                for col in df.columns:
                    if col != 'class' and df[col].dtype == 'object':
                        categorical_cols.append(col)
                
                numerical_cols = [col for col in df.columns if col not in categorical_cols and col != 'class']
                
                st.session_state.data = df
                st.session_state.categorical_cols = categorical_cols
                st.session_state.numerical_cols = numerical_cols
                
                st.success(f"Uploaded dataset with {df.shape[0]} samples and {df.shape[1]} features!")
                st.dataframe(df.head())
                
                # Show class distribution
                if 'class' in df.columns:
                    st.subheader("Class Distribution")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df['class'].value_counts().plot(kind='bar', ax=ax)
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error loading the dataset: {e}")
    
    # Data exploration (if data is loaded)
    if st.session_state.data is not None:
        df = st.session_state.data
        
        with st.expander("Dataset Statistics"):
            st.write("Dataset Shape:", df.shape)
            st.write("Data Types:")
            st.write(df.dtypes)
            
            # Show basic stats for numerical columns
            numerical_cols = st.session_state.numerical_cols
            if len(numerical_cols) > 0:
                st.write("Numerical Statistics:")
                st.write(df[numerical_cols].describe())

# Model Training page
elif app_mode == "Model Training":
    st.header("Model Training & Evaluation")
    
    if st.session_state.data is None:
        st.warning("Please create or upload a dataset first!")
        if st.button("Go to Dataset Section"):
            app_mode = "Dataset"
            st.experimental_rerun()
    else:
        df = st.session_state.data
        
        # Simple preprocessing
        st.subheader("Data Preprocessing")
        
        # Get categorical and numerical columns
        categorical_cols = st.session_state.categorical_cols
        numerical_cols = st.session_state.numerical_cols
        
        st.write(f"Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}")
        st.write(f"Numerical columns: {', '.join(numerical_cols) if numerical_cols else 'None'}")
        
        # Simple train-test split
        test_size = st.slider("Test set size (%):", 10, 50, 20, 5) / 100
        
        # Model selection
        st.subheader("Select Model")
        model_type = st.selectbox("Choose a model:", ["Random Forest", "Logistic Regression"])
        
        # Simple model parameters
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of trees:", 10, 200, 100, 10)
        else:  # Logistic Regression
            C = st.slider("Regularization strength (C):", 0.1, 10.0, 1.0, 0.1)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Preprocessing data and training model..."):
                # Preprocessing
                X = df.drop('class', axis=1).copy()
                y = df['class'].copy()
                
                # Encode categorical variables
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                
                # Encode target if needed
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.session_state.label_encoder = le
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Train model
                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                else:  # Logistic Regression
                    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
                
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.session_state.model_name = model_type
                
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
                
                # Show metrics
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Accuracy: {accuracy:.4f}")
                    st.write(f"Precision: {precision:.4f}")
                    st.write(f"Recall: {recall:.4f}")
                    st.write(f"F1 Score: {f1:.4f}")
                
                with col2:
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred))
                
                # Feature Importance (for Random Forest)
                if model_type == "Random Forest":
                    st.subheader("Feature Importance")
                    
                    feature_imp = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_imp[:10], ax=ax)
                    ax.set_title('Top 10 Feature Importance')
                    st.pyplot(fig)

# Prediction page
elif app_mode == "Prediction":
    st.header("Live Prediction")
    
    if st.session_state.model is None:
        st.warning("Please train a model first!")
        if st.button("Go to Model Training"):
            app_mode = "Model Training"
            st.experimental_rerun()
    else:
        model = st.session_state.model
        model_name = st.session_state.model_name
        
        st.write(f"Using {model_name} model for predictions")
        
        # Create tabs for different prediction methods
        pred_tab = st.radio("Select prediction method:", 
                          ["Predefined Examples", "Test Sample", "Manual Input", "Batch Prediction (CSV)"])
        
        # 1. PREDEFINED EXAMPLES - New option
        if pred_tab == "Predefined Examples":
            st.subheader("Select a predefined network traffic example")
            
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
            
            # Dropdown for selecting example
            selected_example = st.selectbox(
                "Choose a predefined example:",
                list(predefined_examples.keys())
            )
            
            # Display the selected example
            example_data = predefined_examples[selected_example]
            st.write("Example details:")
            st.json(example_data)
            
            # Make prediction button
            if st.button("Predict", key="predict_example"):
                # Prepare input as DataFrame
                input_df = pd.DataFrame([example_data])
                
                # Handle categorical variables
                categorical_cols = st.session_state.categorical_cols
                for col in categorical_cols:
                    if col in input_df.columns:
                        # Map categorical variables to encoded values
                        if col == 'protocol_type':
                            mapping = {'tcp': 0, 'udp': 1, 'icmp': 2}
                        elif col == 'service':
                            mapping = {'http': 0, 'ftp': 1, 'smtp': 2, 'ssh': 3, 'dns': 4}
                        elif col == 'flag':
                            mapping = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTO': 3, 'RSTR': 4}
                        else:
                            # For other categorical columns, use a simpler mapping
                            unique_vals = list(set(st.session_state.data[col]))
                            mapping = {val: idx for idx, val in enumerate(unique_vals)}
                        
                        input_df[col] = input_df[col].map(mapping)
                
                # Make prediction
                pred = model.predict(input_df)[0]
                
                # Convert prediction to class name if we have a label encoder
                if st.session_state.label_encoder is not None:
                    pred_class = st.session_state.label_encoder.inverse_transform([pred])[0]
                else:
                    pred_class = pred
                
                # Determine if it's an attack
                is_attack = pred_class != 'Normal' if isinstance(pred_class, str) else pred_class != 0
                
                # Show prediction result with styling
                if is_attack:
                    st.markdown("""
                    <div style="background-color: #FFCCCC; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="color: #CC0000;">⚠️ ALERT: Potential intrusion detected!</h3>
                        <p>The model has classified this traffic as potentially malicious.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"Classification: {pred_class}")
                else:
                    st.markdown("""
                    <div style="background-color: #CCFFCC; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="color: #007700;">✅ Normal traffic detected</h3>
                        <p>The model has classified this traffic as normal network activity.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show prediction probabilities if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    class_names = st.session_state.label_encoder.classes_ if st.session_state.label_encoder else [f"Class {i}" for i in range(len(probs))]
                    
                    # Create table with class probabilities
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probs
                    })
                    
                    st.subheader("Prediction Confidence")
                    
                    # Format probabilities as percentages
                    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                    st.table(prob_df)
        
        # 2. TEST SAMPLE OPTION - Original code with fixes
        elif pred_tab == "Test Sample":
            if st.session_state.X_test is not None:
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Select a sample from test set
                sample_idx = st.slider("Select test sample index:", 0, len(X_test)-1, 0)
                
                # Display sample - Handle both DataFrame and NumPy array
                sample = X_test.iloc[sample_idx] if hasattr(X_test, 'iloc') else X_test[sample_idx]
                st.write("Selected sample:")
                st.dataframe(pd.DataFrame(sample).T)
                
                # Make prediction
                if st.button("Predict", key="predict_test"):
                    pred = model.predict([sample])[0]
                    
                    # Get actual class - Fixed line
                    actual = y_test[sample_idx]
                    
                    # Convert prediction to class name if we have a label encoder
                    if st.session_state.label_encoder is not None:
                        pred_class = st.session_state.label_encoder.inverse_transform([pred])[0]
                        actual_class = st.session_state.label_encoder.inverse_transform([actual])[0]
                    else:
                        pred_class = pred
                        actual_class = actual
                    
                    # Determine if it's an attack
                    is_attack = pred_class != 'Normal' if isinstance(pred_class, str) else pred_class != 0
                    
                    # Show prediction result
                    if is_attack:
                        st.error(f"⚠️ **ALERT: Potential intrusion detected!**")
                        st.write(f"Classification: {pred_class}")
                    else:
                        st.success(f"✅ **Normal traffic detected**")
                    
                    st.write(f"Actual class: {actual_class}")
                    
                    # Show prediction probabilities if available
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba([sample])[0]
                        class_names = st.session_state.label_encoder.classes_ if st.session_state.label_encoder else [f"Class {i}" for i in range(len(probs))]
                        
                        prob_df = pd.DataFrame({
                            'Class': class_names,
                            'Probability': probs
                        })
                        
                        # Plot probabilities
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x='Class', y='Probability', data=prob_df, ax=ax)
                        ax.set_title('Prediction Probabilities')
                        st.pyplot(fig)
            else:
                st.warning("No test data available. Please train a model first.")
        
        # 3. MANUAL INPUT OPTION - Original code, kept for flexibility
        elif pred_tab == "Manual Input":
            st.write("Enter feature values manually:")
            
            if st.session_state.X_train is not None:
                X_train = st.session_state.X_train
                
                # Get categorical and numerical columns
                categorical_cols = st.session_state.categorical_cols
                
                # Create input fields for each feature
                manual_input = {}
                
                # Use two columns for better layout
                col1, col2 = st.columns(2)
                
                # Add inputs column by column alternating
                cols = [col1, col2]
                feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(X_train.shape[1])]
                
                for i, col in enumerate(feature_names):
                    cur_col = cols[i % 2]
                    with cur_col:
                        if col in categorical_cols:
                            # Special handling for known categorical columns
                            if col == 'protocol_type':
                                manual_input[col] = st.selectbox(f"{col}:", ["tcp", "udp", "icmp"])
                            elif col == 'service':
                                manual_input[col] = st.selectbox(f"{col}:", ["http", "ftp", "smtp", "ssh", "dns"])
                            elif col == 'flag':
                                manual_input[col] = st.selectbox(f"{col}:", ["SF", "S0", "REJ", "RSTO", "RSTR"])
                            else:
                                manual_input[col] = st.text_input(f"{col}:", key=f"input_{col}")
                        else:
                            # Numerical features - Get min/max safely from both pandas and NumPy
                            if hasattr(X_train, 'min'):
                                min_val = float(X_train[col].min())
                                max_val = float(X_train[col].max())
                            else:
                                col_idx = list(feature_names).index(col)
                                min_val = float(np.min(X_train[:, col_idx]))
                                max_val = float(np.max(X_train[:, col_idx]))
                            
                            default_val = float((min_val + max_val) / 2)
                            manual_input[col] = st.number_input(f"{col}:", 
                                                              min_value=min_val, 
                                                              max_value=max_val, 
                                                              value=default_val,
                                                              key=f"input_{col}")
                
                # Make prediction
                if st.button("Predict", key="predict_manual"):
                    # Prepare input as DataFrame
                    input_df = pd.DataFrame([manual_input])
                    
                    # Handle categorical variables
                    for col in categorical_cols:
                        if col in input_df.columns:
                            # Map categorical variables to encoded values
                            if col == 'protocol_type':
                                mapping = {'tcp': 0, 'udp': 1, 'icmp': 2}
                            elif col == 'service':
                                mapping = {'http': 0, 'ftp': 1, 'smtp': 2, 'ssh': 3, 'dns': 4}
                            elif col == 'flag':
                                mapping = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTO': 3, 'RSTR': 4}
                            else:
                                # For other categorical columns, use a simpler mapping
                                unique_vals = list(set(st.session_state.data[col]))
                                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                            
                            input_df[col] = input_df[col].map(mapping)
                    
                    # Make prediction
                    pred = model.predict(input_df)[0]
                    
                    # Convert prediction to class name if we have a label encoder
                    if st.session_state.label_encoder is not None:
                        pred_class = st.session_state.label_encoder.inverse_transform([pred])[0]
                    else:
                        pred_class = pred
                    
                    # Determine if it's an attack
                    is_attack = pred_class != 'Normal' if isinstance(pred_class, str) else pred_class != 0
                    
                    # Show prediction result with styling
                    if is_attack:
                        st.markdown("""
                        <div style="background-color: #FFCCCC; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                            <h3 style="color: #CC0000;">⚠️ ALERT: Potential intrusion detected!</h3>
                            <p>The model has classified this traffic as potentially malicious.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(f"Classification: {pred_class}")
                    else:
                        st.markdown("""
                        <div style="background-color: #CCFFCC; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                            <h3 style="color: #007700;">✅ Normal traffic detected</h3>
                            <p>The model has classified this traffic as normal network activity.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show prediction probabilities if available
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_df)[0]
                        class_names = st.session_state.label_encoder.classes_ if st.session_state.label_encoder else [f"Class {i}" for i in range(len(probs))]
                        
                        # Create table with class probabilities
                        prob_df = pd.DataFrame({
                            'Class': class_names,
                            'Probability': probs
                        })
                        
                        st.subheader("Prediction Confidence")
                        
                        # Format probabilities as percentages
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                        st.table(prob_df)
            else:
                st.warning("No training data available. Please train a model first.")
        
        # 4. BATCH PREDICTION - New option for CSV upload
        elif pred_tab == "Batch Prediction (CSV)":
            st.subheader("Batch Prediction on CSV File")
            
            st.write("""
            Upload a CSV file with network traffic data for batch prediction. 
            The CSV should contain the same features used during model training.
            """)
            
            # Sample CSV template download
            if st.button("Download CSV Template"):
                # Create a sample CSV template based on the features
                if st.session_state.data is not None:
                    features = [col for col in st.session_state.data.columns if col != 'class']
                    sample_df = pd.DataFrame(columns=features)
                    
                    # Add a single sample row with default values
                    sample_row = {}
                    for col in features:
                        if col in st.session_state.categorical_cols:
                            if col == 'protocol_type':
                                sample_row[col] = 'tcp'
                            elif col == 'service':
                                sample_row[col] = 'http'
                            elif col == 'flag':
                                sample_row[col] = 'SF'
                            else:
                                sample_row[col] = 'value'
                        else:
                            sample_row[col] = 0.0
                    
                    sample_df = pd.concat([sample_df, pd.DataFrame([sample_row])], ignore_index=True)
                    
                    # Create download link
                    csv = sample_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:text/csv;base64,{b64}" download="prediction_template.csv">Download CSV Template</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("No dataset information available. Please create or upload a dataset first.")
            
            # Upload CSV for prediction
            uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Load the CSV
                    prediction_df = pd.read_csv(uploaded_file)
                    
                    # Show preview
                    st.write("Data Preview:")
                    st.dataframe(prediction_df.head())
                    
                    # Check for required columns
                    required_cols = [col for col in st.session_state.X_train.columns] if hasattr(st.session_state.X_train, 'columns') else []
                    
                    missing_cols = [col for col in required_cols if col not in prediction_df.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        if st.button("Run Batch Prediction"):
                            with st.spinner("Processing batch predictions..."):
                                # Preprocess data (encode categorical variables)
                                pred_data = prediction_df.copy()
                                
                                # Handle categorical columns
                                categorical_cols = st.session_state.categorical_cols
                                for col in categorical_cols:
                                    if col in pred_data.columns:
                                        # Map categorical variables to encoded values
                                        if col == 'protocol_type':
                                            mapping = {'tcp': 0, 'udp': 1, 'icmp': 2}
                                        elif col == 'service':
                                            mapping = {'http': 0, 'ftp': 1, 'smtp': 2, 'ssh': 3, 'dns': 4}
                                        elif col == 'flag':
                                            mapping = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTO': 3, 'RSTR': 4}
                                        else:
                                            # For other categorical columns, use a simpler mapping
                                            unique_vals = list(set(st.session_state.data[col]))
                                            mapping = {val: idx for idx, val in enumerate(unique_vals)}
                                        
                                        pred_data[col] = pred_data[col].map(mapping)
                                
                                # Make predictions
                                predictions = model.predict(pred_data)
                                
                                # Get prediction probabilities if available
                                if hasattr(model, "predict_proba"):
                                    probabilities = model.predict_proba(pred_data)
                                    max_probs = np.max(probabilities, axis=1)
                                else:
                                    max_probs = [None] * len(predictions)
                                
                                # Convert predictions to class names if we have a label encoder
                                if st.session_state.label_encoder is not None:
                                    pred_classes = st.session_state.label_encoder.inverse_transform(predictions)
                                else:
                                    pred_classes = predictions
                                
                                # Create results DataFrame
                                results_df = prediction_df.copy()
                                results_df['Predicted_Class'] = pred_classes
                                results_df['Confidence'] = [f"{p:.2%}" if p is not None else "N/A" for p in max_probs]
                                results_df['Is_Attack'] = [cls != 'Normal' if isinstance(cls, str) else cls != 0 for cls in pred_classes]
                                
                                # Display results
                                st.subheader("Prediction Results")
                                
                                # Use custom styling for the results table
                                def highlight_attacks(val):
                                    if val == True:
                                        return 'background-color: #FFCCCC'
                                    else:
                                        return 'background-color: #CCFFCC'
                                
                                # Apply styling to dataframe
                                styled_results = results_df.style.applymap(
                                    highlight_attacks, subset=['Is_Attack']
                                )
                                
                                st.dataframe(styled_results)
                                
                                # Summary of results
                                attack_count = sum(results_df['Is_Attack'])
                                normal_count = len(results_df) - attack_count
                                
                                st.subheader("Summary")
                                st.write(f"Total samples: {len(results_df)}")
                                st.write(f"Normal traffic detected: {normal_count}")
                                st.write(f"Potential attacks detected: {attack_count}")
                                
                                # Create download link for results
                                csv = results_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:text/csv;base64,{b64}" download="prediction_results.csv">Download Results CSV</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                except Exception as e:
                    st.error(f"Error processing the uploaded file: {str(e)}")

# Download page
elif app_mode == "Download":
    st.header("Download Resources")
    
    if st.session_state.data is not None:
        st.subheader("Download Dataset")
        
        # Download original dataset
        st.markdown(
            download_link(st.session_state.data, 'intrusion_dataset.csv', 'Download Dataset (CSV)'),
            unsafe_allow_html=True
        )
        
        # Option to save dataset locally
        st.subheader("Save Dataset Locally")
        save_filename = st.text_input("Filename:", "intrusion_dataset.csv")
        
        if st.button("Save Dataset to Local Folder"):
            success, message = save_dataset_locally(st.session_state.data, save_filename)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    if st.session_state.model is not None:
        st.subheader("Download Trained Model")
        
        # Serialize model
        model_pkl = pickle.dumps(st.session_state.model)
        
        # Create download link
        st.markdown(
            download_link(model_pkl, 'intrusion_model.pkl', 'Download Model (pickle)'),
            unsafe_allow_html=True
        )
        
        # Save model locally
        st.subheader("Save Model Locally")
        model_filename = st.text_input("Model filename:", "intrusion_model.pkl")
        
        if st.button("Save Model to Local Folder"):
            try:
                with open(model_filename, 'wb') as f:
                    pickle.dump(st.session_state.model, f)
                st.success(f"Model successfully saved to {model_filename}")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
        
        # Instructions for using the model
        with st.expander("How to use the downloaded model"):
            st.code('''
import pickle
import pandas as pd
import numpy as np

# Load the model
with open('intrusion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example: Load a CSV file with new network traffic data
new_data = pd.read_csv('new_traffic.csv')

# Preprocess the data (ensure same preprocessing as training)
# 1. Encode categorical columns (protocol_type, service, flag)
# 2. Ensure all required columns are present

# Make predictions
predictions = model.predict(new_data)

# If you need prediction probabilities
probabilities = model.predict_proba(new_data)
            ''')

# Add a script to generate datasets (can be executed separately)
if app_mode == "Download" and st.session_state.data is None:
    st.subheader("Generate Datasets for Later Use")
    
    st.write("""
    If you want to generate datasets without running the full application, 
    you can use the script below. Save it as a .py file and run it to generate 
    CSV files with synthetic network intrusion data.
    """)
    
    with st.expander("Dataset Generator Script"):
        st.code('''
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_synthetic_data(n_samples=1000, n_classes=2, filename="intrusion_dataset.csv"):
    """Generate synthetic network intrusion data and save to CSV"""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,  # Fixed number of features for simplicity
        n_informative=8,
        n_redundant=2,
        n_classes=n_classes,
        weights=[0.8, 0.2] if n_classes == 2 else None,  # 80% normal, 20% attack
        random_state=42
    )
    
    # Create DataFrame with named features
    feature_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
                   'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'count']
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add class column with actual class names
    if n_classes == 2:
        class_names = ['Normal', 'Attack']
    else:
        class_names = ['Normal', 'DOS', 'Probe', 'R2L', 'U2R'][:n_classes]
    
    df['class'] = [class_names[i] for i in y]
    
    # Make protocol_type, service, and flag categorical
    df['protocol_type'] = df['protocol_type'].apply(lambda x: ['tcp', 'udp', 'icmp'][int((x + 3) * 10) % 3])
    df['service'] = df['service'].apply(lambda x: ['http', 'ftp', 'smtp', 'ssh', 'dns'][int((x + 3) * 10) % 5])
    df['flag'] = df['flag'].apply(lambda x: ['SF', 'S0', 'REJ', 'RSTO', 'RSTR'][int((x + 3) * 10) % 5])
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    print(f"Generated {n_samples} samples with {n_classes} classes")
    
    return df

# Generate binary classification dataset (Normal vs Attack)
generate_synthetic_data(n_samples=1000, n_classes=2, filename="binary_intrusion_dataset.csv")

# Generate multi-class dataset (Normal, DOS, Probe, R2L, U2R)
generate_synthetic_data(n_samples=2000, n_classes=5, filename="multiclass_intrusion_dataset.csv")
        ''')
        
        st.write("This will create two files:")
        st.write("1. `binary_intrusion_dataset.csv` - A dataset with Normal and Attack classes")
        st.write("2. `multiclass_intrusion_dataset.csv` - A dataset with 5 different classes of network traffic")

