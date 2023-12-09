import streamlit as st
import time
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('agg')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define selected features globally
selected_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
                      "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"]

# Set dark theme for Streamlit
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1E1E1E;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data()
def load_data():
    # Your existing code
    # Your existing code
    # Load Breast Cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Extract only the selected features
    df_selected_features = df[selected_features]

    return df_selected_features, data.target


@st.cache_data()
def build_model(X_train, y_train, X_test, y_test):
    # Your existing code
    # Your existing code

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and compile the model
    N, D = X_train.shape
    model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(D,)), tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

    return model, r, scaler


# Waiting time for the App to display
with st.spinner('Loading...'):
    time.sleep(10)
st.success('')

# Streamlit App
st.title("Breast Cancer Classification using Deep Learning")


# Load data and build model
df_selected_features, target = load_data()
X_train, X_test, y_train, y_test = train_test_split(df_selected_features, target, test_size=0.33)
model, r, scaler = build_model(X_train, y_train, X_test, y_test)

# Display the dataset
st.subheader("Variable Information")
st.dataframe(df_selected_features)

# Display the target values 0,1
st.markdown('**These are the Ten real-valued features that are computed for each cell nucleus**')
st.markdown('**Based on these features, output results are classified as Benign (Not Cancerous) or Malignant (Cancerous)**')

# Display model training results
st.subheader("Model Training Results")
st.line_chart(pd.DataFrame(r.history))

# Input Section
st.subheader("Prediction Section")

# Example input fields (customize based on your features)
input_features = {}

for feature in selected_features:
    min_val = float(df_selected_features[feature].min())
    max_val = float(df_selected_features[feature].max())
    default_val = float(df_selected_features[feature].mean())
    input_features[feature] = st.slider(f"{feature} Input", min_val, max_val, default_val)


# Make Prediction
prediction_button = st.button("Make Prediction")

if prediction_button:
    # Format the input data for prediction
    input_data = scaler.transform([[input_features[feature] for feature in selected_features]])
    
    # Make prediction
    prediction = model.predict(input_data)[0, 0]
    
    # Display the prediction result
    result_text = "The Tumor is Classified as Malignant (Cancerous)" if prediction > 0.5 else "The Tumor is Classified as Benign (Not Cancerous)"
    st.subheader("Prediction Result:")
    st.write(result_text)
    
    
# Plot accuracy and loss over epochs
st.subheader("Accuracy and Loss Over Epochs")

# Display line chart for accuracy
st.subheader("Train Accuracy vs Validation Accuracy")
st.line_chart(pd.DataFrame({"Train Accuracy": r.history['accuracy'], "Validation Accuracy": r.history['val_accuracy']}))

# Display line chart for loss
st.subheader("Train Loss vs Validation Loss")
st.line_chart(pd.DataFrame({"Train Loss": r.history['loss'], "Validation Loss": r.history['val_loss']}))

