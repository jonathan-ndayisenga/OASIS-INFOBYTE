import streamlit as st
import joblib
import numpy as np
from PIL import Image  # For displaying the iris image

# Set page config
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌼",
    layout="centered"
)

# Load model and class names
try:
    model = joblib.load('iris_model_fixed.pkl')
    CLASS_NAMES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    # Verify model has correct number of features
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != 4:
        st.error(f"⚠️ Model expects {model.n_features_in_} features, but app provides 4. Please retrain your model!")
        st.stop()
        
except FileNotFoundError:
    st.error("Model file 'iris_model_fixed.pkl' not found. Please train the model first.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# App Header
st.title('🌼 Iris Flower Classifier')
st.markdown("Predict the species based on flower measurements")

# Sidebar with inputs
with st.sidebar:
    st.header("🔍 Input Features")
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 5.0, 3.5, 0.1)
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2, 0.1)

# Create input array (ensure correct feature order)
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction Section
st.header("📊 Prediction Results")

try:
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Species", CLASS_NAMES[prediction])
        
        # Show confidence as a progress bar
        confidence = probabilities.max()
        st.progress(int(confidence * 100))
        st.caption(f"Confidence: {confidence:.1%}")
        
    with col2:
        # Display probabilities for all classes
        st.write("**Probabilities:**")
        for i, prob in enumerate(probabilities):
            st.write(f"{CLASS_NAMES[i]}: {prob:.1%}")
    
    # Visual feedback
    if confidence > 0.9:
        st.balloons()
    
except Exception as e:
    st.error(f"❌ Prediction failed: {str(e)}")
    st.write("Please check your model and input data format.")

# Add some educational content
with st.expander("ℹ️ About Iris Flowers"):
    st.write("""
    The Iris dataset contains measurements for three species:
    - **Iris-setosa**
    - **Iris-versicolor**
    - **Iris-virginica**
    
    Sepal = The outer protective leaf
    Petal = The inner colorful leaf
    """)
    try:
        img = Image.open('iris_photo.jpg')  # Optional: add an iris image
        st.image(img, caption='Iris Flower Species')
    except:
        pass

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")