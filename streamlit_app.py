import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Load model (you'll need to save your trained model first)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('skin_cancer_model.h5')
        return model
    except Exception as e:
        st.error(f"Model not found: {str(e)}")
        st.info("Please ensure the model file 'skin_cancer_model.h5' is in the repository root.")
        return None

def predict_image(img, model):
    """Predict if skin lesion is benign or malignant"""
    # Resize image to match model input
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        result = "Malignant"
        risk_level = "High Risk"
        color = "red"
    else:
        result = "Benign"
        risk_level = "Low Risk" 
        color = "green"
        confidence = 1 - confidence
    
    return result, confidence, risk_level, color

# Main app
def main():
    st.title("🔬 Skin Cancer Detection System")
    st.markdown("Upload a dermatoscopic image to analyze for potential skin cancer")
    
    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.info("""
        This AI model uses Convolutional Neural Networks to classify skin lesions as:
        - **Benign**: Non-cancerous
        - **Malignant**: Potentially cancerous
        
        **Dataset**: ISIC (International Skin Imaging Collaboration)
        """)
        
        st.warning("⚠️ This is for educational purposes only. Always consult a medical professional for proper diagnosis.")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear dermatoscopic image of the skin lesion"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                result, confidence, risk_level, color = predict_image(img, model)
            
            # Display results
            st.markdown(f"### Prediction: :{color}[{result}]")
            st.markdown(f"**Risk Level**: :{color}[{risk_level}]")
            st.markdown(f"**Confidence**: {confidence:.2%}")
            
            # Progress bar for confidence
            st.progress(confidence)
            
            # Additional info based on result
            if result == "Malignant":
                st.error("⚠️ The model suggests this lesion may be malignant. Please consult a dermatologist immediately.")
            else:
                st.success("✅ The model suggests this lesion appears benign. Continue regular skin monitoring.")
    
    # Sample images section
    st.markdown("---")
    st.subheader("📋 Model Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Accuracy", "95%")
    with col2:
        st.metric("Validation Accuracy", "91%")
    with col3:
        st.metric("Image Resolution", "128x128")

if __name__ == "__main__":
    main()