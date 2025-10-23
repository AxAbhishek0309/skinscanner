# Run this after training your model to save it for Streamlit
# Add this to the end of your convolution.py file

# Save the trained model
cnn.save('skin_cancer_model.h5')
print("Model saved as 'skin_cancer_model.h5'")

# Optional: Save model in SavedModel format (recommended for production)
cnn.save('skin_cancer_model_savedmodel')
print("Model also saved in SavedModel format")