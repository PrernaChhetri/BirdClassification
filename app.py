import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("D:\BC10.h5")

# Define the Streamlit app
st.title("Bird Species Classification")

# Upload an image for classification
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make predictions on the uploaded image
    predictions = model.predict(img)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # List of class labels
    class_labels = ["Cattle-Egret", "Common-kingfisher", "Common-Myna", "Common-Rosefinch", "Common-Tailorbird", 
                "Coppersmith-Barbet", "Forest-Wagtail", "Hoopoe", "House-Crow", "Indian-Pitta", "Indian-Roller", "Jungle-babbler", 
                "Northern-Lapwing", "Red-Wattled-Lapwing", "Ruddy-Shelduck", "Rufous-Treepie", "White-Breasted-Waterhen", " "]  

    # Get the predicted class name
    predicted_class = class_labels[predicted_class_index]

    # Get the confidence of the prediction
    confidence = predictions[0][predicted_class_index]

    # Define a threshold for confidence
    confidence_threshold = 0.7  # You can adjust this threshold as needed

    # Check if the prediction confidence is below the threshold
    if confidence < confidence_threshold:
        st.write("The model is uncertain about the prediction.")
    else:
        # Display the predicted class name
        st.write(f"The predicted class is: {predicted_class}")
