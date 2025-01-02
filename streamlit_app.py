import streamlit as st
from tensorflow.keras.models import load_model, Model
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load the model
model = load_model("C:\\Users\\hp\\Downloads\\model.h5")

# Class labels
class_labels = [
    "Mild Demented",
    "Moderate Demented",
    "Normal",
    "Very Mild Dementia",
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor"
]

# Grad-CAM functions
def preprocess_image(image, target_size=(224, 224)):
    img = np.array(image.resize(target_size)) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_gradcam(model, image, last_conv_layer_name='conv5_block16_concat', pred_index=None):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None. Check the model's last conv layer.")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    
    if tf.reduce_max(heatmap) == 0:
        raise ValueError("Heatmap contains only zero values. Check input or model.")
    
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_image, alpha=0.2, colormap=cv2.COLORMAP_JET):
    img = np.array(original_image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# Streamlit UI
st.title("Brain MRI Diagnosis with Grad-CAM")
st.sidebar.header("Upload MRI Scan")

uploaded_file = st.sidebar.file_uploader("Choose an MRI Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image and predict
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display prediction results
    predicted_class = class_labels[predicted_class_idx]
    st.markdown(f"### Prediction: {predicted_class}")
    if confidence > 0.85:
        st.markdown(
        f"<h1 style='font-size:18px;'>Confidence: {confidence*100:.2f}%</h1>",
        unsafe_allow_html=True
        )



    # Generate Grad-CAM heatmap for relevant classes
    relevant_classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    if predicted_class in relevant_classes :
        st.markdown("### Grad-CAM Heatmap")
        heatmap = generate_gradcam(model, img_array, last_conv_layer_name='conv5_block16_concat', pred_index=predicted_class_idx)
        superimposed_img = overlay_heatmap(heatmap, image)
        st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)
    else:
        st.info("Grad-CAM visualization is not available for the predicted class.")
else:
    st.sidebar.write("Please upload an MRI image to proceed.")
