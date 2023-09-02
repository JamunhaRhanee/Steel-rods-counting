import streamlit as st
import numpy as np
import torch


# Load your pre-trained YOLOv8 model 
model_weights_path = 'C:/Users/Jamunha Rhanee/Downloads/best.pt'
model = torch.load(model_weights_path, map_location=torch.device('cpu'))


# Categories of rod thicknesses
thickness_categories = {0: "16mm", 1: "8mm", 2: "32mm"}

def count_rods(image):
    # Perform inference using the YOLOv8 model
    with torch.no_grad():
        results = model(image)

    # Count rods based on their thickness categories
    rod_counts = {category: 0 for category in thickness_categories.values()}
    for detection in results.pred[0]:
        class_idx = int(detection['label'])
        if class_idx in thickness_categories:
            rod_counts[thickness_categories[class_idx]] += 1

    return rod_counts

st.title("Steel Rod Counting")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    rod_counts = count_rods(image)

    st.write("Rod Counts:")
    for category, count in rod_counts.items():
        st.write(f"{category}: {count}")
        
if __name__ == "__model__":
    model()
