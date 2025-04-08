import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

@st.cache_resource
def load_yolo_model():
    """Loads the YOLO model."""
    model_path = resource_path("best.pt")
    

def process_yolo_image(image, model):
    """Processes a single image using the YOLO model."""
    try:
        original_image_np = np.array(image)
        image_height, image_width = original_image_np.shape[:2]
        scaling_factor = max(image_height, image_width) / 1000
        font_scale = 0.6 * scaling_factor
        thickness = int(2 * scaling_factor)

        results = model.predict(source=original_image_np, save=False)
        objects_detected = any(len(result.boxes) > 0 for result in results)
        annotated_image = original_image_np.copy()

        if not objects_detected:
            no_object_text = "NO OBJECT DETECTED"
            position_no_object = (10, 30)
            cv2.putText(annotated_image, no_object_text, position_no_object, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        else:
            roi_color = (0, 255, 0)
            text_color = (255, 255, 255)
            edb_mask = np.zeros(annotated_image.shape[:2], dtype=np.uint8)

            for result in results:
                labels = result.names
                confidences = result.boxes.conf

                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.xy
                else:
                    masks = []

                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                    label = f"{labels[int(result.boxes.cls[i])]} {confidences[i]:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

                    if masks:
                        mask_points = np.array(masks[i], dtype=np.int32)
                        mask_image = np.zeros(annotated_image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask_image, [mask_points], 255)
                        edb_mask = cv2.bitwise_or(edb_mask, mask_image)
                        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            cv2.drawContours(annotated_image, [contour], -1, roi_color, 2)

            b, g, r = cv2.split(annotated_image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            cl_b, cl_g, cl_r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
            clahe_image = cv2.merge((cl_b, cl_g, cl_r))

            hsv_image = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2HSV)
            lower_bound_black, upper_bound_black = np.array([0, 0, 0]), np.array([180, 255, 83])
            lower_bound_yellow, upper_bound_yellow = np.array([0, 85, 70]), np.array([23, 255, 255])

            mask_black = cv2.inRange(hsv_image, lower_bound_black, upper_bound_black)
            mask_yellow = cv2.inRange(hsv_image, lower_bound_yellow, upper_bound_yellow)

            masked_black = cv2.bitwise_and(mask_black, edb_mask)
            masked_yellow = cv2.bitwise_and(mask_yellow, edb_mask)

            contours_black, _ = cv2.findContours(masked_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(masked_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours_black:
                cv2.drawContours(clahe_image, [contour], -1, (0, 0, 255), 2)  # Red for feathers
            for contour in contours_yellow:
                cv2.drawContours(clahe_image, [contour], -1, (0, 255, 255), 2)  # Yellow for oxidation

            cv2.putText(clahe_image, 'Red: Feathers', (10, int(30 * scaling_factor)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            cv2.putText(clahe_image, 'Yellow: Iron Oxidation / Natural Pigments', (10, int(60 * scaling_factor)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

            return Image.fromarray(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
        return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image

def main():
    st.title("AviNest Analyzer")
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("Choose an option", ["Homepage", "YOLO Image Analysis"])

    yolo_model = load_yolo_model()

    if app_mode == "Homepage":
        st.header("Welcome to AviNest Analyzer")
        st.write("This application uses YOLOv8 for edible bird's nest detection and impurity analysis.")
        st.write("You can upload an image for analysis using the YOLO model.")

        image_paths = [
            resource_path("images1.jpg"),
            resource_path("images2.jpeg")
        ]
        for path in image_paths:
            if os.path.exists(path):
                image = Image.open(path)
                st.image(image, caption=os.path.basename(path), use_container_width=True)
            else:
                st.warning(f"Image not found: {path}")

        st.markdown("---")
        st.subheader("About")
        st.info("This application is for demonstration purposes.")
        st.markdown("© 2024 AviNest. All rights reserved.")

    elif app_mode == "YOLO Image Analysis":
        st.header("YOLO Image Analysis")
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button("Analyze Image"):
                    if yolo_model:
                        with st.spinner("Analyzing image..."):
                            processed_image = process_yolo_image(image, yolo_model)
                        st.subheader("Processed Image")
                        st.image(processed_image, caption="Processed Image with YOLO Detection and Impurity Analysis", use_container_width=True)
                    else:
                        st.error("YOLO model not loaded. Please check the logs.")
            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")

if __name__ == "__main__":
    main()