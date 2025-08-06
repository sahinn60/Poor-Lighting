import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def enhance_low_light_image(uploaded_image):
    # Read image bytes and convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original is None:
        st.error("⛔ Could not read the image.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Step 1: Gray Level Thresholding for low-light detection
    threshold_value = 80  # adjust as needed
    _, low_light_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)

    # Step 3: Adaptive Spatial Filtering (Gaussian Blur)
    smoothed = cv2.GaussianBlur(hist_eq, (5, 5), 0)

    # Step 4: Sharpening (Laplacian + Unsharp Masking)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = cv2.addWeighted(smoothed, 1.5, laplacian, -0.5, 0)

    # Step 5: Replace low-light areas with enhanced version
    enhanced_gray = np.where(low_light_mask == 255, sharpened, gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    mask_3ch = cv2.merge([low_light_mask] * 3)
    result = np.where(mask_3ch == 255, enhanced_bgr, original)

    return original, result

# Streamlit UI
st.set_page_config(page_title="📸 Poor Lighting Fixer", layout="centered")
st.title("📸 Poor Lighting Fixer for Mobile Photos")
st.markdown("""
Improve your low-light photos automatically using histogram equalization, adaptive filters, and intelligent enhancement — all in-browser, no installation needed!
""")

uploaded_file = st.file_uploader("📂 Upload a photo (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_img, enhanced_img = enhance_low_light_image(uploaded_file)

    if original_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, channels="BGR", caption="🔹 Original", use_column_width=True)
        with col2:
            st.image(enhanced_img, channels="BGR", caption="✨ Enhanced", use_column_width=True)

        # Download button
        _, buffer = cv2.imencode(".jpg", enhanced_img)
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=buffer.tobytes(),
            file_name="enhanced_output.jpg",
            mime="image/jpeg"
        )
