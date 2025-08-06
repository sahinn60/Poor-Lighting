import streamlit as st
import cv2
import numpy as np

def enhance_low_light_image(uploaded_image):
    # Convert uploaded image to numpy array
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original is None:
        st.error("❌ Could not read the image.")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Detect low-light regions using thresholding
    threshold_value = 80
    _, low_light_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)

    # Gaussian Blur
    smoothed = cv2.GaussianBlur(hist_eq, (5, 5), 0)

    # Laplacian + Unsharp Masking for Sharpening
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = cv2.addWeighted(smoothed, 1.5, laplacian, -0.5, 0)

    # Apply enhanced areas to original
    enhanced_gray = np.where(low_light_mask == 255, sharpened, gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    mask_3ch = cv2.merge([low_light_mask] * 3)
    result = np.where(mask_3ch == 255, enhanced_bgr, original)

    return original, result

# Streamlit UI
st.set_page_config(page_title="📸 Low-Light Photo Enhancer", layout="centered")
st.title("📸 Low-Light Photo Enhancer")
st.markdown("Improve photos taken in poor lighting — especially with mobile cameras!")

uploaded_file = st.file_uploader("📂 Upload a photo (JPEG or PNG)", type=["jpg", "jpeg", "png"])

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
