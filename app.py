import streamlit as st
from PIL import Image
import uuid
from storage_utils import save_document
from gemini_utils import extract_text_with_gemini, translate_with_gemini

st.set_page_config(page_title="ScriptRevive", layout="wide")
st.title("🧾 ScriptRevive")
st.caption("Digitize, Translate, and Preserve Historical Handwritten Regional Documents")

# File Upload
uploaded_file = st.file_uploader("Upload a historical document", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", use_container_width=True)

    # Step 1: OCR
    st.subheader("Step 1: OCR")
    ocr_method = st.radio("Choose OCR Method", ["Tesseract", "Gemini Vision"])

    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""

    if st.button("Run OCR"):
        if ocr_method == "Tesseract":
            lang = st.selectbox("Select language for OCR", ["eng", "hin", "tam", "tel", "kan"])
            st.session_state.extracted_text = extract_text_with_gemini(image, lang=lang)
        elif ocr_method == "Gemini Vision":
            st.info("Using Gemini Vision API...")
            st.session_state.extracted_text = extract_text_with_gemini(image)

    if st.session_state.extracted_text:
        st.text_area("Extracted Text", value=st.session_state.extracted_text, height=200)

        # Step 2: Translation
        st.subheader("Step 2: Translate")

        if "translated_text" not in st.session_state:
            st.session_state.translated_text = ""

        if st.button("Translate to English"):
            st.session_state.translated_text = translate_with_gemini(st.session_state.extracted_text)

        if st.session_state.translated_text:
            st.text_area("Translated Text", value=st.session_state.translated_text, height=200)

            # Step 3: Save
            st.subheader("Step 3: Save")
            if st.button("Save Document"):
                file_id = str(uuid.uuid4())
                save_document(
                    st.session_state.extracted_text,
                    st.session_state.translated_text,
                    filename=file_id
                )
                st.success(f"Saved as {file_id}.json")
