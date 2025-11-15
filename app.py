# app.py
import streamlit as st
import json
import pandas as pd
from PIL import Image

from utils import pil_from_streamlit_uploaded
from ocr_pipeline import (
    detect_handwriting_block,
    enhance_handwriting,
    run_ocr_lines,
    extract_structured_from_text
)

# --------------------------
# Streamlit page setup
# --------------------------
st.set_page_config(page_title="Prescribed", layout="wide")
st.title("🧠 Prescribed")

# --------------------------
# Initialize session state
# --------------------------
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

if "structured" not in st.session_state:
    st.session_state.structured = []

if "meds_list" not in st.session_state:
    st.session_state.meds_list = []

# --------------------------
# Cache OCR processing
# --------------------------
@st.cache_data(show_spinner=False)
def cached_ocr(pil_img):
    cropped = detect_handwriting_block(pil_img)
    enhanced = enhance_handwriting(cropped)
    text = run_ocr_lines(enhanced)
    structured = extract_structured_from_text(text)
    return text, structured

# --------------------------
# File uploader
# --------------------------
uploaded_file = st.file_uploader(
    "Upload prescription image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------
# Run OCR button
# --------------------------
if uploaded_file and st.button("Run OCR"):
    pil_img = pil_from_streamlit_uploaded(uploaded_file)
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing prescription image..."):
        raw_text, structured = cached_ocr(pil_img)
        st.session_state.ocr_text = raw_text
        st.session_state.structured = structured

        # Update meds list
        for r in structured:
            if r not in st.session_state.meds_list:
                st.session_state.meds_list.append(r)

# --------------------------
# Display OCR output
# --------------------------
st.subheader("Raw OCR Output:")
st.text_area("Raw Text", value=st.session_state.ocr_text, height=200)

# --------------------------
# Editable structured lines
# --------------------------
st.subheader("Extracted Prescription Lines")
df_rows = []
for i, r in enumerate(st.session_state.structured):
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    raw_t = col1.text_input(f"Raw {i+1}", value=r["raw"], key=f"raw_{i}")
    med_t = col2.text_input(f"Medication {i+1}", value=r["med_match"], key=f"med_{i}")
    dose_t = col3.text_input(f"Dose {i+1}", value=str(r["dose"] or ""), key=f"dose_{i}")
    freq_t = col4.text_input(f"Frequency {i+1}", value=str(r["frequency"] or ""), key=f"freq_{i}")

    df_rows.append({
        "raw": raw_t,
        "med": med_t,
        "dose": dose_t,
        "frequency": freq_t
    })

# --------------------------
# Manual line addition form
# --------------------------
st.markdown("---")
with st.form("manual_line_form"):
    new_line = st.text_input("Add a manual prescription line:")
    submitted = st.form_submit_button("Add Line")

    if submitted and new_line.strip():
        new_struct = extract_structured_from_text(new_line)
        st.session_state.structured.extend(new_struct)
        st.session_state.ocr_text += "\n" + new_line

        # Update meds list
        for r in new_struct:
            if r not in st.session_state.meds_list:
                st.session_state.meds_list.append(r)

# --------------------------
# Display meds list (persistent)
# --------------------------
st.subheader("Detected Medicines")
st.json(st.session_state.meds_list)

# --------------------------
# Export
# --------------------------
st.subheader("Export")
st.download_button(
    "Download JSON",
    data=json.dumps({"lines": df_rows}, indent=2),
    file_name="prescription.json",
    mime="application/json"
)

df = pd.DataFrame(df_rows)
st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="prescription.csv",
    mime="text/csv"
)
