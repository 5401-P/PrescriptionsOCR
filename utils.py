# utils.py
import io
from PIL import Image

def pil_from_streamlit_uploaded(uploaded_file):
    """Convert Streamlit uploaded file to a PIL RGB image."""
    return Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")