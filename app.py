




# requirements.txt should include:
# streamlit==1.35.0
# scikit-learn==1.5.2
# python-docx==1.1.0
# PyPDF2==3.0.1
# gdown==5.1.0
# numpy==1.26.4
# pandas==2.1.3

import streamlit as st
import pickle
import docx
import PyPDF2
import re
import os
import gdown
from io import BytesIO

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        # 1. Download clf.pkl from Google Drive if missing
        if not os.path.exists('clf.pkl'):
            gdown.download(
                'https://drive.google.com/uc?id=1vQi73uuWL1X-UPumNjbf1363GElghKxY',
                'clf.pkl',
                quiet=False
            )
        
        # 2. Load all models
        return {
            'svc_model': pickle.load(open('clf.pkl', 'rb')),
            'tfidf': pickle.load(open('tfidf.pkl', 'rb')),
            'encoder': pickle.load(open('encoder.pkl', 'rb'))
        }
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.error("Required files:")
        st.error("1. clf.pkl (from Google Drive)")
        st.error("2. tfidf.pkl (in GitHub repo)")
        st.error("3. encoder.pkl (in GitHub repo)")
        st.stop()

# --- Text Cleaning ---
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# --- File Extractors ---
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
        return ''.join([page.extract_text() or '' for page in pdf_reader.pages])
    except Exception as e:
        st.error(f"PDF error: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(BytesIO(file.read()))
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"DOCX error: {str(e)}")
        return ""

def extract_text_from_txt(file):
    try:
        file.seek(0)
        return str(file.read(), 'utf-8', errors='ignore')
    except Exception as e:
        st.error(f"TXT error: {str(e)}")
        return ""

# --- Main App ---
def main():
    st.set_page_config(page_title="Resume Classifier", layout="wide")
    st.title("📄 Resume Domain Classifier")
    
    # Load models
    models = load_models()
    
    # File upload
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resume Content")
            try:
                # Extract text
                if uploaded_file.name.endswith('.pdf'):
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.endswith('.docx'):
                    text = extract_text_from_docx(uploaded_file)
                else:
                    text = extract_text_from_txt(uploaded_file)
                
                # Show extracted text
                if st.toggle("Show raw text"):
                    st.text_area("Content", text, height=300)
                
                # Process if text exists
                if text.strip():
                    with col2:
                        st.subheader("Domain")
                        cleaned = cleanResume(text)
                        vector = models['tfidf'].transform([cleaned]).toarray()
                        pred = models['encoder'].inverse_transform(
                            models['svc_model'].predict(vector)
                        )[0]
                        st.success(f"**Category this resume belongs to :** {pred}")
                else:
                    st.warning("No text could be extracted")
                    
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")

if __name__ == "__main__":
    main()
