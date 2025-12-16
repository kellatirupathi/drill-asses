import streamlit as st
import pandas as pd
import os
import zipfile
import tempfile
import shutil
import base64
import logging
import json
import re
import toml
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from random import uniform

# --- GLOBAL CONFIGURATION ---
st.set_page_config(page_title="Assessment Q&A Extraction", layout="wide", page_icon="🖼️")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_secret(key):
    """Helper to get secrets from toml or st.secrets"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to .streamlit/secrets.toml in the root directory
    secrets_path = os.path.join(current_dir, ".streamlit", "secrets.toml")
    
    val = None
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r") as f:
                secrets = toml.load(f)
                val = secrets.get(key)
        except:
            pass
    
    if not val:
        try:
            val = st.secrets.get(key)
        except:
            pass
    return val

def get_mistral_api_keys():
    """Returns a list of available Mistral Keys"""
    keys = [key for key in [
        get_secret("MISTRAL_API_KEY_1"), get_secret("MISTRAL_API_KEY_2"),
        get_secret("MISTRAL_API_KEY_3"), get_secret("MISTRAL_API_KEY_4"),
        get_secret("MISTRAL_API_KEY")
    ] if key]
    return keys

# ==============================================================================
#  ASSESSMENT Q&A LOGIC (PDF/Image/ZIP Extraction)
# ==============================================================================

class MistralClientAssessment:
    def __init__(self):
        self.api_keys = get_mistral_api_keys()
        self.chat_endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.ocr_endpoint = "https://api.mistral.ai/v1/ocr"
        self.chat_model = "mistral-large-latest"
        self.ocr_model = "mistral-ocr-latest"

    def get_api_key(self):
        if not self.api_keys: return ""
        return self.api_keys[int(time.time()) % len(self.api_keys)]

    def perform_ocr(self, file_path):
        """
        OCR for Images AND PDFs. Adapts MIME type based on file extension.
        """
        api_key = self.get_api_key()
        if not api_key: return "", []
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".pdf": mime_type = "application/pdf"
            elif ext == ".png": mime_type = "image/png"
            elif ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
            elif ext == ".webp": mime_type = "image/webp"
            elif ext == ".tiff": mime_type = "image/tiff"
            else: mime_type = "image/jpeg"

            with open(file_path, "rb") as file_data:
                base64_content = base64.b64encode(file_data.read()).decode('utf-8')

            payload = {
                "model": self.ocr_model,
                "document": {
                    "type": "image_url", 
                    "image_url": f"data:{mime_type};base64,{base64_content}"
                },
                "include_image_base64": False 
            }
            
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.post(self.ocr_endpoint, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                full_text = ""
                image_ids_found = []
                
                if 'pages' in result:
                    for page in result['pages']:
                        full_text += page.get('markdown', '') + "\n\n"
                        if 'images' in page:
                            for img_obj in page['images']:
                                image_ids_found.append(img_obj.get('id', 'unknown'))
                return full_text, image_ids_found
            else:
                logger.error(f"OCR Error {response.status_code}: {response.text}")
                return "", []
        except Exception as e:
            logger.error(f"OCR Exception: {e}")
            return "", []

    def extract_questions(self, raw_text, image_ids, company_context="Unknown", filename="Unknown"):
        if not raw_text or len(raw_text.strip()) < 5: return []
        api_key = self.get_api_key()
        
        safe_text = raw_text[:100000]

        prompt = f"""
        You are an Expert Technical Interview Data Extractor.
        Context: The file is from a folder named '{company_context}'.
        Source File: {filename}
        OCR detected these image IDs in text: {image_ids}
        
        **YOUR TASK:** Extract questions strictly following these rules:

        1. **CLEAN TEXT:** Remove image placeholders like `![image_id]`. Do not include "Refer to image below".
        2. **MCQ Formatting:** Combine Question Text + ALL Options (A, B, C, D) into `question_text`.
        3. **Coding Formatting:** Extract ENTIRE problem description VERBATIM.
        4. **Handling Images:** If a diagram is crucial, set `has_image` to "Yes".

        RAW TEXT:
        {safe_text} 

        **OUTPUT JSON:**
        {{
            "questions": [
                {{
                    "category": "Topic (e.g. Java, Aptitude, SQL)",
                    "question_text": "Cleaned text...",
                    "difficulty": "Easy/Medium",
                    "has_image": "Yes/No"
                }}
            ]
        }}
        """

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": self.chat_model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.1}

        for attempt in range(3):
            try:
                response = requests.post(self.chat_endpoint, headers=headers, json=payload, timeout=90)
                if response.status_code == 200:
                    try:
                        return json.loads(response.json()['choices'][0]['message']['content']).get("questions", [])
                    except: return []
                elif response.status_code == 429: time.sleep(3); continue
                else: return []
            except: time.sleep(1)
        return []

def parse_folder_name(folder_name):
    clean_name = folder_name.strip()
    if " - " in clean_name:
        parts = clean_name.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    elif "_" in clean_name:
        parts = clean_name.split("_", 1)
        return parts[0].strip(), parts[1].strip()
    else:
        return clean_name, "N/A"

def get_all_files_recursively(root_folder):
    """
    Recursively finds all valid image/pdf files in root_folder and subfolders.
    Ignores CSV, XLSX, etc.
    """
    target_files = []
    # Valid Extensions ONLY
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.pdf', '.tiff', '.bmp')
    
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for f in filenames:
            if not f.startswith('.'): # Ignore hidden files
                if f.lower().endswith(valid_extensions):
                    target_files.append(os.path.join(dirpath, f))
    return target_files

def main():

    if not get_mistral_api_keys():
        st.error("⚠️ No Mistral API Keys found in secrets. Please check your configuration.")
        return

    # --- ZIP UPLOAD ---
    uploaded_zip = st.file_uploader("Upload ZIP File", type="zip")
    
    if uploaded_zip:
        if st.button("🚀 Start Extraction", type="primary"):
            final_data = []
            
            with st.spinner("Processing ZIP file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    shutil.copyfileobj(uploaded_zip, tmp_zip)
                    tmp_zip_path = tmp_zip.name
                
                temp_root = tempfile.mkdtemp()
                
                try:
                    with zipfile.ZipFile(tmp_zip_path, 'r') as zf:
                        zf.extractall(temp_root)
                except Exception as e:
                    st.error(f"Zip Extraction Error: {e}")
                    return
                finally:
                    if os.path.exists(tmp_zip_path): os.remove(tmp_zip_path)

            if temp_root and os.path.exists(temp_root):
                client = MistralClientAssessment()
                items = os.listdir(temp_root)
                
                top_level_folders = [d for d in items if os.path.isdir(os.path.join(temp_root, d))]
                
                if not top_level_folders: top_level_folders = ["General"]
                     
                total_folders = len(top_level_folders)
                main_bar = st.progress(0, text="Overall Progress")
                status_text = st.empty()

                for idx, folder_name in enumerate(top_level_folders):
                    if folder_name == "General":
                        full_company_path = temp_root
                        company_name, job_id = "General", "N/A"
                    else:
                        full_company_path = os.path.join(temp_root, folder_name)
                        company_name, job_id = parse_folder_name(folder_name)

                    status_text.write(f"📂 Processing Company: **{company_name}**")
                    
                    # --- RECURSIVE FILE SEARCH ---
                    found_files = get_all_files_recursively(full_company_path)
                    
                    if not found_files:
                        continue

                    folder_bar = st.progress(0, text=f"Analyzing {len(found_files)} files...")
                    
                    for i, file_path in enumerate(found_files):
                        filename = os.path.basename(file_path)
                        
                        # --- DETERMINE SUBFOLDER CATEGORY ---
                        # This gets the name of the folder immediately containing the image
                        # e.g., if path is .../Company/MCQ/image.png -> parent is "MCQ"
                        parent_dir_name = os.path.basename(os.path.dirname(file_path))
                        
                        # If the parent is the company folder itself, label generally
                        if parent_dir_name == folder_name or parent_dir_name == company_name:
                            folder_source = "General"
                        else:
                            folder_source = parent_dir_name

                        # Perform OCR
                        ocr_text, img_ids = client.perform_ocr(file_path)
                        
                        if ocr_text:
                            # Pass Folder Source to AI for context
                            context_str = f"{company_name} - Folder: {folder_source}"
                            questions = client.extract_questions(ocr_text, img_ids, context_str, filename)
                            
                            if questions:
                                for q in questions:
                                    has_img = q.get("has_image", "No")
                                    img_ref = f"See: {filename}" if has_img.lower() in ["yes", "true"] else ""
                                    
                                    final_data.append({
                                        "Company Name": company_name,
                                        "Job ID": job_id,
                                        "Folder Source": folder_source, # NEW COLUMN
                                        "Tech Stack": q.get("category", "Uncategorized"),
                                        "Question": q.get("question_text", ""),
                                        "Difficulty": q.get("difficulty", "Unknown"),
                                        "Image Reference": img_ref,
                                        "Source File": filename 
                                    })
                        folder_bar.progress((i + 1) / len(found_files))
                    
                    folder_bar.empty()
                    main_bar.progress((idx + 1) / total_folders)

                status_text.write("✅ Processing Complete!")
                
                if temp_root and os.path.exists(temp_root):
                    shutil.rmtree(temp_root)

                if final_data:
                    df = pd.DataFrame(final_data)
                    # Reorder columns for better readability
                    cols = ["Company Name", "Job ID", "Folder Source", "Tech Stack", "Question", "Difficulty", "Image Reference", "Source File"]
                    # Only select columns that exist
                    df = df[[c for c in cols if c in df.columns]]
                    
                    st.success(f"Extracted {len(df)} questions!")
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download CSV", data=csv, file_name="Assessment_Questions_DeepExtract.csv", mime="text/csv")
                else:
                    st.warning("No questions found in the files provided.")

if __name__ == "__main__":
    main()