import streamlit as st
import pandas as pd
import requests
import json
import time
import os
import toml
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from random import uniform

# ==============================================================================
#  CONFIG & SECRETS
# ==============================================================================
st.set_page_config(page_title="Drilldown Analysis", layout="wide", page_icon="⚡")

def get_secret(key):
    """Helper to get secrets from toml or st.secrets"""
    # Adjust path because this file is now inside /pages/ folder
    # We need to go up two levels to find .streamlit/secrets.toml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
    secrets_path = os.path.join(parent_dir, ".streamlit", "secrets.toml")
    
    val = None
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r") as f:
                secrets = toml.load(f)
                val = secrets.get(key)
        except:
            pass
            
    # Fallback to Streamlit native secrets (Cloud)
    if not val:
        try:
            val = st.secrets.get(key)
        except:
            pass
    return val

# Load Keys
MISTRAL_API_KEYS = [k for k in [get_secret("MISTRAL_API_KEY_1"), get_secret("MISTRAL_API_KEY_2"), get_secret("MISTRAL_API_KEY_3")] if k]

# ==============================================================================
#  AI LOGIC
# ==============================================================================

def analyze_with_mistral(prompt: str, api_key: str) -> str:
    if not api_key: return json.dumps({"extracted_data": []})
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": "You are a data standardization expert. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(3):
        try:
            resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
            elif resp.status_code == 429:
                time.sleep(3 + uniform(0, 2))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return json.dumps({"extracted_data": []})

def process_drilldown_row(row_data, api_key):
    # 1. Metadata
    date_val = row_data.get('Interview Date', 'N/A')
    user_id = row_data.get('User ID', 'N/A')
    job_id = row_data.get('Job ID', 'N/A')

    # 2. Extract Text from UPDATED Columns
    rounds_data = {}
    
    # --- MAPPING COLUMNS TO ROUNDS ---
    if pd.notna(row_data.get('Assessment questions')): 
        rounds_data['Assessment'] = str(row_data['Assessment questions'])
        
    if pd.notna(row_data.get('Technical round Questions')): 
        rounds_data['Technical Round 1'] = str(row_data['Technical round Questions'])
        
    # ADDED: Technical2 round Questions
    if pd.notna(row_data.get('Technical2 round Questions')): 
        rounds_data['Technical Round 2'] = str(row_data['Technical2 round Questions'])
        
    if pd.notna(row_data.get('H.R Questions')): 
        rounds_data['HR Round'] = str(row_data['H.R Questions'])
        
    if pd.notna(row_data.get('Managerial Round questions')): 
        rounds_data['Managerial Round'] = str(row_data['Managerial Round questions'])
        
    if pd.notna(row_data.get('CEO/Founder/Director Round Questions')): 
        rounds_data['CEO/Director Round'] = str(row_data['CEO/Founder/Director Round Questions'])

    # Cleanup (Ignore empty/short answers/NaNs)
    clean_rounds = {k: v for k, v in rounds_data.items() if len(v) > 3 and v.lower() not in ['nan', 'no', 'yes', 'na', 'none', 'null']}
    
    if not clean_rounds: return []

    # 3. Build Prompt
    context_text = ""
    for r_name, r_text in clean_rounds.items():
        context_text += f"\n[ROUND: {r_name}]\n{r_text}\n"

    prompt = f"""
    Analyze the interview notes below and extract every single question asked.
    
    RULES:
    1. "round": Use the round name provided in the text (e.g., Technical Round 1, Technical Round 2, HR Round).
    2. "question": The exact question text.
    3. "tech_stack": Choose BEST FIT from: [Java, React JS, CSS, JavaScript, SDLC, Node JS, Python, General, SQL, AI/ML, QA/Testing, AWS, HTML, Spring Boot, DevOps, DSA, C#, Power BI, Web Development, C++, Excel, Salesforce, Aptitude and Logical Reasoning]
       * If question is about Introduction or Project or Behavioral -> Use "General".
    4. "type": Choose from: [Technical Theory, Coding, Self-Introduction, General, Project Explanation, Logical Reasoning, Aptitude, English, Career]
    5. "difficulty": [Easy, Medium, Hard]

    INPUT DATA:
    {context_text}
    
    Return JSON object with key "extracted_data".
    Example Output:
    {{
        "extracted_data": [
            {{"round": "Technical Round 2", "question": "Explain closures.", "tech_stack": "JavaScript", "type": "Technical Theory", "difficulty": "Medium"}}
        ]
    }}
    """

    try:
        response = analyze_with_mistral(prompt, api_key)
        items = json.loads(response).get("extracted_data", [])
        
        processed_rows = []
        for item in items:
            processed_rows.append({
                "Date Of Question Gathering": date_val,
                "User ID": user_id,
                "Job ID": job_id,
                "Interview Round": item.get("round", "Unknown"),
                "Question Text": item.get("question", ""),
                "Question Type": item.get("type", "General"),
                "Tech Stack": item.get("tech_stack", "General"),
                "Difficulty Level": item.get("difficulty", "Medium"),
                "Curriculum Coverage": "Covered"
            })
        return processed_rows
    except:
        return []

# ==============================================================================
#  UI LAYOUT
# ==============================================================================

if not MISTRAL_API_KEYS:
    st.error("⚠️ No API Keys found in secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False, engine='python')
    df.columns = df.columns.str.strip() 
    st.info(f"Loaded **{len(df)}** candidate rows.")
    
    if st.button("🚀 Start Live Analysis", type="primary"):
        progress_bar = st.progress(0)
        stats_place = st.empty()
        table_place = st.empty()
        
        rows_to_process = df.to_dict('records')
        total = len(rows_to_process)
        all_results = []
        
        # Using max_workers based on number of keys to avoid rate limits
        workers = len(MISTRAL_API_KEYS) + 1
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, row in enumerate(rows_to_process):
                key = MISTRAL_API_KEYS[i % len(MISTRAL_API_KEYS)]
                futures.append(executor.submit(process_drilldown_row, row, key))
            
            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data: all_results.extend(data)
                completed_count += 1
                
                progress_bar.progress(completed_count / total)
                stats_place.markdown(f"**Processed: {completed_count}/{total}** | **Questions Extracted: {len(all_results)}**")
                
                if all_results:
                    live_df = pd.DataFrame(all_results)
                    # Limit height for better UI performance during updates
                    table_place.dataframe(live_df, use_container_width=True, height=400)

        st.success("✅ Extraction Complete!")
        
        if all_results:
            final_df = pd.DataFrame(all_results)
            target_cols = ["Date Of Question Gathering", "User ID", "Job ID", "Interview Round", "Question Text", "Question Type", "Tech Stack", "Difficulty Level", "Curriculum Coverage"]
            final_df = final_df[[c for c in target_cols if c in final_df.columns]]
            
            csv_data = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Final CSV", data=csv_data, file_name="Drilldown_Questions_Processed.csv", mime="text/csv")