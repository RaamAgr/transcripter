# app.py — COMPLETE BATCH TRANSCRIBER (BASE64 IN PROMPT VERSION)
# -----------------------------------------------------------------------------
# FEATURES:
# 1. Multi-file Excel Upload (Merges files).
# 2. Custom Node/Puppeteer API (Strict 'provider' + 'prompt' schema).
# 3. Audio embedding: Base64 string is injected INSIDE the prompt text.
# 4. Robust Retries (Downloads & API calls).
# 5. High Concurrency (Adjustable via slider).
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import json
import os
import time
import logging
import mimetypes
import random
import math 
import html
import base64
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
API_URL = "https://ai-api-283392349456.europe-west1.run.app/ask"

# Streaming download chunk size (8KB)
DOWNLOAD_CHUNK_SIZE = 8192

# Retry configuration
MAX_WORKER_RETRIES = 3   # Attempts per row
WORKER_BACKOFF_BASE = 5  # Seconds base for backoff

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcriber")

# --- UI STYLING (CSS) ---
BASE_CSS = """
<style>
/* Card look for transcript entries */
.call-card {
    border: 1px solid var(--border-color, #e6e6e6);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    background: var(--card-bg, #fff);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Transcript scroll area */
.transcript-box {
    max-height: 320px;
    overflow: auto;
    padding: 8px;
    border-radius: 6px;
    background: var(--transcript-bg, #fafafa);
    border: 1px solid var(--border-color, #eee);
    font-family: monospace; 
    white-space: pre-wrap; 
}

/* Speaker colors */
.speaker1 { color: #1f77b4; font-weight: 600; display: block; margin-bottom: 4px; }
.speaker2 { color: #d62728; font-weight: 600; display: block; margin-bottom: 4px; }
.other-speech { color: #333; display: block; margin-bottom: 4px; }

/* Metadata row */
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }

/* Theming */
.dark-theme {
    --card-bg: #0f1115;
    --transcript-bg: #0b0c0f;
    --border-color: #222428;
    --meta-color: #9aa0a6;
    color: #e6eef3;
}
.light-theme {
    --card-bg: #ffffff;
    --transcript-bg: #fafafa;
    --border-color: #e6e6e6;
    --meta-color: #666666;
    color: #111;
}
</style>
"""

st.set_page_config(page_title="Batch Transcriber (Embedded Audio)", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --- NETWORK UTILITIES ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    """Sleeps with exponential backoff + jitter."""
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 60)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 2.0, **kwargs) -> requests.Response:
    """Robust request wrapper."""
    last_exc = None
    # Default timeout is high (180s) to allow large Base64 payloads to upload
    kwargs.setdefault('timeout', 180)
    
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, **kwargs)
            
            # Treat 429 and 5xx as transient errors
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning(f"Transient HTTP {resp.status_code} from {url} (attempt {attempt+1}). Retrying...")
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning(f"RequestException on {method} {url}: {str(e)} (attempt {attempt+1})")
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
     
    if last_exc:
        raise last_exc
    raise Exception("make_request_with_retry: retries exhausted")


# --- MIME TYPE UTILS ---

COMMON_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wave",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac"
}

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    """Detects MIME type from URL path or Content-Type header."""
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
    
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype:
                return k, ctype
        guessed_ext = mimetypes.guess_extension(ctype)
        if guessed_ext:
            return guessed_ext.lower(), ctype
            
    return ".mp3", "audio/mpeg"


# --- CUSTOM API INTERACTION ---

def call_custom_api(full_embedded_prompt: str) -> str:
    """
    Calls the custom API with strict schema:
    { "provider": "gemini", "prompt": <Instruction + Base64> }
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "provider": "gemini",
        "prompt": full_embedded_prompt
    }
    
    # Send request
    resp = make_request_with_retry("POST", API_URL, json=payload, headers=headers)
    
    if resp.status_code != 200:
        return f"API ERROR {resp.status_code}: {resp.text}"
        
    try:
        body = resp.json()
        answer = body.get("answer", "")
        if not answer:
            # If 'answer' is missing, dump the body for debug
            return f"NO TRANSCRIPT (Empty 'answer'). Body: {str(body)}"
        return answer
    except ValueError:
        return f"PARSE ERROR: Non-JSON response. Body: {resp.text[:200]}..."

def build_instruction_header(language_label: str) -> str:
    """Returns the text instructions to be placed at the top of the prompt."""
    return f"""
Transcribe this audio in {language_label} exactly as spoken.

CRITICAL REQUIREMENTS:
1. Label every line with 'Speaker 1:' or 'Speaker 2:'.
2. NEVER merge dialogue from two speakers.
3. Timestamp format MUST be: [0ms-2500ms].
4. Language: Hinglish (Latin script only). No Devanagari.

STRICT OUTPUT FORMAT:
[timestamp] Speaker X: line of dialogue
"""


# --- DATA PREPARATION ---

def prepare_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Marks rows for processing based on recording_url presence."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    final_rows = []
    for index, row in df.iterrows():
        row_data = row.copy()
        url_val = row_data.get('recording_url')
        
        if pd.notna(url_val) and str(url_val).strip() != "":
            row_data['processing_action'] = 'TRANSCRIBE'
            row_data['status'] = 'Pending'
        else:
            row_data['processing_action'] = 'SKIP'
            row_data['transcript'] = "⚠️ Skipped: No recording URL."
            row_data['status'] = '⚠️ Skipped'
            row_data['error'] = 'Missing recording_url'
        
        final_rows.append(row_data)
    
    return pd.DataFrame(final_rows).reset_index(drop=True)


# --- WORKER FUNCTION ---

def process_single_row(index: int, row: pd.Series, instruction_header: str) -> Dict[str, Any]:
    """
    Worker:
    1. Downloads audio.
    2. Converts to Base64.
    3. Concatenates Instructions + Base64.
    4. Calls API.
    """
    mobile = str(row.get("mobile_number", "Unknown"))
    
    result = {
        "index": index,
        "mobile_number": mobile,
        "recording_url": row.get("recording_url"),
        "transcript": row.get("transcript", ""), 
        "status": row.get("status", "Pending"),
        "error": row.get("error", None),
    }

    if row.get("processing_action") == "SKIP":
        return result

    audio_url = row.get("recording_url")

    # --- JOB-LEVEL RETRY LOOP ---
    for worker_attempt in range(MAX_WORKER_RETRIES):
        try:
            parsed = urlparse(audio_url)

            # 1. Download Audio
            logger.info(f"Attempt {worker_attempt + 1}: Downloading {mobile}...")
            r = make_request_with_retry("GET", audio_url, stream=True)
            
            if r.status_code != 200:
                raise Exception(f"Download failed: HTTP {r.status_code}")

            header_ct = r.headers.get("content-type", "")
            ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)

            audio_buffer = BytesIO()
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    audio_buffer.write(chunk)
            
            # 2. Convert to Base64 String
            audio_buffer.seek(0)
            b64_bytes = base64.b64encode(audio_buffer.read())
            b64_string = b64_bytes.decode('utf-8')

            # 3. Construct the Payload Prompt
            # This embeds the file content directly into the text prompt
            full_prompt_payload = (
                f"{instruction_header}\n\n"
                f"--- AUDIO DATA START ---\n"
                f"MIME_TYPE: {mime_type}\n"
                f"BASE64_DATA: {b64_string}\n"
                f"--- AUDIO DATA END ---"
            )

            # 4. Call API
            logger.info(f"Attempt {worker_attempt + 1}: Sending payload to API for {mobile}...")
            transcript = call_custom_api(full_prompt_payload)
            
            result["transcript"] = transcript
            
            # Basic error checking on the response text
            if "API ERROR" in transcript or "PARSE ERROR" in transcript:
                result["status"] = "❌ Error"
                # If retries remain, raise exception to trigger retry logic
                if worker_attempt < MAX_WORKER_RETRIES - 1:
                    raise Exception(f"API Error Response: {transcript}")
            else:
                result["status"] = "✅ Success"
                return result

        except Exception as e:
            # If this was the last attempt
            if worker_attempt == MAX_WORKER_RETRIES - 1:
                logger.exception(f"Final failure for {mobile}: {str(e)}")
                result["transcript"] = f"SYSTEM ERROR: {str(e)}"
                result["status"] = "❌ Failed"
                result["error"] = str(e)
            else:
                # Transient error, wait and retry
                logger.warning(f"Retry {mobile} (Attempt {worker_attempt + 1}): {str(e)}")
                _sleep_with_jitter(WORKER_BACKOFF_BASE, worker_attempt)

    return result


# --- RESULT MERGING & DISPLAY ---

def merge_results_with_original(df_consolidated: pd.DataFrame, processed_results: list) -> pd.DataFrame:
    results_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
    cols_to_update = ["transcript", "status", "error"]
    
    # Remove existing columns to avoid duplication
    df_base = df_consolidated.drop(columns=[c for c in cols_to_update if c in df_consolidated.columns])
    
    # Merge on index
    merged = df_base.merge(results_df[["index"] + cols_to_update], left_index=True, right_on="index", how="left")
    
    if "index" in merged.columns:
        merged = merged.drop(columns=["index"])
    return merged

def colorize_transcript_html(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "<div class='other-speech'>No transcript</div>"
    
    lines = text.splitlines()
    html_output = ""
    for line in lines:
        clean = line.strip()
        if not clean: continue
        escaped = html.escape(clean)
        lc = clean.lower()
        if "speaker 1:" in lc:
            html_output += f"<div class='speaker1'>{escaped}</div>"
        elif "speaker 2:" in lc:
            html_output += f"<div class='speaker2'>{escaped}</div>"
        else:
            html_output += f"<div class='other-speech'>{escaped}</div>"
    return f"<div>{html_output}</div>"


# --- MAIN APP ---

def main():
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = []
    if "final_df" not in st.session_state:
        st.session_state.final_df = pd.DataFrame()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        st.info("Target: Custom Node API (Base64 Embed)")
        
        # Concurrency Slider
        max_workers = st.slider("Concurrency", min_value=1, max_value=32, value=4, 
                              help="Number of simultaneous requests.")
        
        st.divider()
        language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
        theme_choice = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)

    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

    # --- Main Content ---
    st.write("### 📂 Batch Transcriber (Embedded Audio)")
    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)
    
    # Status Placeholders
    progress_bar = st.empty()
    status_text = st.empty()
    result_placeholder = st.empty()
    
    start_button = st.button("🚀 Start Processing", type="primary")

    if start_button:
        # 1. File Validation
        if not uploaded_files:
            st.error("Please upload at least one Excel file.")
            st.stop()

        all_dfs = []
        for file in uploaded_files:
            try:
                all_dfs.append(pd.read_excel(file))
            except Exception as e:
                st.warning(f"Error reading {file.name}: {e}")
        
        if not all_dfs:
            st.error("No valid data found.")
            st.stop()

        # 2. Merge Data
        raw_df = pd.concat(all_dfs, ignore_index=True)
        
        if "recording_url" not in raw_df.columns:
            st.error("Missing required column: 'recording_url'")
            st.stop()

        # 3. Prep Data
        df_ready = prepare_all_rows(raw_df)
        instruction_header = build_instruction_header(language_mode)
        total_items = len(df_ready)
        
        status_text.info(f"Processing {total_items} items...")
        progress_bar.progress(0.0)
        
        processed_results = []
        st.session_state.processed_results = []

        # 4. Processing Loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_row, idx, row, instruction_header): idx
                for idx, row in df_ready.iterrows()
            }

            completed = 0
            for future in as_completed(futures):
                res = future.result()
                processed_results.append(res)
                completed += 1
                
                # Update Status
                progress_bar.progress(completed / total_items)
                status_text.markdown(f"**{completed}/{total_items}** Completed")
                
                # Live Preview (Last 5)
                recent = sorted(processed_results, key=lambda r: r["index"])[-5:]
                if recent:
                    result_placeholder.dataframe(
                        pd.DataFrame(recent)[["mobile_number", "status", "transcript"]],
                        hide_index=True,
                        height=200
                    )

        # 5. Finalize
        final_df = merge_results_with_original(df_ready, processed_results)
        st.session_state.final_df = final_df
        status_text.success("Processing Complete!")

    # --- Results Viewer ---
    final_df = st.session_state.final_df
    
    if not final_df.empty:
        st.markdown("---")
        st.markdown("## 🎛️ Transcript Browser")
        
        # Filters
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1: search_q = st.text_input("Search", placeholder="Search text...")
        with c2: status_sel = st.selectbox("Status", ["All", "Success", "Failed"])
        with c3: speaker_sel = st.selectbox("Speaker", ["All", "Speaker 1", "Speaker 2"])
        with c4: per_page = st.selectbox("Per Page", [5, 10, 20], index=1)
        
        # Apply Logic
        view = final_df.copy()
        
        if status_sel != "All":
            if status_sel == "Success":
                view = view[view["status"].str.contains("Success", case=False, na=False)]
            else:
                view = view[view["status"].str.contains("Failed|Error|Skipped", case=False, na=False)]
                
        if search_q:
            q = search_q.lower()
            mask = (
                view["transcript"].fillna("").str.lower().str.contains(q) |
                view["mobile_number"].astype(str).str.lower().str.contains(q)
            )
            view = view[mask]

        if speaker_sel != "All":
            key = "speaker 1" if speaker_sel == "Speaker 1" else "speaker 2"
            view = view[view["transcript"].fillna("").str.lower().str.contains(key)]

        # Pagination
        total_len = len(view)
        st.write(f"**Found {total_len} items**")
        
        pages = max(1, math.ceil(total_len / per_page))
        curr_page = st.number_input("Page", min_value=1, max_value=pages, value=1)
        start_idx = (curr_page - 1) * per_page
        end_idx = start_idx + per_page
        
        page_data = view.iloc[start_idx:end_idx]

        # Download Button
        out_buf = BytesIO()
        view.to_excel(out_buf, index=False)
        st.download_button("📥 Download Results", out_buf.getvalue(), "transcripts.xlsx")

        # Display Cards
        for _, row in page_data.iterrows():
            mobile_disp = row.get("mobile_number", "Unknown")
            status_disp = row.get("status", "")
            
            with st.expander(f"{mobile_disp} — {status_disp}", expanded=False):
                url_htm = html.escape(str(row.get('recording_url', '-')))
                st.markdown(f"<div class='meta-row'><b>URL:</b> {url_htm}</div>", unsafe_allow_html=True)
                
                trans_html = colorize_transcript_html(row.get("transcript", ""))
                st.markdown(f"<div class='transcript-box'>{trans_html}</div>", unsafe_allow_html=True)
                
                if row.get("error"):
                    st.error(f"Error: {row.get('error')}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
