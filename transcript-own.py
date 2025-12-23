# app.py — COMPLETE BATCH TRANSCRIBER (CLOUD RUN AGENT INTEGRATION)
# -----------------------------------------------------------------------------
# FEATURES:
# 1. Multi-file Excel Upload (Merges multiple files).
# 2. INTEGRATED: Custom Browser-Agent API (Cloud Run).
# 3. Unique Mobile Number Logic (Processes each contact only once).
# 4. Latest Recording Priority (Always picks the most recent valid audio).
# 5. Outcome History Fallback (Compiles outcomes if no audio exists).
# 6. High Concurrency (Slider up to 128 workers).
# 7. Robust MIME Detection & Retries.
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import json
import os
import time
import logging
import mimetypes
import tempfile
import random
import math 
import html
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
# Your deployed Cloud Run Agent Endpoint
CUSTOM_API_URL = "https://ai-api-283392349456.europe-west1.run.app/ask"

# Streaming download chunk size (8KB)
DOWNLOAD_CHUNK_SIZE = 8192

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

/* Speaker colors for Diarization */
.speaker1 { color: #1f77b4; font-weight: 600; display: block; margin-bottom: 4px; }
.speaker2 { color: #d62728; font-weight: 600; display: block; margin-bottom: 4px; }
.other-speech { color: #333; display: block; margin-bottom: 4px; }

/* Compact metadata row */
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }

/* Theming variables */
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

/* Search box styling */
.search-box { margin-bottom: 10px; padding: 6px; border-radius: 6px; border: 1px solid var(--border-color, #eee); width:100%; }
</style>
"""

st.set_page_config(page_title="Batch Transcriber", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --- NETWORK UTILITIES ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    """
    Sleeps for a random amount of time to prevent thundering herd problems.
    """
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 30)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, timeout: int = 60, **kwargs) -> requests.Response:
    """
    Robust wrapper for requests with exponential backoff + jitter.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            # Treat 429 and 5xx as transient errors
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning("Transient HTTP %s from %s (attempt %d). Retrying...", resp.status_code, url, attempt + 1)
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning("RequestException on %s %s: %s (attempt %d)", method, url, str(e), attempt + 1)
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
    
    if last_exc:
        raise last_exc
    raise Exception("make_request_with_retry: retries exhausted without a response")


# --- MIME TYPE & FILE EXTENSION HANDLING ---

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
    """
    Determine extension and mime type from URL path and header.
    """
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    
    # 1. Trust extension if it is a known audio type
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
    
    # 2. Try header Content-Type
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype:
                return k, ctype
        
        guessed_ext = mimetypes.guess_extension(ctype)
        if guessed_ext:
            return guessed_ext.lower(), ctype
            
    # 3. Last fallback
    return ".mp3", "audio/mpeg"


# --- PROMPT BUILDER ---

def build_prompt(language_label: str) -> str:
    """
    Constructs the system prompt for strict diarization and formatting.
    """
    return f"""
Transcribe this call in Hindi and English exactly as spoken.

SPEAKER IDENTIFICATION RULES:
1. Identify AGENT vs CUSTOMER contextually.
2. Use labels 'Agent:' and 'Customer:' (fallback to Speaker 1/2 if unclear).
3. MAINTAIN CONSISTENCY.

FORMATTING:
- [0ms-1500ms] Label: Dialogue
- Raw milliseconds for timestamps.
- Separate speakers onto new lines.

LANGUAGE:
- Hindi words in Hinglish (Latin script).
- NO Devanagari.
- Context: {language_label}

Return ONLY the transcript text.
"""


# --- DATA CONSOLIDATION LOGIC ---

def consolidate_data_by_mobile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates data to ensure unique mobile number processing.
    Picks latest recording OR compiles outcome history.
    """
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by=['mobile_number', 'date'], ascending=[True, False])
    
    unique_mobiles = df['mobile_number'].unique()
    final_rows = []

    for mobile in unique_mobiles:
        group = df[df['mobile_number'] == mobile]
        
        # Valid recordings
        valid_recording_rows = group[group['recording_url'].notna() & (group['recording_url'] != "")]
        
        if not valid_recording_rows.empty:
            # Case A: Audio Exists (Pick latest)
            selected_row = valid_recording_rows.iloc[0].copy()
            selected_row['processing_action'] = 'TRANSCRIBE'
        else:
            # Case B: Fallback to History
            selected_row = group.iloc[0].copy()
            selected_row['processing_action'] = 'SKIP'
            
            outcomes = group['outcome'].fillna("").astype(str).tolist()
            outcome_history = " ".join([x.strip() for x in outcomes if x.strip()])
            
            selected_row['transcript'] = f"Outcomes: {outcome_history}"
            selected_row['status'] = '⚠️ Skipped (No Audio)'
            selected_row['error'] = 'No recording_url found in history'
        
        final_rows.append(selected_row)
    
    return pd.DataFrame(final_rows).reset_index(drop=True)


# --- WORKER / PROCESSING FUNCTION ---

def process_single_row(index: int, row: pd.Series, prompt_template: str) -> Dict[str, Any]:
    """
    Worker function. Downloads audio and sends to Custom Cloud Run API.
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

    # SKIP logic from consolidation
    if row.get("processing_action") == "SKIP":
        return result

    audio_url = row.get("recording_url")
    if not audio_url or not isinstance(audio_url, str):
        result.update({"status": "❌ Failed", "error": "Invalid URL"})
        return result

    tmp_path = None

    try:
        # 1. Download File
        parsed = urlparse(audio_url)
        r = make_request_with_retry("GET", audio_url, stream=True)
        
        if r.status_code != 200:
            raise Exception(f"Audio download failed ({r.status_code})")

        header_ct = r.headers.get("content-type", "")
        ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)
        expected_size = r.headers.get("content-length")

        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk: tmp.write(chunk)
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path)
        
        # Integrity Check
        if expected_size:
            if file_size < int(expected_size):
                raise Exception(f"Incomplete download: Expected {expected_size} bytes, got {file_size}")

        # 2. Send to Custom Cloud Run API (with Retry)
        # We try up to 3 times to handle browser glitches on the server side
        max_api_retries = 3
        transcript_text = None
        
        for attempt in range(max_api_retries):
            with open(tmp_path, "rb") as f:
                # Prepare Multipart Upload
                files = {
                    'file': (f"audio{ext}", f, mime_type)
                }
                data = {
                    'provider': 'gemini',
                    'prompt': prompt_template
                }
                
                # High timeout (600s) because the browser agent needs time to type/upload/think
                try:
                    resp = requests.post(CUSTOM_API_URL, files=files, data=data, timeout=600)
                    
                    if resp.status_code == 200:
                        ans = resp.json().get("answer", "")
                        if ans and "error" not in ans.lower():
                            transcript_text = ans
                            break # Success!
                        else:
                            logger.warning(f"API returned empty/error on attempt {attempt+1}: {ans}")
                    else:
                        logger.warning(f"API HTTP {resp.status_code} on attempt {attempt+1}")
                        
                except Exception as e:
                    logger.warning(f"API Connection error on attempt {attempt+1}: {e}")
                
                time.sleep(2 * (attempt + 1)) # Backoff

        if transcript_text:
            result["transcript"] = transcript_text
            result["status"] = "✅ Success"
        else:
            result["status"] = "❌ Failed"
            result["error"] = "Agent API failed after retries"
            result["transcript"] = "Error: Could not retrieve transcript from Agent."

    except Exception as e:
        logger.exception("Processing failed for row %s: %s", index, str(e))
        result["transcript"] = f"SYSTEM ERROR: {str(e)}"
        result["status"] = "❌ Failed"
        result["error"] = str(e)

    finally:
        # Cleanup local temp file
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    return result


# --- RESULT MERGING & DISPLAY UTILS ---

def merge_results_with_original(df_consolidated: pd.DataFrame, processed_results: list) -> pd.DataFrame:
    """
    Merges worker results back into the consolidated DataFrame.
    """
    results_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
    cols_to_update = ["transcript", "status", "error"]
    
    df_base = df_consolidated.drop(columns=[c for c in cols_to_update if c in df_consolidated.columns])
    merged = df_base.merge(results_df[["index"] + cols_to_update], left_index=True, right_on="index", how="left")
    
    if "index" in merged.columns:
        merged = merged.drop(columns=["index"])
    
    return merged

def colorize_transcript_html(text: str) -> str:
    """
    Format transcript text into color-coded HTML.
    """
    if not isinstance(text, str) or not text.strip():
        return "<div class='other-speech'>No transcript</div>"

    lines = text.splitlines()
    html_output = ""
    
    for line in lines:
        clean = line.strip()
        if not clean: continue
        
        escaped_line = html.escape(clean)
        lc = clean.lower()
        
        if "agent:" in lc:
            html_output += f"<div class='speaker1'>{escaped_line}</div>"
        elif "customer:" in lc:
            html_output += f"<div class='speaker2'>{escaped_line}</div>"
        else:
            html_output += f"<div class='other-speech'>{escaped_line}</div>"
            
    return f"<div>{html_output}</div>"


# --- MAIN APPLICATION ENTRY POINT ---

def main():
    # Initialize Session State
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = []
    if "final_df" not in st.session_state:
        st.session_state.final_df = pd.DataFrame()

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("Configuration")
        st.info(f"Agent Endpoint:\n{urlparse(CUSTOM_API_URL).hostname}")
        
        # Max workers slider
        max_workers = st.slider("Concurrency (Threads)", min_value=1, max_value=128, value=4,
                                help="Higher = faster, but ensure Cloud Run can handle the load.")
        
        st.divider()
        
        language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
        
        theme_choice = st.radio("Theme", options=["Light", "Dark"], index=0, horizontal=True)

    # Apply Theme Class
    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

    # --- FILE UPLOADER ---
    st.write("### 📂 Upload Call Data")
    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

    # Status Containers
    progress_bar = st.empty()
    status_text = st.empty()
    result_placeholder = st.empty()

    start_button = st.button("🚀 Start Batch Processing", type="primary")

    # --- PROCESSING LOGIC ---
    if start_button:
        if not uploaded_files:
            st.error("Please upload at least one Excel file.")
            st.stop()

        # 1. READ FILES
        all_dfs = []
        for file in uploaded_files:
            try:
                df_single = pd.read_excel(file)
                all_dfs.append(df_single)
            except Exception as e:
                st.warning(f"Skipping {file.name}: {e}")
                continue
        
        if not all_dfs:
            st.stop()

        raw_df = pd.concat(all_dfs, ignore_index=True)

        # 2. CONSOLIDATE
        status_text.info("Consolidating data by unique mobile number...")
        df_consolidated = consolidate_data_by_mobile(raw_df)
        
        prompt_template = build_prompt(language_mode)
        total_rows = len(df_consolidated)
        
        status_text.info(f"Processing {total_rows} unique contacts with {max_workers} threads...")
        progress_bar.progress(0.0)

        processed_results = []
        st.session_state.processed_results = []

        # 3. EXECUTE
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_row, idx, row, prompt_template): idx
                for idx, row in df_consolidated.iterrows()
            }

            completed = 0
            for future in as_completed(futures):
                res = future.result()
                processed_results.append(res)
                completed += 1
                
                # Update UI
                progress_bar.progress(completed / total_rows)
                status_text.markdown(f"Processed **{completed}/{total_rows}** unique contacts.")
                
                # Live Preview
                recent = sorted(processed_results, key=lambda r: r["index"])[-5:]
                if recent:
                    result_placeholder.dataframe(
                        pd.DataFrame(recent)[["mobile_number", "status", "transcript"]], 
                        hide_index=True
                    )

        # 4. MERGE
        final_df = merge_results_with_original(df_consolidated, processed_results)
        st.session_state.final_df = final_df
        status_text.success("Batch Processing Complete!")

    # --- RESULTS VIEWER ---
    final_df = st.session_state.final_df

    if not final_df.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("## 🎛️ Transcript Browser")

        # Filters
        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
        with col_a:
            search_q = st.text_input("Search", placeholder="Search transcript, phone, or URL...")
        with col_b:
            status_sel = st.selectbox("Status", ["All", "Success", "Failed", "Skipped"])
        with col_c:
            speaker_sel = st.selectbox("Speaker", ["All", "Agent", "Customer"])
        with col_d:
            per_page = st.selectbox("Per page", [5, 10, 20, 50], index=1)

        # Apply Filters
        view_df = final_df.copy()
        
        if status_sel != "All":
            if status_sel == "Success":
                view_df = view_df[view_df["status"].str.contains("Success", case=False, na=False)]
            elif status_sel == "Failed":
                view_df = view_df[view_df["status"].str.contains("Failed|Error", case=False, na=False)]
            elif status_sel == "Skipped":
                view_df = view_df[view_df["status"].str.contains("Skipped", case=False, na=False)]

        if search_q.strip():
            q = search_q.lower()
            mask = (
                view_df["transcript"].fillna("").str.lower().str.contains(q) |
                view_df["mobile_number"].astype(str).str.lower().str.contains(q)
            )
            view_df = view_df[mask]

        if speaker_sel != "All":
            key = "agent:" if speaker_sel == "Agent" else "customer:"
            view_df = view_df[view_df["transcript"].fillna("").str.lower().str.contains(key)]

        total_items = len(view_df)
        st.markdown(f"**Showing {total_items} result(s)**")

        # Pagination
        pages = max(1, math.ceil(total_items / per_page))
        page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
        start = (page_idx - 1) * per_page
        end = start + per_page
        page_df = view_df.iloc[start:end]

        # Export
        out_buf = BytesIO()
        view_df.to_excel(out_buf, index=False)
        st.download_button(
            "📥 Download Filtered Results",
            data=out_buf.getvalue(),
            file_name=f"transcripts_export_{int(time.time())}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Render Cards
        for idx, row in page_df.iterrows():
            mobile_display = row.get('mobile_number', 'Unknown')
            status_display = row.get('status', '')
            
            with st.expander(f"{mobile_display} — {status_display}", expanded=False):
                url_val = html.escape(str(row.get('recording_url', 'None')))
                st.markdown(f"<div class='meta-row'><b>URL:</b> {url_val}</div>", unsafe_allow_html=True)
                
                transcript_html = colorize_transcript_html(row.get("transcript", ""))
                st.markdown(f"<div class='transcript-box'>{transcript_html}</div>", unsafe_allow_html=True)
                
                if row.get("error"):
                    st.error(f"Error: {row.get('error')}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
