# app.py — COMPLETE BATCH TRANSCRIBER (ASYNC HEADER + CHAT UI + AUTH FIXED + RAW JSON VIEW)
# -----------------------------------------------------------------------------
# FEATURES:
# 1. ASYNC HEADER: Sends 'x-async-mode: true' to trigger job queueing.
# 2. NEW CHAT HEADER: Sends 'new-chat: true' to force fresh session context.
# 3. AUTHENTICATION: Requires 'x-api-key' for both submission and polling.
# 4. ROBUST POLLING: Submits -> Gets Job ID -> Polls for completion.
# 5. CHAT UI: Modern, color-coded bubbles for easy reading.
# 6. RAW INSPECTOR: View the raw JSON response via an info button.
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
CUSTOM_API_URL = "https://tulsa-intervention-stranger-subsidiary.trycloudflare.com/ask"

# Streaming download chunk size (8KB)
DOWNLOAD_CHUNK_SIZE = 8192

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcriber")

# --- UI STYLING (MODERN CHAT LOOK) ---
BASE_CSS = """
<style>
/* Main Container */
.main { background-color: var(--bg-color); }

/* Card Styling */
.call-card {
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    background: var(--card-bg);
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Transcript Container */
.transcript-box {
    max-height: 400px;
    overflow-y: auto;
    padding: 15px;
    border-radius: 8px;
    background: var(--transcript-bg);
    border: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* Chat Bubbles */
.chat-bubble {
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.5;
    max-width: 90%;
    position: relative;
    font-family: 'Source Sans Pro', sans-serif;
}

/* Agent (Left, Blueish) */
.agent-bubble {
    align-self: flex-start;
    background-color: #e3f2fd;
    color: #0d47a1;
    border-bottom-left-radius: 2px;
}

/* Customer (Right, Greenish) */
.customer-bubble {
    align-self: flex-end;
    background-color: #e8f5e9;
    color: #1b5e20;
    border-bottom-right-radius: 2px;
}

/* Unknown/Other (Gray) */
.other-bubble {
    align-self: center;
    background-color: #f5f5f5;
    color: #424242;
    font-style: italic;
    width: 100%;
    text-align: center;
}

/* Timestamp Label */
.bubble-meta {
    display: block;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 4px;
    opacity: 0.7;
    text-transform: uppercase;
}

/* Dark Mode Overrides */
.dark-theme .agent-bubble { background-color: #1e3a5f; color: #bbdefb; }
.dark-theme .customer-bubble { background-color: #1b4d3e; color: #c8e6c9; }
.dark-theme .other-bubble { background-color: #333; color: #bbb; }
.dark-theme .transcript-box { background: #0e1117; border-color: #262730; }

/* Theming Variables */
.dark-theme {
    --card-bg: #1f2937;
    --transcript-bg: #111827;
    --border-color: #374151;
}
.light-theme {
    --card-bg: #ffffff;
    --transcript-bg: #f9fafb;
    --border-color: #e5e7eb;
}
</style>
"""

st.set_page_config(page_title="Batch Transcriber", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --- NETWORK UTILITIES ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 30)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, timeout: int = 60, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            # Retry on 429 or 5xx errors
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning(f"Transient HTTP {resp.status_code}. Retrying...")
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error: {e}. Retrying...")
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
      
    if last_exc: raise last_exc
    raise Exception("make_request_with_retry failed")


# --- MIME TYPE DETECTION ---

COMMON_AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wave", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".webm": "audio/webm", ".flac": "audio/flac"
}

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
      
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
      
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype: return k, ctype
        guessed = mimetypes.guess_extension(ctype)
        if guessed: return guessed.lower(), ctype
              
    return ".mp3", "audio/mpeg"


# --- PROMPT BUILDER ---

def build_prompt(language_label: str) -> str:
    return f"""
Transcribe this call in Hindi and English exactly as spoken. PUT LINE BREAKS PROPERLY in the output

SPEAKER RULES:
1. Identify 'Agent:' vs 'Customer:' contextually.
2. Use EXACTLY these labels: 'Agent:', 'Customer:'.
3. If unclear, fallback to 'Speaker 1:', 'Speaker 2:'.

STRICT OUTPUT FORMAT:
[0ms-1500ms] Agent: Hello sir.
[1500ms-3000ms] Customer: Hi I need help.

LANGUAGE:
- Hindi words in Hinglish (Latin script). NO Devanagari.
- Context: {language_label}

FINAL OUTPUT RULES - **MOST IMPORTANT
- HAVE PROPER LINE BREAKS IN CUSTOMER AND AGENT ALWAYS. **line must start from a new line when agent or customer speaks in the output**
- Return ONLY the transcript and ask NO QUESTION and no other text, JUST THE TRANSCRIPT AND HAVE PROPER LINE BREAKS IN CUSTOMER AND AGENT ALWAYS
- AGAIN SAYING, PUT LINE BREAKS PROPERLY.
"""


# --- DATA CONSOLIDATION ---

def consolidate_data_by_mobile(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by=['mobile_number', 'date'], ascending=[True, False])
      
    unique_mobiles = df['mobile_number'].unique()
    final_rows = []

    for mobile in unique_mobiles:
        group = df[df['mobile_number'] == mobile]
        valid_rows = group[group['recording_url'].notna() & (group['recording_url'] != "")]
        
        if not valid_rows.empty:
            row = valid_rows.iloc[0].copy()
            row['processing_action'] = 'TRANSCRIBE'
        else:
            row = group.iloc[0].copy()
            row['processing_action'] = 'SKIP'
            outcomes = group['outcome'].fillna("").astype(str).tolist()
            history = " | ".join([x.strip() for x in outcomes if x.strip()])
            row['transcript'] = f"No Audio. History: {history}"
            row['status'] = '⚠️ Skipped'
            row['error'] = 'No recording found'
        
        final_rows.append(row)
      
    return pd.DataFrame(final_rows).reset_index(drop=True)


# --- CORE: ASYNC WORKER (SUBMIT -> POLL) ---

def process_single_row(index: int, row: pd.Series, prompt_template: str, api_key: str) -> Dict[str, Any]:
    mobile = str(row.get("mobile_number", "Unknown"))
    result = {
        "index": index, 
        "mobile_number": mobile, 
        "recording_url": row.get("recording_url"),
        "transcript": row.get("transcript", ""), 
        "status": row.get("status", "Pending"), 
        "error": row.get("error", None),
        "raw_json": {}  # Stores raw API responses for inspection
    }

    if row.get("processing_action") == "SKIP":
        return result

    audio_url = row.get("recording_url")
    tmp_path = None

    try:
        # 1. Download
        parsed = urlparse(audio_url)
        r = make_request_with_retry("GET", audio_url, stream=True)
        if r.status_code != 200: raise Exception(f"Download failed: {r.status_code}")

        ext, mime = detect_extension_and_mime(parsed.path, r.headers.get("content-type"))
        expected_size = r.headers.get("content-length")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk: tmp.write(chunk)
            tmp_path = tmp.name

        if expected_size and os.path.getsize(tmp_path) < int(expected_size):
            raise Exception("Incomplete download")

        # 2. SUBMIT JOB (Async Header + Auth + New Chat)
        job_id = None
        with open(tmp_path, "rb") as f:
            files = {'file': (f"audio{ext}", f, mime)}
            data = {'provider': 'gemini', 'prompt': prompt_template}
            # CRITICAL: Send Async Mode Header AND API Key AND New Chat
            headers = {
                'x-async-mode': 'true',
                'x-api-key': api_key,
                'new-chat': 'true'  # Added as requested
            }
            
            # Use allow_redirects=False to catch if we are being sent to a login page
            sub_resp = requests.post(CUSTOM_API_URL, files=files, data=data, headers=headers, timeout=30, allow_redirects=True)
            
            # --- ROBUST RESPONSE HANDLING ---
            # Check if response is HTML (Login Page) disguised as 200 OK
            content_type = sub_resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise Exception("Auth Failed: Server returned Login Page instead of JSON. Check API Key.")

            # Accept 200 (OK) or 202 (Accepted) as success if they contain a Job ID
            if sub_resp.status_code in [200, 202]:
                try:
                    resp_json = sub_resp.json()
                    # Store submission response in raw_json
                    result["raw_json"]["submission"] = resp_json
                    
                    job_id = resp_json.get("jobId")
                    if not job_id:
                          # Fallback: maybe it returned the answer directly?
                          if "result" in resp_json:
                              result["transcript"] = resp_json["result"].get("answer", "")
                              result["status"] = "✅ Success (Direct)"
                              return result
                          else:
                              raise Exception(f"No Job ID in response: {str(resp_json)}")
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {sub_resp.text[:100]}")
            
            elif sub_resp.status_code in [401, 403]:
                raise Exception("Authentication Failed: Check API Key")
            else:
                raise Exception(f"Submission Error {sub_resp.status_code}: {sub_resp.text[:200]}")

        # 3. POLL FOR RESULTS (If we got a Job ID)
        if job_id:
            start_time = time.time()
            # Ensure we strip trailing slash and append /ask/{jobId}
            poll_url = f"{CUSTOM_API_URL.rstrip('/ask')}/ask/{job_id}" 
            # Send API key and new-chat header during polling too
            poll_headers = {
                'x-api-key': api_key,
                'new-chat': 'true'
            }
            
            while time.time() - start_time < 900: # 15 min timeout
                try:
                    poll_resp = requests.get(poll_url, headers=poll_headers, timeout=10)
                    
                    # Check for Auth Fail during polling
                    if poll_resp.headers.get("Content-Type", "").startswith("text/html"):
                         raise Exception("Auth Failed during polling (Login Page returned)")
                    
                    if poll_resp.status_code == 200:
                        job_data = poll_resp.json()
                        # Store poll response in raw_json
                        result["raw_json"]["latest_poll"] = job_data
                        
                        status = job_data.get("status")
                        
                        if status == "completed":
                            ans = job_data.get("result", {}).get("answer", "")
                            result["transcript"] = ans
                            result["status"] = "✅ Success"
                            break
                        elif status == "failed":
                            raise Exception(f"Job failed: {job_data.get('error')}")
                        
                        # If queued/processing, wait
                        time.sleep(3) 
                    else:
                        if poll_resp.status_code in [401, 403]:
                            raise Exception("Auth Error during Polling")
                        time.sleep(3)
                except Exception as e:
                    logger.warning(f"Polling error: {e}")
                    if "Auth" in str(e): raise e
                    time.sleep(3)
            else:
                raise Exception("Polling timed out (15m)")

    except Exception as e:
        result["status"] = "❌ Failed"
        result["error"] = str(e)
        result["transcript"] = f"Error: {str(e)}"
      
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    return result


# --- UI HELPERS ---

def merge_results_with_original(df_orig: pd.DataFrame, results: list) -> pd.DataFrame:
    res_df = pd.DataFrame(sorted(results, key=lambda r: r["index"]))
    # Columns to update
    cols = ["transcript", "status", "error", "raw_json"]
    # Drop existing cols if they exist to avoid conflict
    df_base = df_orig.drop(columns=[c for c in cols if c in df_orig.columns])
    merged = df_base.merge(res_df[["index"] + cols], left_index=True, right_on="index", how="left")
    return merged.drop(columns=["index"]) if "index" in merged.columns else merged

def colorize_transcript_html(text: str) -> str:
    """
    Converts transcript text into a beautiful chat interface.
    """
    if not isinstance(text, str) or not text.strip():
        return "<div class='chat-bubble other-bubble'>No transcript available</div>"

    html_parts = []
    lines = text.splitlines()
      
    for line in lines:
        clean = line.strip()
        if not clean: continue
        
        lower_line = clean.lower()
        escaped_text = html.escape(clean)
        
        # Determine Bubble Type
        bubble_class = "other-bubble"
        speaker_label = "System"
        
        if "agent:" in lower_line or "speaker 1:" in lower_line:
            bubble_class = "agent-bubble"
            speaker_label = "Agent"
        elif "customer:" in lower_line or "speaker 2:" in lower_line:
            bubble_class = "customer-bubble"
            speaker_label = "Customer"
            
        # Build HTML
        html_parts.append(
            f"""<div class='chat-bubble {bubble_class}'>
                <span class='bubble-meta'>{speaker_label}</span>
                {escaped_text}
            </div>"""
        )
            
    return "".join(html_parts)


# --- MAIN APP ---

def main():
    if "final_df" not in st.session_state: st.session_state.final_df = pd.DataFrame()

    with st.sidebar:
        st.header("Configuration")
        st.info(f"API Endpoint:\n{urlparse(CUSTOM_API_URL).hostname}")
        
        # API Key Input
        api_key = st.text_input("API Key", type="password", help="Enter the x-api-key for authentication")
        
        # Concurrency slider
        max_workers = st.slider("Threads", 1, 64, 12, help="Parallel jobs submitted to API.")
        
        st.divider()
        language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
        theme_choice = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)

    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

    st.write("### 📂 Upload Call Data")
    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)
      
    progress_bar = st.empty()
    status_text = st.empty()
    result_placeholder = st.empty()
    start_btn = st.button("🚀 Start Batch Processing", type="primary")

    if start_btn:
        if not uploaded_files:
            st.error("Upload a file first.")
            st.stop()
            
        if not api_key:
            st.error("🔒 API Key is required to proceed.")
            st.stop()

        all_dfs = [pd.read_excel(f) for f in uploaded_files]
        df_consolidated = consolidate_data_by_mobile(pd.concat(all_dfs, ignore_index=True))
        
        prompt = build_prompt(language_mode)
        total = len(df_consolidated)
        
        status_text.info(f"Queuing {total} calls with {max_workers} threads...")
        progress_bar.progress(0.0)
        
        processed_results = []
        
        # Pass api_key to the worker function
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_row, i, r, prompt, api_key): i for i, r in df_consolidated.iterrows()}
            
            completed = 0
            for future in as_completed(futures):
                processed_results.append(future.result())
                completed += 1
                progress_bar.progress(completed / total)
                status_text.markdown(f"Processed **{completed}/{total}** calls.")
                
                # Live Preview
                recent = sorted(processed_results, key=lambda x: x["index"])[-3:]
                if recent:
                    result_placeholder.dataframe(
                        pd.DataFrame(recent)[["mobile_number", "status", "transcript"]],
                        hide_index=True
                    )

        st.session_state.final_df = merge_results_with_original(df_consolidated, processed_results)
        status_text.success("Done!")

    # --- VIEWER ---
    final_df = st.session_state.final_df
    if not final_df.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("## 🎛️ Transcript Browser")
        
        # Filters
        c1, c2, c3 = st.columns([3, 1, 1])
        search_txt = c1.text_input("Search")
        status_filter = c2.selectbox("Status", ["All", "Success", "Failed", "Skipped"])
        per_page = c3.selectbox("Per Page", [5, 10, 20], index=1)
        
        # Filter Logic
        view = final_df.copy()
        if status_filter != "All":
            view = view[view["status"].str.contains(status_filter, case=False, na=False)]
        if search_txt:
            view = view[view.astype(str).apply(lambda x: x.str.contains(search_txt, case=False).any(), axis=1)]
            
        # Pagination
        total_items = len(view)
        st.caption(f"Showing {total_items} results")
        page_current = st.number_input("Page", 1, max(1, math.ceil(total_items/per_page)), 1)
        
        # Export
        output = BytesIO()
        # Drop raw_json before export to keep Excel clean
        export_df = view.drop(columns=["raw_json"], errors='ignore')
        export_df.to_excel(output, index=False)
        st.download_button("📥 Download Excel", output.getvalue(), "transcripts.xlsx")
        
        # Render
        start = (page_current - 1) * per_page
        for _, row in view.iloc[start : start + per_page].iterrows():
            with st.expander(f"{row['mobile_number']} — {row['status']}"):
                
                # --- NEW FEATURE: Info Button for Raw Response ---
                col_info, col_link = st.columns([0.05, 0.95])
                with col_info:
                    with st.popover("ℹ️", help="View Raw API Response"):
                        st.markdown("#### Raw API Data")
                        st.json(row.get("raw_json", {}))
                
                with col_link:
                    st.markdown(f"**URL:** `{row['recording_url']}`")
                
                html_view = colorize_transcript_html(row.get("transcript", ""))
                st.markdown(f"<div class='transcript-box'>{html_view}</div>", unsafe_allow_html=True)
                if row.get("error"): st.error(row["error"])

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
