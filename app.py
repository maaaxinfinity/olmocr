from flask import Flask, request, jsonify, send_file
import os
import tempfile
import subprocess
import json
import base64
from werkzeug.utils import secure_filename
import shutil
import time
import threading
import glob
import requests # Import requests library
import logging
import sys

app = Flask(__name__)

# --- Logging Setup ---
# Force removal of existing handlers to ensure ours takes precedence
for handler in app.logger.handlers[:]:
    app.logger.removeHandler(handler)

# Add our StreamHandler
handler = logging.StreamHandler(sys.stderr) # Explicitly use stderr
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Set a default level (will be overridden by LOG_LEVEL env var later in __main__)
app.logger.setLevel(logging.INFO)
# We also need to set the handler level initially, although it will be reset in __main__
handler.setLevel(logging.INFO)

# Prevent messages from propagating to the root logger
app.logger.propagate = False
# --------------------

# --- Configuration ---
# Create work directory
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localworkspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# Specify model path
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/model_Q4/olmocr")

# Dify base URL (for downloading files)
# IMPORTANT: Set this environment variable if Dify is not running on localhost:3000
DIFY_BASE_URL = os.environ.get("DIFY_BASE_URL", "http://localhost:3000")

# Task status flag and lock
current_task = None
task_lock = threading.Lock()
# ---------------------

# Helper function to download file from Dify URL
def download_dify_file(file_url_rel, filename_orig, target_dir):
    # Now receives relative URL and original filename separately
    if not file_url_rel or not filename_orig:
        app.logger.error("download_dify_file called with missing file_url_rel or filename_orig") # Added log
        raise ValueError("download_dify_file requires both file_url_rel and filename_orig")

    # Sanitize filename for saving, trying to preserve original extension
    filename_secure = secure_filename(filename_orig)
    original_ext = os.path.splitext(filename_orig.lower())[1] or '.pdf' # Default to .pdf if no ext
    # If secure_filename removed the extension, add the original one back
    if not filename_secure or os.path.splitext(filename_secure.lower())[1] != original_ext:
        base_name = os.path.splitext(filename_secure)[0] if filename_secure else filename_orig # Use original if secure is empty
        filename = base_name + original_ext
    else:
        filename = filename_secure

    app.logger.info(f"Sanitized filename: '{filename_orig}' -> '{filename}'") # Added log

    # Construct full download URL, ensuring no double slashes
    full_download_url = f"{DIFY_BASE_URL.rstrip('/')}/{file_url_rel.lstrip('/')}"
    target_path = os.path.join(target_dir, filename)
    app.logger.info(f"Attempting to download from URL: {full_download_url}") # Added log
    app.logger.info(f"Saving to local path: {target_path}") # Added log

    try:
        response = requests.get(full_download_url, stream=True, timeout=60)
        app.logger.info(f"Download request sent. Response status code: {response.status_code}") # Added log
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        app.logger.info(f"Successfully downloaded and saved '{filename_orig}' as '{filename}' to {target_path}") # Enhanced log
        # Return the saved path and the sanitized/saved filename
        return target_path, filename
    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout occurred while downloading {filename_orig} from {full_download_url}")
        raise ConnectionError(f"Timeout downloading file {filename_orig} from Dify {full_download_url}")
    except requests.exceptions.HTTPError as e:
        app.logger.error(f"HTTP Error {e.response.status_code} while downloading {filename_orig} from {full_download_url}: {e.response.text[:500]}") # Log response text excerpt
        raise ConnectionError(f"HTTP Error {e.response.status_code} downloading file {filename_orig} from Dify {full_download_url}")
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Failed to download file {filename_orig} from Dify {full_download_url}: {e}", exc_info=True) # Add stack trace
        raise ConnectionError(f"Failed to download file {filename_orig} from Dify {full_download_url}: {e}")
    except IOError as e:
        app.logger.error(f"Failed to save downloaded file {filename_orig} to {target_path}: {e}", exc_info=True) # Add stack trace
        raise IOError(f"Failed to save downloaded file {filename_orig}: {e}")
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during download/saving of {filename_orig}: {e}", exc_info=True) # Catch-all with trace
        raise IOError(f"Failed to save downloaded file {filename_orig}: {e}")

# Helper function to save base64 encoded file (Adding logging here too for completeness)
def save_base64_file(b64_string, filename_req, target_dir):
    app.logger.info(f"Attempting to save base64 data for requested filename: {filename_req}") # Added log
    try:
        filename = secure_filename(filename_req)
        # Ensure filename has a pdf extension if missing
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        target_path = os.path.join(target_dir, filename)
        app.logger.info(f"Saving base64 data to local path: {target_path}") # Added log
        pdf_data = base64.b64decode(b64_string)
        with open(target_path, 'wb') as f:
            f.write(pdf_data)
        app.logger.info(f"Successfully saved base64 file as '{filename}' from request '{filename_req}'") # Enhanced log
        # Return the original filename for mapping, and the saved path
        return target_path, filename_req
    except base64.binascii.Error as e:
        app.logger.error(f"Invalid base64 encoding for file {filename_req}: {e}") # Added log
        raise ValueError(f"Invalid base64 encoding for file {filename_req}: {e}")
    except IOError as e:
        app.logger.error(f"Failed to save base64 file {filename_req} to {target_path}: {e}", exc_info=True) # Added log with trace
        raise IOError(f"Failed to save base64 file {filename_req}: {e}")
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during base64 saving of {filename_req}: {e}", exc_info=True) # Catch-all with trace
        raise IOError(f"Failed to save base64 file {filename_req}: {e}")

# Define tool schema
# Simplified Schema - One endpoint handles all
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_pdf",
            "description": "使用olmocr分析单个PDF文件. 需要提供文件URL和原始文件名.",
            "parameters": {
                # Schema now uses GET parameters, defined in schema file
                # This is just a placeholder now
                "type": "object",
                "properties": {
                    "file_url": {"type": "string"},
                    "filename": {"type": "string"}
                },
                "required": ["file_url", "filename"]
            }
        }
    }
]

@app.route("/tools", methods=["GET"])
def get_tools():
    """返回工具的schema定义 (需手动更新schema文件以匹配GET参数)"""
    return jsonify(TOOLS_SCHEMA)

@app.route("/status", methods=["GET"])
def get_status():
    """返回当前处理状态"""
    global current_task
    with task_lock:
        if current_task:
            task_info_safe = {k: v for k, v in current_task.items() if k != 'command'}
            return jsonify({
                "is_processing": True,
                "status": "busy",
                "task_info": task_info_safe
            })
        else:
            return jsonify({
                "is_processing": False,
                "status": "ready",
                "task_info": None
            })

# --- Main Analysis Endpoint (Now GET) --- 
@app.route("/analyze_pdf", methods=["GET"])
def analyze_pdf():
    """处理单个PDF分析请求 (通过URL和文件名参数). OLMOCR仍顺序执行."""
    request_start_time = time.time() # Log request duration
    app.logger.info("-" * 40) # Separator for new request
    app.logger.info(f"Received GET request for /analyze_pdf")
    global current_task

    # Extract parameters from query string
    file_url = request.args.get('file_url')
    original_filename = request.args.get('filename')

    app.logger.info(f"Received Parameters: file_url='{file_url}', filename='{original_filename}'") # Log received params

    if not file_url or not original_filename:
        app.logger.warning("Request rejected: Missing 'file_url' or 'filename' query parameter.") # Added log
        return jsonify({"error": "缺少必需的查询参数 'file_url' 或 'filename'"}), 400

    app.logger.info(f"Attempting to acquire task lock for filename: {original_filename}")
    with task_lock:
        app.logger.info("Task lock acquired.")
        if current_task:
            task_info_safe = {k: v for k, v in current_task.items() if k != 'command'}
            app.logger.warning(f"Server busy, rejecting request for {original_filename}. Current task: {task_info_safe}") # Added log
            return jsonify({
                "error": "服务器忙，当前有任务正在处理中",
                "status": "busy",
                "task_info": task_info_safe
            }), 429

        # Initialize task status for a single file
        current_task = {
            "type": "single_pdf_analysis_get",
            "start_time": time.time(),
            "status": "initializing",
            "input_filename": original_filename, # Log input filename
            "input_url": file_url
        }
        app.logger.info(f"Initialized new task: {current_task}") # Added log

    # Create a main temporary directory for this single file task
    main_task_workspace = None
    saved_pdf_path = None # Define outside try block
    try:
        # --- Stage 1: Prepare the single input file --- 
        main_task_workspace = tempfile.mkdtemp(dir=WORKSPACE_DIR)
        app.logger.info(f"Created main task workspace: {main_task_workspace}")
        pdf_save_dir = os.path.join(main_task_workspace, "pdf_to_process") # Dir for one pdf
        os.makedirs(pdf_save_dir, exist_ok=True)
        app.logger.info(f"Created PDF save subdirectory: {pdf_save_dir}")

        with task_lock:
            current_task["status"] = "downloading_pdf"
            current_task["current_file_preparing"] = original_filename
            current_task["workspace"] = main_task_workspace # Log workspace
            app.logger.info(f"Task status updated: {current_task['status']}")
        
        app.logger.info(f"Starting download for: {original_filename}")
        # Download the file using the provided URL and filename
        saved_pdf_path, saved_filename = download_dify_file(file_url, original_filename, pdf_save_dir)
        app.logger.info(f"File download successful. Saved path: {saved_pdf_path}, Saved filename: {saved_filename}")

        with task_lock:
            # Update status after successful download/save
            current_task["status"] = "file_prepared"
            current_task["prepared_filepath"] = saved_pdf_path # Log path
            del current_task["current_file_preparing"] # Clear prep state
            app.logger.info(f"Task status updated: {current_task['status']}")
        # ---------------------------------------------

        # --- Stage 2: Process the single prepared file --- 
        file_status = "failed" # Default status for this file
        file_content = None
        error_detail = None
        result_files = [] # Initialize

        with task_lock:
            current_task["status"] = "processing_pdf"
            current_task["current_file_processing"] = original_filename
            app.logger.info(f"Task status updated: {current_task['status']}")

        # OLMOCR processing is still sequential relative to other API calls due to the lock
        # Use the main_task_workspace for olmocr run (it contains the pdf_save_dir)
        olmocr_output_path = os.path.join(main_task_workspace, "results")
        os.makedirs(olmocr_output_path, exist_ok=True)
        app.logger.info(f"Created olmocr output directory: {olmocr_output_path}")

        try:
            # Execute olmocr command for the single file in the workspace
            # IMPORTANT: The olmocr script expects the DIRECTORY containing PDFs, not the PDF itself.
            # And it processes ALL PDFs in that directory (or specified by --pdfs relative to it).
            # Since we only have one PDF, we pass the pdf_save_dir as the input path
            # and optionally use --pdfs to be explicit (though maybe not needed if it's the only pdf there)
            cmd = [
                "python", "-m", "olmocr.pipeline",
                pdf_save_dir, # Directory containing the PDF to process
                #"--pdfs", saved_filename, # Use saved_filename relative to pdf_save_dir
                "--output", olmocr_output_path, # Explicitly set output directory
                "--model", MODEL_PATH
            ]
            app.logger.info(f"Running olmocr command for '{original_filename}': {' '.join(cmd)}")
            with task_lock: # Log command within task status
                current_task['command'] = ' '.join(cmd)

            process_start_time = time.time() # Log OLMOCR duration
            process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600) # Increased timeout to 10 mins
            process_duration = time.time() - process_start_time
            app.logger.info(f"olmocr process for '{original_filename}' finished in {process_duration:.2f} seconds with return code {process.returncode}")
            if process.stdout:
                app.logger.debug(f"olmocr stdout for {original_filename}:\n{process.stdout}") # Log stdout at debug level
            if process.stderr:
                # Log stderr as warning or error based on return code
                log_level = logging.WARNING if process.returncode == 0 else logging.ERROR
                app.logger.log(log_level, f"olmocr stderr for {original_filename}:\n{process.stderr}")

            if process.returncode != 0:
                error_detail = f"olmocr处理失败 (Code: {process.returncode}): {process.stderr}" # Keep original error format
                app.logger.error(error_detail)
                raise RuntimeError(error_detail)

            # Find the result file in the workspace result dir
            # Result filename usually contains the original PDF name (or sanitized version)
            # Example: output_1.证据材料卷（第一册）（公安卷）_可搜索.jsonl
            # Let's be more robust and just find any output_*.jsonl file
            result_files = glob.glob(os.path.join(olmocr_output_path, "output_*.jsonl"))
            app.logger.info(f"Searching for result file in {olmocr_output_path}. Found: {result_files}") # Added log
            if not result_files:
                error_detail = f"未在 {olmocr_output_path} 中找到olmocr结果文件 (output_*.jsonl)" # More specific error
                app.logger.error(error_detail)
                raise FileNotFoundError(error_detail)
            elif len(result_files) > 1:
                 app.logger.warning(f"发现多个结果文件，将使用第一个: {result_files[0]}")

            result_file_path = result_files[0]
            app.logger.info(f"Reading results from: {result_file_path}") # Added log
            # Read result content
            with open(result_file_path, 'r', encoding='utf-8') as f:
                result_content = f.read().strip()
            app.logger.info(f"Successfully read {len(result_content)} characters from result file.")

            # Parse results
            app.logger.info("Parsing result content (JSONL)...")
            file_results_parsed = []
            lines = result_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    try:
                        file_results_parsed.append(json.loads(line))
                    except json.JSONDecodeError as json_err:
                        app.logger.error(f"Failed to parse JSON line {i+1} in {result_file_path}: {json_err}")
                        app.logger.error(f"Problematic line content: {line[:200]}...")
                        # Decide whether to skip or raise an error - let's skip for now
                        # raise ValueError(f"结果文件 {result_file_path} 中包含无效的JSON行")
            app.logger.info(f"Successfully parsed {len(file_results_parsed)} lines from result file.")

            # Extract text
            app.logger.info("Extracting text from parsed results...")
            extracted_text = ""
            if file_results_parsed:
                for res in file_results_parsed:
                    if 'text' in res:
                        extracted_text += res['text'] + "\n\n"
            app.logger.info(f"Extracted {len(extracted_text)} characters of text.")

            file_content = extracted_text.strip()
            file_status = "success"
            app.logger.info(f"Processing successful for {original_filename}")

        except FileNotFoundError as e:
            # Already logged above, just set error detail
            error_detail = str(e)
            file_status = "error_finding_result"
        except RuntimeError as e: # Catch olmocr execution error
            # Already logged above, just set error detail
            error_detail = str(e)
            file_status = "error_olmocr_execution"
        except subprocess.TimeoutExpired as e:
            error_detail = f"处理文件 '{original_filename}' 时超时 ({e.timeout} 秒)"
            app.logger.error(error_detail)
            file_status = "error_timeout"
        except json.JSONDecodeError as e: # Catch potential error during manual parsing if needed
            error_detail = f"解析结果文件 '{result_file_path}' 时出错: {e}"
            app.logger.error(error_detail, exc_info=True)
            file_status = "error_parsing_result"
        except Exception as e:
            error_detail = f"处理文件 '{original_filename}' 时发生意外错误: {e}"
            app.logger.error(error_detail, exc_info=True) # Log full trace for unexpected errors
            file_status = "error_unexpected"
        finally:
             with task_lock:
                 if current_task and "current_file_processing" in current_task:
                     del current_task["current_file_processing"]
                     app.logger.info("Cleared 'current_file_processing' from task status.")
        # ---------------------------------------------

        # --- Stage 3: Prepare final response --- 
        app.logger.info(f"Preparing final response for {original_filename}. Status: {file_status}")
        final_result = {
            "filename": original_filename, # Use the original filename provided in the request
            "status": file_status,
            "content": file_content if file_status == "success" else None,
            "error": error_detail if file_status != "success" else None
        }
        
        response_data = {"analysis_output": final_result}
        response_status_code = 200 if file_status == "success" else 500 # Return 500 for errors
        
        app.logger.info(f"Sending response. Status Code: {response_status_code}, Filename: {original_filename}, Result Status: {file_status}")
        if error_detail:
             app.logger.info(f"Error Detail in Response: {error_detail}")
        elif file_status == "success":
             app.logger.info(f"Response Content Length: {len(file_content) if file_content else 0}")
        
        return jsonify(response_data), response_status_code

    except ConnectionError as e:
        # Handle download/connection errors specifically
        app.logger.error(f"Connection error during file preparation for {original_filename}: {e}", exc_info=True)
        with task_lock:
            current_task = None # Release lock as task failed early
        return jsonify({
            "error": f"无法从提供的URL下载文件: {e}",
            "filename": original_filename,
            "status": "error_downloading"
        }), 500
    except ValueError as e:
        # Handle potential base64 errors if that path were used, or other value errors
        app.logger.error(f"Value error during file preparation for {original_filename}: {e}", exc_info=True)
        with task_lock:
            current_task = None
        return jsonify({
            "error": f"文件准备阶段出错: {e}",
            "filename": original_filename,
            "status": "error_preparation_value"
        }), 400 # Bad request if value error (e.g., bad base64)
    except IOError as e:
        # Handle file saving errors (download or base64)
        app.logger.error(f"IO error during file preparation for {original_filename}: {e}", exc_info=True)
        with task_lock:
            current_task = None
        return jsonify({
            "error": f"保存文件时出错: {e}",
            "filename": original_filename,
            "status": "error_saving_file"
        }), 500
    except Exception as e:
        # Catch-all for unexpected errors during setup or processing stages before response prep
        app.logger.error(f"Unexpected error during processing for {original_filename}: {e}", exc_info=True)
        # Ensure lock is released if an unexpected error occurs
        with task_lock:
            current_task = None
        return jsonify({"error": f"处理过程中发生意外服务器错误: {e}"}), 500
    finally:
        # --- Stage 4: Cleanup and release lock --- 
        app.logger.info("Entering finally block for cleanup.")
        with task_lock:
            if current_task:
                current_task["end_time"] = time.time()
                current_task["duration"] = current_task["end_time"] - current_task["start_time"]
                current_task["status"] = "completed" # Mark as completed regardless of internal status
                app.logger.info(f"Task completed. Final status: {current_task}")
                current_task = None # Release the lock/task slot
                app.logger.info("Task slot released.")
        
        # Clean up the main temporary workspace directory
        if main_task_workspace and os.path.exists(main_task_workspace):
            try:
                shutil.rmtree(main_task_workspace)
                app.logger.info(f"Successfully removed workspace: {main_task_workspace}")
            except OSError as e:
                app.logger.error(f"无法删除临时工作区 {main_task_workspace}: {e}")
        elif main_task_workspace:
            app.logger.warning(f"Workspace directory {main_task_workspace} not found for cleanup.")
        request_duration = time.time() - request_start_time
        app.logger.info(f"Request completed in {request_duration:.2f} seconds.")
        app.logger.info("-" * 40) # Separator end of request
# -----------------------------------------

if __name__ == '__main__':
    # Set log level from environment variable, default to INFO
    log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Set the level for the logger AND its handlers
    app.logger.setLevel(log_level)
    for h in app.logger.handlers: # Should only be our handler now
        h.setLevel(log_level)

    # Check environment variable to enable debug mode
    debug_mode_env = os.environ.get('FLASK_DEBUG_MODE', '0').lower()
    debug_enabled = debug_mode_env in ['1', 'true', 'yes', 'on']

    # Log initial settings *before* app.run
    # Use level=logging.INFO explicitly here in case the logger level was set higher
    app.logger.log(logging.INFO, f"Attempting to set log level to: {logging.getLevelName(log_level)} (LOG_LEVEL='{log_level_name}')")
    app.logger.log(logging.INFO, f"Debug mode setting: {'ENABLED' if debug_enabled else 'DISABLED'} (FLASK_DEBUG_MODE='{debug_mode_env}')")
    # Log the final effective level of the logger itself
    app.logger.log(logging.INFO, f"Effective logger level set to: {logging.getLevelName(app.logger.getEffectiveLevel())}")

    # Run the Flask app
    # Flask might adjust logging when debug=True
    app.run(host='0.0.0.0', port=5555, debug=debug_enabled) 