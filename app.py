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

app = Flask(__name__)

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
        raise ValueError("download_dify_file requires both file_url_rel and filename_orig")

    filename = secure_filename(filename_orig) # Sanitize for saving
    # Ensure filename has the correct extension if missing from secure_filename result
    original_ext = os.path.splitext(filename_orig.lower())[1] or '.pdf' # Default to .pdf if no ext
    if not filename.lower().endswith(original_ext):
         filename = os.path.splitext(filename)[0] + original_ext

    download_url = f"{DIFY_BASE_URL.rstrip('/')}{file_url_rel}"
    target_path = os.path.join(target_dir, filename)

    try:
        app.logger.info(f"Downloading from {download_url} to {target_path}")
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        app.logger.info(f"Successfully downloaded {filename_orig}")
        # Return the saved path and the sanitized/saved filename
        return target_path, filename
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download file {filename_orig} from Dify {download_url}: {e}")
    except Exception as e:
        raise IOError(f"Failed to save downloaded file {filename_orig}: {e}")

# Helper function to save base64 encoded file
def save_base64_file(b64_string, filename_req, target_dir):
    try:
        filename = secure_filename(filename_req)
        # Ensure filename has a pdf extension if missing
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        target_path = os.path.join(target_dir, filename)
        pdf_data = base64.b64decode(b64_string)
        with open(target_path, 'wb') as f:
            f.write(pdf_data)
        app.logger.info(f"Successfully saved base64 file as {filename}")
        # Return the original filename for mapping, and the saved path
        return target_path, filename_req
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 encoding for file {filename_req}: {e}")
    except Exception as e:
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
    global current_task

    # Extract parameters from query string
    file_url = request.args.get('file_url')
    original_filename = request.args.get('filename')

    app.logger.info(f"Received request for /analyze_pdf (GET)")
    app.logger.debug(f"Query Parameters: file_url={file_url}, filename={original_filename}")

    if not file_url or not original_filename:
        return jsonify({"error": "缺少必需的查询参数 'file_url' 或 'filename'"}, 400)

    with task_lock:
        if current_task:
            task_info_safe = {k: v for k, v in current_task.items() if k != 'command'}
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

    # Create a main temporary directory for this single file task
    main_task_workspace = None 
    try:
        # --- Stage 1: Prepare the single input file --- 
        main_task_workspace = tempfile.mkdtemp(dir=WORKSPACE_DIR)
        pdf_save_dir = os.path.join(main_task_workspace, "pdf_to_process") # Dir for one pdf
        os.makedirs(pdf_save_dir, exist_ok=True)

        with task_lock:
            current_task["status"] = "downloading_pdf"
            current_task["current_file_preparing"] = original_filename
        
        # Download the file using the provided URL and filename
        saved_pdf_path, saved_filename = download_dify_file(file_url, original_filename, pdf_save_dir)
        app.logger.info(f"File prepared at: {saved_pdf_path}")

        with task_lock:
            # Update status after successful download/save
            current_task["status"] = "file_prepared"
            current_task["prepared_filepath"] = saved_pdf_path # Log path
            del current_task["current_file_preparing"] # Clear prep state
        # ---------------------------------------------

        # --- Stage 2: Process the single prepared file --- 
        file_status = "failed" # Default status for this file
        file_content = None
        error_detail = None

        with task_lock:
            current_task["status"] = "processing_pdf"
            current_task["current_file_processing"] = original_filename

        # OLMOCR processing is still sequential relative to other API calls due to the lock
        # Use the main_task_workspace for olmocr run (it contains the pdf_save_dir)
        olmocr_output_path = os.path.join(main_task_workspace, "results")
        os.makedirs(olmocr_output_path, exist_ok=True)

        try:
            # Execute olmocr command for the single file in the workspace
            cmd = [
                "python", "-m", "olmocr.pipeline",
                main_task_workspace, # The workspace containing the pdf subdir
                "--pdfs", saved_pdf_path, # Point directly to the saved PDF
                "--model", MODEL_PATH
            ]
            app.logger.info(f"Running olmocr for {original_filename}: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300) # Add timeout
            app.logger.info(f"olmocr process for {original_filename} finished with code {process.returncode}")
            if process.stderr:
                app.logger.warning(f"olmocr stderr for {original_filename}:\n{process.stderr}")

            if process.returncode != 0:
                raise RuntimeError(f"olmocr处理失败 (Code: {process.returncode}): {process.stderr}")

            # Find the result file in the workspace result dir
            result_files = glob.glob(os.path.join(olmocr_output_path, "output_*.jsonl"))
            if not result_files:
                raise FileNotFoundError("未找到olmocr结果文件")

            # Read result content
            with open(result_files[0], 'r', encoding='utf-8') as f:
                result_content = f.read().strip()

            # Parse results
            file_results_parsed = []
            for line in result_content.split('\n'):
                if line.strip():
                    file_results_parsed.append(json.loads(line))

            # Extract text
            extracted_text = ""
            if file_results_parsed:
                for res in file_results_parsed:
                    if 'text' in res:
                        extracted_text += res['text'] + "\n\n"

            file_content = extracted_text.strip()
            file_status = "success"

        except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as e:
            error_detail = f"处理文件 '{original_filename}' 时出错: {e}"
            app.logger.error(error_detail)
        except Exception as e:
            error_detail = f"处理文件 '{original_filename}' 时发生意外错误: {e}"
            app.logger.error(error_detail, exc_info=True)
        # ---------------------------------------------

        # --- Stage 3: Prepare final response --- 
        analysis_result = {
            # Use original filename provided in request
            "filename": original_filename,
            "content": file_content,
            "status": file_status,
            "error": error_detail, # Will be None on success
            "processing_time": time.time() - current_task["start_time"]
        }
        response_data = {"analysis_output": analysis_result}

        # Clean up the main task workspace
        shutil.rmtree(main_task_workspace, ignore_errors=True)
        main_task_workspace = None

        # Reset task status
        with task_lock:
            current_task = None

        return jsonify(response_data)

    except (ValueError, ConnectionError, IOError) as e: # Errors during initial file prep
        err_msg = f"文件准备错误 (url={file_url}, filename={original_filename}): {e}"
        app.logger.error(err_msg)
        with task_lock:
            current_task = None
        if main_task_workspace and os.path.exists(main_task_workspace):
             shutil.rmtree(main_task_workspace, ignore_errors=True)
        return jsonify({"error": err_msg}), 400
    except Exception as e: # Catch-all for unexpected errors
        err_msg = f"服务器内部错误: {e}"
        app.logger.error(err_msg, exc_info=True)
        with task_lock:
            current_task = None
        if main_task_workspace and os.path.exists(main_task_workspace):
             shutil.rmtree(main_task_workspace, ignore_errors=True)
        return jsonify({"error": err_msg}), 500

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s [%(threadName)s]')
    app.run(host="0.0.0.0", port=5555, debug=True) 