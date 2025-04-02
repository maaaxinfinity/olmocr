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
def download_dify_file(file_info, target_dir):
    if 'url' not in file_info or 'filename' not in file_info:
        raise ValueError("Invalid Dify file info object, missing 'url' or 'filename'")

    filename = secure_filename(file_info['filename'])
    # Ensure filename has the correct extension if missing from secure_filename result
    original_ext = os.path.splitext(file_info['filename'].lower())[1]
    if not filename.lower().endswith(original_ext):
         filename = os.path.splitext(filename)[0] + original_ext

    download_url = f"{DIFY_BASE_URL.rstrip('/')}{file_info['url']}"
    target_path = os.path.join(target_dir, filename)

    try:
        app.logger.info(f"Downloading from {download_url} to {target_path}")
        response = requests.get(download_url, stream=True, timeout=60) # Add timeout
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        app.logger.info(f"Successfully downloaded {file_info['filename']}")
        # Return the original filename for mapping, and the saved path
        return target_path, file_info['filename']
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download file {file_info['filename']} from Dify {download_url}: {e}")
    except Exception as e:
        raise IOError(f"Failed to save downloaded file {file_info['filename']}: {e}")

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
            "description": "使用olmocr顺序分析一个或多个PDF文件并提取内容. 可通过Dify文件对象列表或base64列表提供.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_files": { # Dify input
                        "type": "array",
                        "description": "(推荐) Dify文件对象列表 (至少一个文件).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "filename": {"type": "string"}
                            },
                            "required": ["url", "filename"]
                        },
                        "minItems": 1
                    },
                     "pdf_list": { # Direct input (alternative)
                        "type": "array",
                        "description": "(备选) PDF文件列表, 每个包含filename和pdf_base64 (若提供'input_files'则忽略此项).",
                        "items": {
                             "type": "object",
                             "properties": {
                                 "filename": {"type": "string"},
                                 "pdf_base64": {"type": "string", "format": "byte"}
                             },
                              "required": ["filename", "pdf_base64"],
                              "minItems": 1
                         }
                    }
                },
                # API will prioritize 'input_files' if both are present
            }
        }
    }
]

@app.route("/tools", methods=["GET"])
def get_tools():
    """返回工具的schema定义"""
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

# --- Main Analysis Endpoint --- 
@app.route("/analyze_pdf", methods=["POST"])
def analyze_pdf():
    """处理单个或多个PDF分析请求 (Dify文件格式或Base64列表), 顺序执行olmocr."""
    global current_task

    with task_lock:
        if current_task:
            task_info_safe = {k: v for k, v in current_task.items() if k != 'command'}
            return jsonify({
                "error": "服务器忙，当前有任务正在处理中",
                "status": "busy",
                "task_info": task_info_safe
            }), 429

        # Initialize task status
        current_task = {
            "type": "sequential_pdf_analysis",
            "start_time": time.time(),
            "status": "initializing"
        }

    # Create a main temporary directory for the whole request
    main_task_workspace = None 
    files_to_process_info = [] # List of dicts: {'path': saved_path, 'original_filename': original_fname}
    total_files_requested = 0
    input_source = None
    # Expecting the Dify input variable name as the top-level key
    dify_input_key = 'input_files' 

    try:
        # Log raw request body and headers for debugging
        raw_body = request.get_data()
        app.logger.info(f"Received request for /analyze_pdf")
        app.logger.debug(f"Request Headers: {request.headers}")
        try:
            # Try decoding raw body as UTF-8 for logging, ignore errors if not decodable
            app.logger.debug(f"Raw Request Body: {raw_body.decode('utf-8', errors='ignore')}")
        except Exception:
             app.logger.debug(f"Raw Request Body (bytes): {raw_body}")
        
        data = request.json
        app.logger.info(f"Parsed JSON data: {json.dumps(data, indent=2) if data else 'None'}") # Pretty print JSON
        
        if not data:
            raise ValueError("请求体不能为空或JSON解析失败")

        # --- Stage 1: Prepare all input files (Download or Save Base64) ---
        main_task_workspace = tempfile.mkdtemp(dir=WORKSPACE_DIR)
        pdf_save_dir = os.path.join(main_task_workspace, "pdfs_to_process")
        os.makedirs(pdf_save_dir, exist_ok=True)

        # Check for the top-level key Dify uses (e.g., 'input_files')
        if dify_input_key in data:
            # Dify passes the file info nested inside the variable object
            # The actual file list is usually under a key named "files"
            dify_variable_value = data[dify_input_key]
            if isinstance(dify_variable_value, dict) and 'files' in dify_variable_value and isinstance(dify_variable_value['files'], list):
                app.logger.info(f"Detected Dify input structure under key '{dify_input_key}'. Extracting from nested 'files' array.")
                dify_files_info = dify_variable_value['files'] # <--- Extract the nested list
                if not dify_files_info:
                     raise ValueError(f"Dify variable '{dify_input_key}' contained an empty nested 'files' list.")
                
                input_source = f'dify_{dify_input_key}'
                total_files_requested = len(dify_files_info)
                with task_lock:
                    current_task["status"] = "preparing_files"
                    current_task["total_files_requested"] = total_files_requested
                    current_task["files_prepared_count"] = 0

                for idx, file_info in enumerate(dify_files_info):
                    original_fname = file_info.get('filename', f'unknown_dify_{idx}.pdf')
                    with task_lock:
                         current_task["current_file_preparing"] = original_fname
                    try:
                        # Pass the file_info object directly
                        saved_path, original_fname_used = download_dify_file(file_info, pdf_save_dir)
                        files_to_process_info.append({'path': saved_path, 'original_filename': original_fname_used})
                        with task_lock:
                            current_task["files_prepared_count"] = len(files_to_process_info)
                    except (ValueError, ConnectionError, IOError) as e:
                         app.logger.warning(f"Skipping Dify file {original_fname} due to prepare error: {e}")
                         continue
            else:
                 # Fallback or error if the structure is not as expected
                 app.logger.warning(f"Received key '{dify_input_key}' but its value is not a dict with a 'files' list: {dify_variable_value}")
                 # Try to see if maybe the top level value IS the list itself (less likely now)
                 if isinstance(dify_variable_value, list) and len(dify_variable_value) >= 1:
                     app.logger.info(f"Attempting to process value of '{dify_input_key}' as the file list directly.")
                     dify_files_info = dify_variable_value
                     # ... (duplicate the file processing loop here or refactor) ...
                     # For simplicity, let's just raise error if nested 'files' not found for now
                     raise ValueError(f"键 '{dify_input_key}' 的值不是预期的包含 'files' 列表的对象结构。")
                 else:
                     raise ValueError(f"键 '{dify_input_key}' 存在但其值不是包含 'files' 列表的对象结构或文件列表。")

        elif 'pdf_list' in data and isinstance(data['pdf_list'], list) and len(data['pdf_list']) >= 1:
            # Priority 2: Base64 list input
            app.logger.info("Detected input using base64 list.")
            pdf_list_info = data['pdf_list']
            input_source = 'base64_list'
            total_files_requested = len(pdf_list_info)
            with task_lock:
                current_task["status"] = "preparing_files"
                current_task["total_files_requested"] = total_files_requested
                current_task["files_prepared_count"] = 0

            for idx, item in enumerate(pdf_list_info):
                 original_fname = item.get('filename', f'unknown_base64_{idx}.pdf')
                 if 'pdf_base64' not in item:
                     app.logger.warning(f"Skipping item in pdf_list {original_fname} due to missing 'pdf_base64'")
                     continue
                 with task_lock:
                     current_task["current_file_preparing"] = original_fname
                 try:
                    saved_path, original_fname_used = save_base64_file(item['pdf_base64'], original_fname, pdf_save_dir)
                    files_to_process_info.append({'path': saved_path, 'original_filename': original_fname_used})
                    with task_lock:
                         current_task["files_prepared_count"] = len(files_to_process_info)
                 except (ValueError, IOError) as e:
                     app.logger.warning(f"Skipping base64 file {original_fname} due to prepare error: {e}")
                     continue
        else:
            raise ValueError(f"无效的输入格式，需要提供 '{dify_input_key}' (其内部包含'files'列表) 或 'pdf_list'. 收到的keys: {list(data.keys())}")

        if not files_to_process_info:
             raise ValueError("没有成功准备任何文件进行处理")
        # ---------------------------------------------

        # --- Stage 2: Process prepared files sequentially --- 
        results_by_original_file = {}
        total_to_process = len(files_to_process_info)
        processed_successfully_count = 0

        with task_lock:
             current_task["input_source"] = input_source
             current_task["total_files_to_process"] = total_to_process
             del current_task["files_prepared_count"] # Remove preparation count
             del current_task["current_file_preparing"]

        for idx, file_info in enumerate(files_to_process_info):
            current_original_filename = file_info['original_filename']
            current_pdf_path = file_info['path']
            file_status = "failed" # Default status for this file
            file_content = None
            error_detail = None

            with task_lock:
                current_task["status"] = f"processing_file_{idx+1}_of_{total_to_process}"
                current_task["current_file_processing"] = current_original_filename

            # Create a sub-workspace for this specific olmocr run
            single_run_workspace = tempfile.mkdtemp(dir=main_task_workspace)
            single_run_output_path = os.path.join(single_run_workspace, "results")
            os.makedirs(single_run_output_path, exist_ok=True)
            # Copy the single PDF into the sub-workspace (olmocr needs it relative)
            single_run_pdf_path = os.path.join(single_run_workspace, os.path.basename(current_pdf_path))
            shutil.copy(current_pdf_path, single_run_pdf_path)

            try:
                # Execute olmocr command for the single file in its sub-workspace
                cmd = [
                    "python", "-m", "olmocr.pipeline",
                    single_run_workspace, # Use the sub-workspace
                    "--pdfs", single_run_pdf_path, # Point to the single PDF within
                    "--model", MODEL_PATH
                ]
                app.logger.info(f"Running olmocr for {current_original_filename}: {' '.join(cmd)}")
                process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300) # Add timeout per file
                app.logger.info(f"olmocr process for {current_original_filename} finished with code {process.returncode}")
                if process.stderr:
                     app.logger.warning(f"olmocr stderr for {current_original_filename}:\n{process.stderr}")

                if process.returncode != 0:
                    raise RuntimeError(f"olmocr处理失败 (Code: {process.returncode}): {process.stderr}")

                # Find the result file in the sub-workspace
                result_files = glob.glob(os.path.join(single_run_output_path, "output_*.jsonl"))
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
                processed_successfully_count += 1

            except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as e:
                error_detail = f"处理文件 '{current_original_filename}' 时出错: {e}"
                app.logger.error(error_detail)
            except Exception as e:
                error_detail = f"处理文件 '{current_original_filename}' 时发生意外错误: {e}"
                app.logger.error(error_detail, exc_info=True)
            finally:
                # Store result (or error) for this file
                results_by_original_file[current_original_filename] = {
                    "content": file_content,
                    "filename": current_original_filename,
                    "status": file_status,
                    "error": error_detail # Will be None on success
                }
                # Clean up the sub-workspace for this run
                shutil.rmtree(single_run_workspace, ignore_errors=True)
        # ---------------------------------------------

        # --- Stage 3: Prepare final response --- 
        final_status = "failure"
        if processed_successfully_count == total_to_process:
            final_status = "success"
        elif processed_successfully_count > 0:
             final_status = "partial_success"

        analysis_result = {
            "results": results_by_original_file,
            "total_files_requested": total_files_requested,
            "total_files_prepared": len(files_to_process_info),
            "processed_successfully": processed_successfully_count,
            "status": final_status,
            "processing_time": time.time() - current_task["start_time"]
        }
        response_data = {"analysis_output": analysis_result}

        # Clean up the main task workspace (incl. downloaded pdfs)
        shutil.rmtree(main_task_workspace, ignore_errors=True)
        main_task_workspace = None

        # Reset task status
        with task_lock:
            current_task = None

        return jsonify(response_data)

    except (ValueError, ConnectionError, IOError) as e: # Errors during initial file prep
        err_msg = f"文件准备错误: {e}"
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