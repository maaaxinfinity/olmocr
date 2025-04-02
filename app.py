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
        app.logger.info(f"Successfully downloaded {filename}")
        return target_path, filename
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download file from Dify {download_url}: {e}")
    except Exception as e:
        raise IOError(f"Failed to save downloaded file {filename}: {e}")

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
        return target_path, filename
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 encoding for file {filename_req}: {e}")
    except Exception as e:
        raise IOError(f"Failed to save base64 file {filename_req}: {e}")

# Define tool schema
# TODO: Update Schema - For now, just modifying the functions
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_pdf",
            "description": "使用olmocr分析单个PDF文件并提取内容. 可通过Dify文件对象或base64提供.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": { # Dify input
                        "type": "array",
                        "description": "(可选) Dify文件对象列表 (若提供, 必须且仅包含一个文件).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "filename": {"type": "string"}
                            },
                            "required": ["url", "filename"],
                            "maxItems": 1
                        }
                    },
                     "pdf_base64": { # Direct input
                        "type": "string",
                        "description": "(可选) PDF文件的base64编码内容 (若提供了'files'则忽略此项)."
                    },
                    "filename": { # Direct input filename
                        "type": "string",
                        "description": "(可选) 当提供'pdf_base64'时对应的PDF文件名 (若提供了'files'则忽略此项)."
                    }
                },
                # Either files OR (pdf_base64 + filename) should be provided
                # OpenAPI schema needs refinement (oneOf)
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_multiple_pdfs",
            "description": "使用olmocr分析多个PDF文件并提取内容. 可通过Dify文件对象或base64列表提供.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": { # Dify input
                        "type": "array",
                        "description": "(可选) Dify文件对象列表.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "filename": {"type": "string"}
                            },
                            "required": ["url", "filename"]
                        }
                    },
                    "pdf_list": { # Direct input
                        "type": "array",
                        "description": "(可选) PDF文件列表, 每个包含filename和pdf_base64 (若提供了'files'则忽略此项).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "pdf_base64": {"type": "string"}
                            },
                             "required": ["filename", "pdf_base64"]
                        }
                    }
                },
                 # Either files OR pdf_list should be provided
                 # OpenAPI schema needs refinement (oneOf)
            }
        }
    }
]

@app.route("/tools", methods=["GET"])
def get_tools():
    """返回工具的schema定义"""
    # Note: This schema definition above is simplified and needs proper 'oneOf' for inputs
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

@app.route("/analyze_pdf", methods=["POST"])
def analyze_pdf():
    """处理单个PDF分析请求 (Dify文件格式或Base64)"""
    global current_task

    with task_lock:
        if current_task:
            task_info_safe = {k: v for k, v in current_task.items() if k != 'command'}
            return jsonify({
                "error": "服务器忙，当前有任务正在处理中",
                "status": "busy",
                "task_info": task_info_safe
            }), 429

        current_task = {
            "type": "single_pdf",
            "start_time": time.time(),
            "status": "initializing"
        }

    task_workspace = None # Define task_workspace outside try block for cleanup
    try:
        data = request.json
        if not data:
            raise ValueError("请求体不能为空")

        # --- Determine input type and prepare file ---
        pdf_path = None
        filename = None
        input_source = None
        # >>> Use 'input_files' as the key from Dify input <<<
        dify_input_key = 'input_files' 

        # Create a temporary directory for this single file task
        task_workspace = tempfile.mkdtemp(dir=WORKSPACE_DIR)

        # Check if the key Dify uses exists and is valid
        if dify_input_key in data and isinstance(data[dify_input_key], list) and len(data[dify_input_key]) == 1:
            # Priority 1: Dify file input using the specified key
            app.logger.info(f"Detected input using Dify key: {dify_input_key}")
            dify_file_info = data[dify_input_key][0]
            pdf_path, filename = download_dify_file(dify_file_info, task_workspace)
            input_source = f'dify_{dify_input_key}'
            with task_lock:
                 current_task["status"] = "pdf_downloaded"
        elif 'pdf_base64' in data and 'filename' in data:
            # Priority 2: Base64 input (fallback/direct API call)
            app.logger.info("Detected input using base64.")
            pdf_path, filename = save_base64_file(data['pdf_base64'], data['filename'], task_workspace)
            input_source = 'base64'
            with task_lock:
                 current_task["status"] = "pdf_saved"
        else:
            # Include the expected key in the error message
            raise ValueError(f"无效的输入格式，需要提供 '{dify_input_key}' (含一个元素) 或 'pdf_base64' 与 'filename'. 收到的keys: {list(data.keys())}")
        # ---------------------------------------------

        with task_lock:
            current_task["filename"] = filename
            current_task["input_source"] = input_source

        # Prepare paths for olmocr
        output_path = os.path.join(task_workspace, "results")
        os.makedirs(output_path, exist_ok=True)

        # Execute olmocr command
        cmd = [
            "python", "-m", "olmocr.pipeline",
            task_workspace,
            "--pdfs", pdf_path,
            "--model", MODEL_PATH
        ]

        with task_lock:
            current_task["status"] = "processing_pdf"

        app.logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True, check=False) # Don't check=True, handle return code below
        app.logger.info(f"olmocr process finished with code {process.returncode}")
        if process.stdout:
            app.logger.debug(f"olmocr stdout:\n{process.stdout}")
        if process.stderr:
             app.logger.warning(f"olmocr stderr:\n{process.stderr}")

        if process.returncode != 0:
            raise RuntimeError(f"PDF处理失败 (olmocr 返回码 {process.returncode}): {process.stderr}")

        # Find the result file
        result_files = glob.glob(os.path.join(output_path, "output_*.jsonl"))

        if not result_files:
            # Sometimes olmocr might fail silently on a specific PDF without error code but no output
            raise FileNotFoundError("未找到处理结果文件 (olmocr可能处理失败)")

        # Read result content
        result_path = result_files[0]
        with open(result_path, 'r', encoding='utf-8') as f:
            result_content = f.read().strip()

        # Parse results
        results = []
        for line in result_content.split('\n'):
            if line.strip():
                results.append(json.loads(line))

        # Extract text
        extracted_text = ""
        if results:
            for result in results:
                if 'text' in result:
                    extracted_text += result['text'] + "\n\n"

        # Prepare response
        response_data = {
            "content": extracted_text.strip(),
            "filename": filename,
            "status": "success",
            "processing_time": time.time() - current_task["start_time"]
        }
        
        # Clean up temporary task workspace
        shutil.rmtree(task_workspace, ignore_errors=True)
        
        # Reset task status
        with task_lock:
            current_task = None

        return jsonify(response_data)

    except (ValueError, ConnectionError, IOError, FileNotFoundError, RuntimeError) as e:
        err_msg = f"文件处理错误: {e}"
        app.logger.error(err_msg)
        with task_lock:
            current_task = None
        # Check if task_workspace was assigned before trying to clean up
        if task_workspace and os.path.exists(task_workspace):
             shutil.rmtree(task_workspace, ignore_errors=True)
        return jsonify({"error": err_msg}), 400
    except Exception as e:
        err_msg = f"服务器内部错误: {e}"
        app.logger.error(err_msg, exc_info=True)
        with task_lock:
            current_task = None
        # Check if task_workspace was assigned before trying to clean up
        if task_workspace and os.path.exists(task_workspace):
             shutil.rmtree(task_workspace, ignore_errors=True)
        return jsonify({"error": err_msg}), 500

@app.route("/analyze_multiple_pdfs", methods=["POST"])
def analyze_multiple_pdfs():
    """处理多个PDF分析请求 (Dify文件格式)"""
    global current_task

    # Check for ongoing task
    with task_lock:
        if current_task:
            task_info_safe = {k: v for k, v in current_task.items() if k != 'command'}
            return jsonify({
                "error": "服务器忙，当前有任务正在处理中",
                "status": "busy",
                "task_info": task_info_safe
            }), 429

        # Set task info
        current_task = {
            "type": "multiple_pdfs",
            "start_time": time.time(),
            "status": "initializing"
        }

    # Create a temporary directory for this multi-file task
    task_workspace = tempfile.mkdtemp(dir=WORKSPACE_DIR)
    pdf_dir = os.path.join(task_workspace, "pdfs") # Store PDFs inside task workspace
    os.makedirs(pdf_dir, exist_ok=True)

    try:
        data = request.json

        # --- Determine input type and prepare files ---
        input_source = None
        files_to_process_info = [] # Store dicts {'path': path, 'original_filename': fname}
        total_files_requested = 0
        # >>> Use 'input_files' as the key from Dify input <<<
        dify_input_key = 'input_files'

        # Check if the key Dify uses exists and is valid
        if dify_input_key in data and isinstance(data[dify_input_key], list) and len(data[dify_input_key]) > 0:
            # Priority 1: Dify file input using the specified key
            app.logger.info(f"Detected input using Dify key: {dify_input_key}")
            dify_files_info = data[dify_input_key]
            input_source = f'dify_{dify_input_key}'
            total_files_requested = len(dify_files_info)
            with task_lock:
                current_task["total_files"] = total_files_requested
                current_task["processed_files_download"] = 0
                current_task["status"] = "downloading_pdfs"

            for idx, file_info in enumerate(dify_files_info):
                try:
                    pdf_path, filename = download_dify_file(file_info, pdf_dir)
                    # Use the original filename provided by Dify for mapping results later
                    files_to_process_info.append({'path': pdf_path, 'original_filename': file_info['filename']})
                    with task_lock:
                        current_task["current_file"] = file_info['filename']
                        current_task["processed_files_download"] = idx + 1
                except (ValueError, ConnectionError, IOError) as e:
                     app.logger.warning(f"Skipping Dify file {file_info.get('filename', 'N/A')} from key '{dify_input_key}' due to download/save error: {e}")
                     continue

        elif 'pdf_list' in data and isinstance(data['pdf_list'], list) and len(data['pdf_list']) > 0:
            # Priority 2: Base64 list input (fallback/direct API call)
            app.logger.info("Detected input using base64 list.")
            pdf_list_info = data['pdf_list']
            input_source = 'base64_list'
            total_files_requested = len(pdf_list_info)
            with task_lock:
                current_task["total_files"] = total_files_requested
                current_task["processed_files_save"] = 0
                current_task["status"] = "saving_pdfs"

            for idx, item in enumerate(pdf_list_info):
                 if 'pdf_base64' not in item or 'filename' not in item:
                     app.logger.warning(f"Skipping item in pdf_list at index {idx} due to missing 'pdf_base64' or 'filename'")
                     continue
                 try:
                    # Use the filename provided in the list for mapping results later
                    pdf_path, saved_filename = save_base64_file(item['pdf_base64'], item['filename'], pdf_dir)
                    files_to_process_info.append({'path': pdf_path, 'original_filename': item['filename']})
                    with task_lock:
                        current_task["current_file"] = item['filename']
                        current_task["processed_files_save"] = idx + 1
                 except (ValueError, IOError) as e:
                     app.logger.warning(f"Skipping base64 file {item.get('filename', 'N/A')} due to save error: {e}")
                     continue
        else:
             # Include the expected key in the error message
            raise ValueError(f"无效的输入格式，需要提供 '{dify_input_key}' (Dify文件对象列表) 或 'pdf_list' (含filename和pdf_base64的对象列表). 收到的keys: {list(data.keys())}")

        if not files_to_process_info:
             raise ValueError("没有成功准备任何文件进行处理")
        # ---------------------------------------------

        with task_lock:
             current_task["input_source"] = input_source
             current_task["files_to_process_count"] = len(files_to_process_info)

        # Prepare paths for olmocr
        output_path = os.path.join(task_workspace, "results")
        os.makedirs(output_path, exist_ok=True)

        # Execute olmocr command for all prepared PDFs in the directory
        cmd = [
            "python", "-m", "olmocr.pipeline",
            task_workspace,
            "--pdfs", f"{pdf_dir}/*.pdf", # Process all PDFs saved in the task's pdf dir
            "--model", MODEL_PATH
        ]

        with task_lock:
            current_task["status"] = "processing_pdfs"

        app.logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        app.logger.info(f"olmocr process finished with code {process.returncode}")
        if process.stdout:
            app.logger.debug(f"olmocr stdout:\n{process.stdout}")
        if process.stderr:
             app.logger.warning(f"olmocr stderr:\n{process.stderr}")

        if process.returncode != 0:
             # Even if it fails, try to collect partial results if any
             app.logger.error(f"olmocr 批量处理失败 (返回码 {process.returncode}): {process.stderr}")
             # Don't raise immediately, process any results found below

        # Read results from the output directory
        result_files = glob.glob(os.path.join(output_path, "output_*.jsonl"))

        # Process result files, mapping back to original filenames
        results_by_original_file = {}
        processed_count = 0
        for result_path in result_files:
            try:
                 # Extract olmocr output base name (without output_ prefix and .jsonl)
                 # This base name should match the secure_filename version saved earlier
                 output_base_name = os.path.basename(result_path).replace("output_", "").replace(".jsonl", "")

                 # Find the corresponding original filename from our prepared list
                 original_filename = None
                 for info in files_to_process_info:
                     # Compare secure_filename(original) with output_base_name
                     # Need to handle potential extension differences if secure_filename stripped it
                     secure_original_base = os.path.splitext(secure_filename(info['original_filename']))[0]
                     output_file_base = os.path.splitext(output_base_name)[0]
                     # A more robust check might involve comparing without extension or handling case
                     if secure_original_base.lower() == output_file_base.lower():
                         original_filename = info['original_filename']
                         break
                 
                 if not original_filename:
                      # Fallback: try matching based on the list order if lengths match?
                      # This is less reliable. Best effort is comparing secure names.
                      app.logger.warning(f"无法将结果文件 {os.path.basename(result_path)} (base: {output_base_name}) 映射回原始文件名，跳过。检查 secure_filename() 是否改变了文件名。")
                      continue # Cannot map back, skip

                 with open(result_path, 'r', encoding='utf-8') as f:
                     result_content = f.read().strip()

                 file_results = []
                 for line in result_content.split('\n'):
                     if line.strip():
                         file_results.append(json.loads(line))

                 extracted_text = ""
                 if file_results:
                     for result in file_results:
                         if 'text' in result:
                             extracted_text += result['text'] + "\n\n"

                 results_by_original_file[original_filename] = {
                     "content": extracted_text.strip(),
                     "filename": original_filename,
                     "status": "success"
                 }
                 processed_count += 1
            except Exception as e:
                 app.logger.warning(f"处理结果文件 {result_path} 时出错: {e}")
                 continue # Skip this specific result file

        # Prepare response - Nest results under 'analysis_output'
        analysis_result = {
            "results": results_by_original_file,
            "total_files_requested": total_files_requested,
            "total_files_prepared": len(files_to_process_info),
            "processed_files": processed_count,
            "status": "success" if process.returncode == 0 and processed_count == len(files_to_process_info) else "partial_success" if processed_count > 0 else "failure",
            "processing_time": time.time() - current_task["start_time"]
        }
        response_data = {"analysis_output": analysis_result}

        # Clean up temporary task workspace
        shutil.rmtree(task_workspace, ignore_errors=True)
        
        # Reset task status
        with task_lock:
            current_task = None

        return jsonify(response_data)

    except (ValueError, ConnectionError, IOError, FileNotFoundError, RuntimeError) as e:
        err_msg = f"批量文件处理错误: {e}"
        app.logger.error(err_msg)
        with task_lock:
            current_task = None
        # Check if task_workspace was assigned before trying to clean up
        if task_workspace and os.path.exists(task_workspace):
             shutil.rmtree(task_workspace, ignore_errors=True)
        return jsonify({"error": err_msg}), 400
    except Exception as e:
        err_msg = f"服务器内部错误: {e}"
        app.logger.error(err_msg, exc_info=True)
        with task_lock:
            current_task = None
        # Check if task_workspace was assigned before trying to clean up
        if task_workspace and os.path.exists(task_workspace):
             shutil.rmtree(task_workspace, ignore_errors=True)
        return jsonify({"error": err_msg}), 500

@app.route("/results/<path:filename>", methods=["GET"])
def get_result_file(filename):
    """获取结果文件 (Note: This might be less useful now as results are in memory)"""
    # This endpoint might need rethinking as results are now aggregated in memory
    # and temporary workspaces are deleted. Returning the final JSON might be better.
    # For now, keep it, but it likely won't find files after task completion.
    result_path = os.path.join(WORKSPACE_DIR, "results", secure_filename(filename))
    if os.path.exists(result_path):
        return send_file(result_path)
    else:
        return jsonify({"error": "文件不存在或已被清理"}), 404

if __name__ == "__main__":
    # Add basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    app.run(host="0.0.0.0", port=5555, debug=True) 