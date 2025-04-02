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

        # Create a temporary directory for this single file task
        task_workspace = tempfile.mkdtemp(dir=WORKSPACE_DIR)

        if 'files' in data and isinstance(data['files'], list) and len(data['files']) == 1:
            # Priority 1: Dify file input
            dify_file_info = data['files'][0]
            pdf_path, filename = download_dify_file(dify_file_info, task_workspace)
            input_source = 'dify_file'
            with task_lock:
                 current_task["status"] = "pdf_downloaded"
        elif 'pdf_base64' in data and 'filename' in data:
            # Priority 2: Base64 input
            pdf_path, filename = save_base64_file(data['pdf_base64'], data['filename'], task_workspace)
            input_source = 'base64'
            with task_lock:
                 current_task["status"] = "pdf_saved"
        else:
            raise ValueError("无效的输入格式，需要提供 'files' (含一个元素) 或 'pdf_base64' 与 'filename'.")
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

    except (ValueError, ConnectionError, IOError) as e:
        with task_lock:
            current_task = None
        if 'task_workspace' in locals() and os.path.exists(task_workspace):
             shutil.rmtree(task_workspace, ignore_errors=True)
        return jsonify({"error": f"文件处理错误: {e}"}), 400
    except Exception as e:
        with task_lock:
            current_task = None
        if 'task_workspace' in locals() and os.path.exists(task_workspace):
             shutil.rmtree(task_workspace, ignore_errors=True)
        app.logger.error(f"Unexpected error in analyze_pdf: {e}", exc_info=True)
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

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

        if not data or 'files' not in data or not isinstance(data['files'], list) or len(data['files']) == 0:
            with task_lock:
                current_task = None
            shutil.rmtree(task_workspace, ignore_errors=True)
            return jsonify({"error": "无效的输入格式，需要包含一个非空的'files'文件对象数组"}), 400

        dify_files_info = data['files']

        # Download all PDFs
        downloaded_pdf_paths = []
        with task_lock:
            current_task["total_files"] = len(dify_files_info)
            current_task["processed_files"] = 0
            current_task["status"] = "downloading_pdfs"

        for idx, file_info in enumerate(dify_files_info):
            try:
                pdf_path, filename = download_dify_file(file_info, pdf_dir)
                downloaded_pdf_paths.append(pdf_path)
                # Update task status
                with task_lock:
                    current_task["current_file"] = filename
                    current_task["processed_files"] = idx + 1
            except (ValueError, ConnectionError, IOError) as e:
                 # Log the error but try to continue with other files if possible
                 app.logger.warning(f"Skipping file due to download/save error: {e}")
                 continue # Skip this file
        
        if not downloaded_pdf_paths:
             with task_lock:
                 current_task = None
             shutil.rmtree(task_workspace, ignore_errors=True)
             return jsonify({"error": "所有文件下载失败"}), 400

        # Prepare paths for olmocr
        output_path = os.path.join(task_workspace, "results")
        os.makedirs(output_path, exist_ok=True)

        # Execute olmocr command for all PDFs in the directory
        cmd = [
            "python", "-m", "olmocr.pipeline",
            task_workspace, # Use task-specific workspace
            "--pdfs", f"{pdf_dir}/*.pdf", # Process all PDFs in the task's pdf dir
            "--model", MODEL_PATH
        ]

        # Update task status
        with task_lock:
            # current_task["command"] = " ".join(cmd)
            current_task["status"] = "processing_pdfs"

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            with task_lock:
                current_task = None
            shutil.rmtree(task_workspace, ignore_errors=True)
            return jsonify({
                "error": "批量PDF处理失败",
                "details": process.stderr
            }), 500

        # Read results from the output directory
        result_files = glob.glob(os.path.join(output_path, "output_*.jsonl"))

        if not result_files:
            with task_lock:
                current_task = None
            shutil.rmtree(task_workspace, ignore_errors=True)
            return jsonify({"error": "未找到处理结果"}), 500

        # Process all result files
        results_by_file = {}
        for result_path in result_files:
            try:
                 # Extract original filename based on how olmocr names output
                 # Assuming output is like 'output_origfilename_pdf.jsonl'
                 base_name = os.path.basename(result_path).replace("output_", "").replace(".jsonl", "")
                 # Find the matching original filename (case-insensitive for robustness)
                 original_filename = next((fi['filename'] for fi in dify_files_info if os.path.splitext(fi['filename'].lower())[0] == os.path.splitext(base_name.lower())[0] ), base_name + ".pdf") # Fallback
                 
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
                 
                 results_by_file[original_filename] = {
                     "content": extracted_text.strip(),
                     "filename": original_filename
                 }
            except Exception as e:
                 app.logger.warning(f"Error processing result file {result_path}: {e}")
                 continue # Skip this result file

        # Prepare response
        response_data = {
            "results": results_by_file,
            "total_files_requested": len(dify_files_info),
            "total_files_downloaded": len(downloaded_pdf_paths),
            "processed_files": len(results_by_file),
            "status": "success",
            "processing_time": time.time() - current_task["start_time"]
        }
        
        # Clean up temporary task workspace
        shutil.rmtree(task_workspace, ignore_errors=True)
        
        # Reset task status
        with task_lock:
            current_task = None

        return jsonify(response_data)

    except Exception as e:
        with task_lock:
            current_task = None
        if 'task_workspace' in locals() and os.path.exists(task_workspace):
             shutil.rmtree(task_workspace, ignore_errors=True)
        app.logger.error(f"Unexpected error in analyze_multiple_pdfs: {e}", exc_info=True)
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

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