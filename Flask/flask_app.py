from flask import Flask, request, jsonify, send_from_directory, abort
import os
import uuid
import threading
import time
import logging
from functools import wraps # Import wraps for decorator

# Import utility functions from api_utils
from api_utils import (
    run_olmocr_on_single_pdf,
    get_gpu_stats,
    clear_temp_workspace,
    clear_all_processed_data,
    list_preview_files,
    list_jsonl_files,
    export_html_archive,
    export_combined_archive,
    tasks, # Import the shared task dictionary
    PROCESSED_PREVIEW_DIR,
    PROCESSED_JSONL_DIR,
    EXPORT_TEMP_DIR_BASE,
    UPLOAD_TEMP_DIR,
    ensure_dirs
)

# Ensure directories exist at startup
ensure_dirs()

app = Flask(__name__)

# Configure logging to match api_utils
log_format = '%s(asctime)s - %s(name)s - %s(levelname)s - %s(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- API Key Authentication ---
# IMPORTANT: Set this environment variable in production!
API_KEY = os.environ.get("API_SECRET_KEY", "default_secret_key_change_me")

def require_api_key(f):
    """Decorator to require API key in request header."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') and request.headers.get('X-API-Key') == API_KEY:
            return f(*args, **kwargs)
        else:
            logger.warning("Unauthorized access attempt.")
            abort(401, description="Unauthorized: Invalid or missing API Key.")
    return decorated_function

# --- API Endpoints ---

@app.route('/process', methods=['POST'])
@require_api_key # Protect this endpoint
def start_processing():
    """Endpoint to upload a PDF and start OLMOCR processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "No selected file or invalid file type (must be .pdf)"}), 400

    # --- Parameter Handling with Modes ---
    mode = request.form.get('mode', 'normal')
    params = {}
    required_params_normal = ['target_dim', 'anchor_len', 'error_rate', 'max_context', 'max_retries']
    required_params_fast = ['target_dim', 'anchor_len', 'max_context']
    missing_params = []

    if mode == 'fast':
        params['error_rate'] = 0.05 # Fast mode override
        params['max_retries'] = 5    # Fast mode override
        # Check required params for fast mode
        for p in required_params_fast:
            value = request.form.get(p)
            if value is None:
                missing_params.append(p)
            else:
                try:
                    # Attempt conversion based on expected type (adjust if needed)
                    if p in ['target_dim', 'anchor_len', 'max_context']:
                        params[p] = int(value)
                    # Add float conversion if other params were required
                    # elif p in [...]:
                    #    params[p] = float(value)
                except ValueError:
                    return jsonify({"error": f"Invalid value type for parameter '{p}'. Expected numeric."}), 400
    else: # Normal mode
        # Check required params for normal mode
        for p in required_params_normal:
            value = request.form.get(p)
            if value is None:
                missing_params.append(p)
            else:
                try:
                    # Attempt conversion based on expected type
                    if p in ['target_dim', 'anchor_len', 'max_context', 'max_retries']:
                        params[p] = int(value)
                    elif p == 'error_rate':
                        params[p] = float(value)
                except ValueError:
                     return jsonify({"error": f"Invalid value type for parameter '{p}'. Expected numeric."}), 400

    if missing_params:
        return jsonify({"error": f"Missing required parameters for mode '{mode}': {', '.join(missing_params)}"}), 400

    # Add fixed worker count
    params['workers'] = 1
    # --- End Parameter Handling ---

    task_id = str(uuid.uuid4())
    # Sanitize filename for upload path, keep original for task info
    safe_filename = "".join(c if c.isalnum() or c in ('.', '-', '_') else '_' for c in file.filename)
    upload_filename = f"{task_id}_{safe_filename}"
    upload_path = os.path.join(UPLOAD_TEMP_DIR, upload_filename)

    try:
        file.save(upload_path)
        logger.info(f"File uploaded to: {upload_path}")

        # Initialize task state (including mode)
        tasks[task_id] = {
            "status": "queued",
            "mode": mode,
            "start_time": time.time(),
            "original_filename": file.filename,
            "logs": [f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Task created (Mode: {mode})"],
            "params": params,
            "result": {
                 "jsonl_path": None,
                 "html_path": None
            },
            "error": None,
            "olmocr_stdout": "",
            "olmocr_stderr": ""
        }

        # Start the OLMOCR process in a background thread
        thread = threading.Thread(
            target=run_olmocr_on_single_pdf,
            args=(upload_path, task_id, params) # Pass the validated params
        )
        thread.daemon = True
        thread.start()

        return jsonify({"message": "Processing started", "task_id": task_id}), 202

    except Exception as e:
        logger.error(f"Error starting processing task: {e}", exc_info=True) # Log traceback
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except OSError as rm_err:
                logger.error(f"Failed to remove uploaded file {upload_path} after error: {rm_err}")
        return jsonify({"error": f"Failed to start processing: {e}"}), 500

@app.route('/process/<task_id>', methods=['GET'])
@require_api_key # Protect this endpoint
def get_task_status(task_id):
    """Endpoint to check the status of a processing task."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    # Optional: Add elapsed time
    task_info = task.copy() # Avoid modifying the original dict directly
    task_info["elapsed_time_seconds"] = time.time() - task["start_time"]

    return jsonify(task_info), 200

@app.route('/files/previews', methods=['GET'])
@require_api_key
def get_preview_files():
    """Endpoint to list available HTML preview files."""
    try:
        files = list_preview_files()
        return jsonify({"preview_files": files}), 200
    except Exception as e:
        logger.error(f"Error listing preview files: {e}")
        return jsonify({"error": f"Could not list preview files: {e}"}), 500

@app.route('/files/previews/<filename>', methods=['GET'])
@require_api_key
def download_preview_file(filename):
    """Endpoint to download a specific HTML preview file."""
    # Basic security check: ensure filename doesn't contain path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        abort(400, description="Invalid filename.")
    try:
        # Use send_from_directory for safer file serving
        return send_from_directory(PROCESSED_PREVIEW_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404, description="File not found.")
    except Exception as e:
        logger.error(f"Error sending preview file {filename}: {e}")
        abort(500, description="Could not send file.")

@app.route('/files/jsonl', methods=['GET'])
@require_api_key
def get_jsonl_files():
    """Endpoint to list available JSONL files."""
    try:
        files = list_jsonl_files()
        return jsonify({"jsonl_files": files}), 200
    except Exception as e:
        logger.error(f"Error listing jsonl files: {e}")
        return jsonify({"error": f"Could not list jsonl files: {e}"}), 500

@app.route('/files/jsonl/<filename>', methods=['GET'])
@require_api_key
def download_jsonl_file(filename):
    """Endpoint to download a specific JSONL file."""
    if '..' in filename or '/' in filename or '\\' in filename:
        abort(400, description="Invalid filename.")
    try:
        return send_from_directory(PROCESSED_JSONL_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404, description="File not found.")
    except Exception as e:
        logger.error(f"Error sending jsonl file {filename}: {e}")
        abort(500, description="Could not send file.")

@app.route('/export/html', methods=['GET'])
@require_api_key
def download_html_export():
    """Endpoint to export all HTML files as a zip archive."""
    try:
        result = export_html_archive()
        if result["status"] == "success" and result["zip_path"]:
            zip_filename = os.path.basename(result["zip_path"])
            # Serve from the EXPORT_TEMP_DIR_BASE directory
            return send_from_directory(EXPORT_TEMP_DIR_BASE, zip_filename, as_attachment=True)
        else:
            return jsonify({"error": result["message"]}), 404 # Or 500 if error during creation
    except Exception as e:
        logger.exception("Error during HTML export route.")
        return jsonify({"error": f"Failed to export HTML: {e}"}), 500

@app.route('/export/<export_format>', methods=['GET'])
@require_api_key
def download_combined_export(export_format):
    """Endpoint to export MD or DOCX (with HTML) as a zip archive."""
    if export_format not in ['md', 'docx']:
        return jsonify({"error": "Invalid export format specified. Use 'md' or 'docx'"}), 400

    try:
        result = export_combined_archive(export_format)
        if result["status"] == "success" and result["zip_path"]:
            zip_filename = os.path.basename(result["zip_path"])
            return send_from_directory(EXPORT_TEMP_DIR_BASE, zip_filename, as_attachment=True)
        elif result["status"] == "error" and "没有 JSONL 文件" in result["message"]:
             return jsonify({"error": result["message"]}), 404
        else:
            # Handle script execution errors or zipping errors
            return jsonify({"error": f"Failed to create {export_format} export: {result['message']}"}), 500
    except Exception as e:
        logger.exception(f"Error during {export_format} export route.")
        return jsonify({"error": f"Failed to export {export_format}: {e}"}), 500

@app.route('/cache/temp', methods=['DELETE'])
@require_api_key
def clear_temp():
    """Endpoint to clear temporary OLMOCR run directories."""
    try:
        result = clear_temp_workspace()
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error clearing temporary workspace.")
        return jsonify({"error": f"Failed to clear temporary directories: {e}"}), 500

@app.route('/cache/processed', methods=['DELETE'])
@require_api_key
def clear_processed():
    """Endpoint to clear all processed PDF, JSONL, and HTML files."""
    try:
        result = clear_all_processed_data()
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error clearing processed data.")
        return jsonify({"error": f"Failed to clear processed data: {e}"}), 500

@app.route('/status/gpu', methods=['GET'])
@require_api_key
def get_gpu_status_api():
    """Endpoint to get GPU status."""
    try:
        stats = get_gpu_stats()
        if "error" in stats:
            # Return a different status code if monitoring is unavailable vs. an error occurred
            if "无法初始化 NVML" in stats["error"]:
                return jsonify(stats), 501 # 501 Not Implemented (or a custom code)
            else:
                return jsonify(stats), 500 # Internal Server Error if failed to query
        return jsonify(stats), 200
    except Exception as e:
        logger.exception("Error getting GPU status.")
        return jsonify({"error": f"Failed to get GPU status: {e}"}), 500

if __name__ == '__main__':
    # Ensure UPLOAD_TEMP_DIR exists before running
    os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
    print(f"INFO: Using API Key: {API_KEY[:4]}...{API_KEY[-4:] if len(API_KEY) > 8 else ''}") # Print partial key for verification
    # Run Flask app (use 0.0.0.0 to be accessible on network)
    # Set debug=True for development only, disable in production
    app.run(host='0.0.0.0', port=7860, debug=False) 