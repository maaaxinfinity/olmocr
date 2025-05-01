from flask import Flask, request, jsonify, send_from_directory, abort
import os
import uuid
import threading
import time
import logging
from functools import wraps # Import wraps for decorator
from flask_cors import CORS # Add CORS import
import queue # Add queue import

# Import utility functions from api_utils
from api_utils import (
    run_olmocr_on_single_pdf,
    get_system_status,
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
CORS(app, resources={r"/*": {"origins": ["*.vercel.app", "*.limitee.cn"]}})

# Configure logging to match api_utils
log_format = '%s(asctime)s - %s(name)s - %s(levelname)s - %s(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Task Queue Setup ---
task_queue = queue.Queue()
MAX_WORKERS = 1 # Limit to 1 concurrent OLMOCR process for now

def processing_worker():
    """Worker thread function to process tasks from the queue."""
    while True:
        try:
            logger.info(f"Worker waiting for task... Queue size: {task_queue.qsize()}")
            upload_path, task_id, params = task_queue.get() # Blocks until a task is available
            logger.info(f"Worker picked up task {task_id} for file: {upload_path}")
            # Run the actual processing function from api_utils
            run_olmocr_on_single_pdf(upload_path, task_id, params)
            logger.info(f"Worker finished task {task_id}")
            task_queue.task_done() # Signal that the task is complete
        except Exception as e:
            # Log any unexpected error in the worker itself
            # Task-specific errors should be handled within run_olmocr_on_single_pdf
            logger.error(f"Error in processing worker: {e}", exc_info=True)
            # If get() failed or task_done() failed, we might need more robust error handling,
            # but for now, just log and continue.

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
    """Endpoint to upload a PDF, add to queue, and start OLMOCR processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "No selected file or invalid file type (must be .pdf)"}), 400

    # --- Parameter Handling with Modes ---
    mode = request.form.get('mode', 'normal')
    params = {}
    if mode == 'fast':
        # Fast mode: Only requires the file. Use fixed error_rate and retries.
        # OLMOCR pipeline's internal defaults will be used for other params.
        params['error_rate'] = 0.03 # Updated requirement
        params['max_retries'] = 5
        # No missing params check needed for fast mode anymore
    else: # Normal mode
        # Check required params for normal mode
        required_params_normal = ['target_dim', 'anchor_len', 'error_rate', 'max_context', 'max_retries']
        missing_params = [] # Reset missing_params for normal mode
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
        logger.info(f"File uploaded to: {upload_path}, adding task {task_id} to queue.")

        # Initialize task state (status is 'queued')
        tasks[task_id] = {
            "status": "queued",
            "mode": mode,
            "start_time": time.time(), # Log when it was added to queue
            "original_filename": file.filename,
            "logs": [f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Task created and queued (Mode: {mode})"],
            "params": params,
            "result": {
                 "jsonl_path": None,
                 "html_path": None
            },
            "error": None,
            "olmocr_stdout": "",
            "olmocr_stderr": ""
        }

        # --- Queue the task instead of starting a thread directly ---
        task_queue.put((upload_path, task_id, params))
        logger.info(f"Task {task_id} added to queue. Queue size: {task_queue.qsize()}")

        return jsonify({"message": "Processing task added to queue", "task_id": task_id}), 202 # Accepted

    except Exception as e:
        # Handle errors during file save or initial task setup
        logger.error(f"Error adding task {task_id} to queue: {e}", exc_info=True)
        # Clean up uploaded file if save was successful but queueing failed (or other error)
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except OSError as rm_err:
                logger.error(f"Failed to remove uploaded file {upload_path} after error: {rm_err}")
        # Remove task entry if it was created
        if task_id in tasks:
            del tasks[task_id]
        return jsonify({"error": f"Failed to add task to queue: {e}"}), 500

@app.route('/process/<task_id>', methods=['GET'])
@require_api_key # Protect this endpoint
def get_task_status(task_id):
    """Endpoint to check the status of a processing task."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    task_info = task.copy() # Avoid modifying the original dict directly

    # --- Determine elapsed time based on status ---
    terminal_states = ["completed", "failed", "completed_with_warnings"]
    if task_info.get("status") in terminal_states and "final_elapsed_time" in task_info:
        # For terminal states, use the stored final time
        task_info["elapsed_time_seconds"] = task_info["final_elapsed_time"]
    elif "start_time" in task_info:
        # For ongoing states, calculate current elapsed time
        task_info["elapsed_time_seconds"] = time.time() - task_info["start_time"]
    else:
        # Fallback if start_time is somehow missing
        task_info["elapsed_time_seconds"] = 0
    # --- End of modification ---

    # Add queue position (approximation)
    try:
        # This is tricky as queue doesn't directly expose elements.
        # We can only show current size.
        task_info["current_queue_size"] = task_queue.qsize()
        # A more complex approach would be needed to find exact position.
    except Exception:
        pass # Ignore errors getting queue size

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

@app.route('/status/system', methods=['GET'])
@require_api_key
def get_system_status_api():
    """Endpoint to get system status (CPU, Memory, GPU)."""
    try:
        stats = get_system_status()
        return jsonify(stats), 200
    except Exception as e:
        logger.exception("Error getting system status.")
        return jsonify({"error": f"Failed to get system status: {e}"}), 500

if __name__ == '__main__':
    # Ensure UPLOAD_TEMP_DIR exists before running
    os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

    # --- Start Worker Threads ---
    logger.info(f"Starting {MAX_WORKERS} processing worker threads...")
    for i in range(MAX_WORKERS):
        worker_thread = threading.Thread(target=processing_worker, daemon=True)
        # worker_thread.daemon = True # Ensure thread exits when main app exits
        worker_thread.start()
        logger.info(f"Worker thread {i+1} started.")

    print(f"INFO: Using API Key: {API_KEY[:4]}...{API_KEY[-4:] if len(API_KEY) > 8 else ''}") # Print partial key for verification
    # Run Flask app (use 0.0.0.0 to be accessible on network)
    # Set debug=True for development only, disable in production
    app.run(host='0.0.0.0', port=7860, debug=False)

# Note: When running with a production server like Gunicorn with multiple workers,
# this simple in-memory queue and worker thread approach will NOT work correctly,
# as each worker process would have its own queue and threads.
# A shared queue (like Redis) and dedicated worker processes (like Celery) would be needed. 