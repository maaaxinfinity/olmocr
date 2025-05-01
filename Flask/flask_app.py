from flask import Flask, request, jsonify, send_from_directory, abort
import os
import uuid
import threading
import time
import logging
from functools import wraps # Import wraps for decorator
from flask_cors import CORS # Add CORS import
import queue # Add queue import
import sqlite3
import json # Needed for serializing logs/params
import platform # Needed for OS check in cancellation
import psutil # Needed for process termination

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
    PROCESSED_PREVIEW_DIR,
    PROCESSED_JSONL_DIR,
    EXPORT_TEMP_DIR_BASE,
    UPLOAD_TEMP_DIR,
    ensure_dirs,
    GRADIO_WORKSPACE_DIR
)

# Ensure directories exist at startup
ensure_dirs()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging to match api_utils
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Database Setup ---
DATABASE_PATH = os.path.join(GRADIO_WORKSPACE_DIR, 'tasks.db')

def get_db_conn():
    """Establishes a connection to the SQLite database."""
    # check_same_thread=False is needed because the connection 
    # might be used by the background worker thread.
    # For more complex apps, a connection pool or passing 
    # connection per request/task would be better.
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def init_db():
    """Initializes the database and creates the tasks table if it doesn't exist."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            mode TEXT,
            start_time REAL NOT NULL, 
            original_filename TEXT,
            logs TEXT,            -- Store as JSON string
            params TEXT,          -- Store as JSON string
            jsonl_path TEXT,
            html_path TEXT,
            error TEXT,
            olmocr_stdout TEXT,
            olmocr_stderr TEXT,
            final_elapsed_time REAL,
            process_pid INTEGER,  -- Added column for OLMOCR process PID
            processing_start_time REAL -- Added column for actual processing start time
        )
        ''')
        conn.commit()
        logger.info(f"Database initialized successfully at {DATABASE_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise # Reraise the exception to prevent app start if DB fails
    finally:
        if conn:
            conn.close()

def add_task_to_db(task_id, task_data):
    """Adds a new task record to the database."""
    conn = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO tasks (
            task_id, status, mode, start_time, original_filename, logs, params, 
            jsonl_path, html_path, error, olmocr_stdout, olmocr_stderr, final_elapsed_time,
            process_pid, -- Added column
            processing_start_time -- Added column
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_id, 
            task_data.get('status', 'unknown'), 
            task_data.get('mode'),
            task_data.get('start_time', time.time()),
            task_data.get('original_filename'),
            json.dumps(task_data.get('logs', [])), # Serialize logs
            json.dumps(task_data.get('params', {})), # Serialize params
            task_data.get('result', {}).get('jsonl_path'),
            task_data.get('result', {}).get('html_path'),
            task_data.get('error'),
            task_data.get('olmocr_stdout'),
            task_data.get('olmocr_stderr'),
            task_data.get('final_elapsed_time'),
            task_data.get('process_pid'), # Added value (will be None initially)
            task_data.get('processing_start_time') # Added value (will be None initially)
        ))
        conn.commit()
        logger.info(f"Task {task_id} added to database.")
    except sqlite3.Error as e:
        logger.error(f"Failed to add task {task_id} to DB: {e}", exc_info=True)
        # Optionally re-raise or handle specific errors (e.g., UNIQUE constraint)
    finally:
        if conn:
            conn.close()

def update_task_in_db(task_id, updates):
    """Updates specific fields of a task record in the database."""
    conn = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        
        set_clauses = []
        values = []
        for key, value in updates.items():
            # Ensure the key is a valid column name to prevent SQL injection risk if keys were dynamic
            # (Here keys are controlled internally, so less risk, but good practice)
            valid_columns = ["status", "mode", "start_time", "original_filename", "logs", 
                             "params", "jsonl_path", "html_path", "error", 
                             "olmocr_stdout", "olmocr_stderr", "final_elapsed_time",
                             "process_pid", "processing_start_time"] # Added process_pid & processing_start_time
            if key in valid_columns:
                set_clauses.append(f"{key} = ?")
                # Serialize if it's logs or params
                if key in ['logs', 'params'] and not isinstance(value, str):
                    values.append(json.dumps(value))
                else:
                     values.append(value)
            else:
                logger.warning(f"Attempted to update invalid column '{key}' for task {task_id}")

        if not set_clauses: # No valid updates provided
            logger.warning(f"No valid fields to update for task {task_id}")
            return

        sql = f"UPDATE tasks SET { ', '.join(set_clauses)} WHERE task_id = ?"
        values.append(task_id)
        
        cursor.execute(sql, tuple(values))
        conn.commit()
        # logger.debug(f"Updated task {task_id} in DB with fields: {list(updates.keys())}")
    except sqlite3.Error as e:
        logger.error(f"Failed to update task {task_id} in DB: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def get_task_from_db(task_id):
    """Retrieves a task record from the database as a dictionary."""
    conn = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            # Convert row object to a dictionary
            task_dict = dict(row)
            # Deserialize JSON fields
            try:
                if task_dict.get('logs'): task_dict['logs'] = json.loads(task_dict['logs'])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not decode logs JSON for task {task_id}")
                task_dict['logs'] = [] # Provide default
            try:
                if task_dict.get('params'): task_dict['params'] = json.loads(task_dict['params'])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not decode params JSON for task {task_id}")
                task_dict['params'] = {} # Provide default
                
            # Ensure process_pid is included (even if NULL)
            task_dict['process_pid'] = task_dict.get('process_pid') 
                
            # Ensure processing_start_time is included
            task_dict['processing_start_time'] = task_dict.get('processing_start_time')

            # Reconstruct the nested 'result' dict for compatibility if needed
            task_dict['result'] = {
                'jsonl_path': task_dict.get('jsonl_path'),
                'html_path': task_dict.get('html_path')
            }
            return task_dict
        else:
            return None
    except sqlite3.Error as e:
        logger.error(f"Failed to get task {task_id} from DB: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

def delete_task_from_db(task_id):
    """Deletes a task record from the database."""
    conn = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Deleted task {task_id} from database.")
            return True
        else:
             logger.warning(f"Attempted to delete task {task_id}, but it was not found in DB.")
             return False # Indicate task was not found
    except sqlite3.Error as e:
        logger.error(f"Failed to delete task {task_id} from DB: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def get_all_tasks_from_db():
    """Retrieves all task records from the database as a list of dictionaries."""
    conn = None
    tasks_list = []
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks ORDER BY start_time DESC") # Order by start time, newest first
        rows = cursor.fetchall()
        for row in rows:
            # Convert row object to a dictionary
            task_dict = dict(row)
            # Deserialize JSON fields
            try:
                if task_dict.get('logs'): task_dict['logs'] = json.loads(task_dict['logs'])
                else: task_dict['logs'] = [] # Default if NULL
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not decode logs JSON for task {task_dict.get('task_id')}")
                task_dict['logs'] = [] # Provide default
            try:
                if task_dict.get('params'): task_dict['params'] = json.loads(task_dict['params'])
                else: task_dict['params'] = {} # Default if NULL
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not decode params JSON for task {task_dict.get('task_id')}")
                task_dict['params'] = {} # Provide default
                
            # Ensure process_pid is included
            task_dict['process_pid'] = task_dict.get('process_pid') 
                
            # Ensure processing_start_time is included
            task_dict['processing_start_time'] = task_dict.get('processing_start_time')

            # Reconstruct the nested 'result' dict for compatibility
            task_dict['result'] = {
                'jsonl_path': task_dict.get('jsonl_path'),
                'html_path': task_dict.get('html_path')
            }
            tasks_list.append(task_dict)
        return tasks_list
    except sqlite3.Error as e:
        logger.error(f"Failed to get all tasks from DB: {e}", exc_info=True)
        return [] # Return empty list on error
    finally:
        if conn:
            conn.close()

# --- End Database Setup & Functions ---

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
            
            # --- Pass the DB update function as the callback ---
            run_olmocr_on_single_pdf(upload_path, task_id, params, update_callback=update_task_in_db)
            # --- End modification ---

            logger.info(f"Worker finished task {task_id}")
            task_queue.task_done() # Signal that the task is complete
        except Exception as e:
            logger.error(f"Error in processing worker: {e}", exc_info=True)
            # If a critical error happens here, the task state in DB might remain 'processing'
            # Consider adding logic here to update the task to 'failed' in DB
            try:
                 update_task_in_db(task_id, {"status": "failed", "error": f"Worker thread error: {e}"})
            except Exception as db_update_err:
                 logger.error(f"[Task {task_id}] Failed to update task status to FAILED in DB after worker error: {db_update_err}")

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
@require_api_key
def start_processing():
    """Endpoint to upload a PDF, add to DB & queue."""
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
        # 1. Prepare task data for DB insertion FIRST
        task_data = {
            "status": "queued",
            "mode": mode, # Get mode again here
            "start_time": time.time(),
            "original_filename": file.filename,
            "logs": [f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Task created and queued"],
            "params": params,
            # Result fields are initially NULL in DB
            "result": {}, # Keep empty for add_task_to_db compatibility
            "error": None,
            "olmocr_stdout": "",
            "olmocr_stderr": "",
            "final_elapsed_time": None,
            "process_pid": None,
            "processing_start_time": None # Added field
        }

        # 2. Add task to Database
        add_task_to_db(task_id, task_data)

        # 3. Save the file
        file.save(upload_path)
        logger.info(f"File uploaded to: {upload_path}")

        # 4. Put the task details (needed by worker) into the queue
        task_queue.put((upload_path, task_id, params))
        logger.info(f"Task {task_id} added to queue. Queue size: {task_queue.qsize()}")

        return jsonify({"message": "Processing task added to queue", "task_id": task_id}), 202 # Accepted

    except Exception as e:
        logger.error(f"Error starting processing task {task_id}: {e}", exc_info=True)
        # Attempt cleanup
        if os.path.exists(upload_path):
            try: os.remove(upload_path) 
            except OSError as rm_err: logger.error(f"Failed to remove uploaded file {upload_path} after error: {rm_err}")
        # Also attempt to delete from DB if it was added
        delete_task_from_db(task_id)
        return jsonify({"error": f"Failed to start processing: {e}"}), 500

@app.route('/process/<task_id>', methods=['GET'])
@require_api_key
def get_task_status(task_id):
    """Endpoint to check the status of a processing task from DB."""
    # --- Fetch task from DB --- 
    task_dict = get_task_from_db(task_id)
    # --- End fetch --- 

    if not task_dict:
        return jsonify({"error": "Task not found"}), 404

    task_info = task_dict # Use the dictionary fetched from DB

    # --- Determine elapsed time based on status and stored values ---
    terminal_states = ["completed", "failed", "completed_with_warnings", "cancelled"]
    current_status = task_info.get("status")
    final_elapsed = task_info.get("final_elapsed_time")
    processing_start_val = task_info.get("processing_start_time")

    if current_status == 'queued':
        task_info["elapsed_time_seconds"] = 0
    elif current_status == 'processing' and processing_start_val is not None:
        task_info["elapsed_time_seconds"] = time.time() - processing_start_val
    elif current_status in terminal_states and final_elapsed is not None:
        task_info["elapsed_time_seconds"] = final_elapsed
    else:
        task_info["elapsed_time_seconds"] = 0 # Fallback
        if current_status == 'processing' and processing_start_val is None:
            logger.warning(f"[Task {task_id}] Status is 'processing' but processing_start_time is missing. Elapsed time set to 0.")
    # --- End of modification ---

    # Add queue position (approximation) - this remains based on the live queue
    try:
        task_info["current_queue_size"] = task_queue.qsize()
    except Exception:
        pass 

    # Ensure process_pid is included in the response
    task_info['process_pid'] = task_dict.get('process_pid') # Already fetched

    # Ensure processing_start_time is included
    task_info['processing_start_time'] = task_dict.get('processing_start_time')

    # Remove raw DB paths from result dict before sending? No, keep for now.
    # del task_info['jsonl_path'] 
    # del task_info['html_path']

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

@app.route('/tasks', methods=['GET'])
@require_api_key
def get_all_tasks():
    """Endpoint to retrieve all task records from the database."""
    logger.info("Received request to get all tasks.")
    try:
        all_tasks = get_all_tasks_from_db()
        # We might need to calculate current elapsed time for ongoing tasks here
        # similar to get_task_status, or let the frontend handle it.
        # For simplicity, let's return the raw data for now.
        # Frontend can calculate elapsed time for non-terminal states if needed.
        for task_info in all_tasks:
            terminal_states = ["completed", "failed", "completed_with_warnings", "cancelled"]
            current_status = task_info.get("status")
            final_elapsed = task_info.get("final_elapsed_time")
            processing_start_val = task_info.get("processing_start_time")

            if current_status == 'queued':
                task_info["elapsed_time_seconds"] = 0
            elif current_status == 'processing' and processing_start_val is not None:
                task_info["elapsed_time_seconds"] = time.time() - processing_start_val
            elif current_status in terminal_states and final_elapsed is not None:
                task_info["elapsed_time_seconds"] = final_elapsed
            else:
                task_info["elapsed_time_seconds"] = 0 # Fallback
                if current_status == 'processing' and processing_start_val is None:
                    logger.warning(f"[Task {task_info.get('task_id')}] Status is 'processing' but processing_start_time is missing. Elapsed time set to 0.")
                
            # Add current queue size info (optional, maybe less useful here)
            try: task_info["current_queue_size"] = task_queue.qsize() 
            except: pass
            
            # Include process_pid
            task_info['process_pid'] = task_info.get('process_pid')
            
            # Ensure processing_start_time is included
            task_info['processing_start_time'] = task_info.get('processing_start_time')
            
        return jsonify(all_tasks), 200
    except Exception as e:
        logger.error(f"Error retrieving all tasks: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve tasks"}), 500

@app.route('/tasks/<task_id>', methods=['DELETE'])
@require_api_key
def delete_task(task_id):
    """Endpoint to delete a task record from DB and its associated files."""
    logger.info(f"Received request to delete task: {task_id}")

    # 1. Find the task in DB to get file paths and PID
    task = get_task_from_db(task_id)
    if not task:
        logger.warning(f"Delete request failed: Task {task_id} not found in DB.")
        return jsonify({"error": "Task not found"}), 404
    
    messages = []
    errors = []
    pid_to_terminate = task.get('process_pid')
    current_status = task.get('status')

    # 2. Attempt to terminate the OLMOCR process (Linux only, if running)
    process_terminated = False
    if platform.system() == "Linux" and pid_to_terminate and current_status == 'processing':
        logger.info(f"[Task {task_id}] Attempting to terminate OLMOCR process with PID: {pid_to_terminate}")
        try:
            parent = psutil.Process(pid_to_terminate)
            children = parent.children(recursive=True)
            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                    messages.append(f"Sent SIGTERM to child process {child.pid}")
                except psutil.NoSuchProcess:
                    messages.append(f"Child process {child.pid} already exited.")
                except Exception as child_err:
                    err_msg = f"Error terminating child process {child.pid}: {child_err}"
                    logger.error(f"[Task {task_id}] {err_msg}")
                    errors.append(err_msg)

            # Wait briefly for children to exit
            _, alive = psutil.wait_procs(children, timeout=0.5)
            for p in alive:
                try: 
                    p.kill(); 
                    messages.append(f"Sent SIGKILL to lingering child process {p.pid}")
                except psutil.NoSuchProcess: pass # Already gone
                except Exception as kill_err: errors.append(f"Error killing child {p.pid}: {kill_err}")

            # Terminate parent
            try:
                parent.terminate()
                messages.append(f"Sent SIGTERM to main process {parent.pid}")
                try:
                    parent.wait(timeout=1) # Wait for graceful exit
                    process_terminated = True
                    messages.append(f"Main process {parent.pid} terminated gracefully.")
                except psutil.TimeoutExpired:
                    logger.warning(f"[Task {task_id}] Main process {parent.pid} did not terminate gracefully, sending SIGKILL.")
                    parent.kill()
                    process_terminated = True
                    messages.append(f"Sent SIGKILL to main process {parent.pid}")
            except psutil.NoSuchProcess:
                 messages.append(f"Main process {pid_to_terminate} already exited.")
                 process_terminated = True # Considered terminated if not found
            except Exception as parent_err:
                 err_msg = f"Error terminating main process {pid_to_terminate}: {parent_err}"
                 logger.error(f"[Task {task_id}] {err_msg}")
                 errors.append(err_msg)

        except psutil.NoSuchProcess:
            messages.append(f"Process with PID {pid_to_terminate} not found (already exited?).")
            process_terminated = True # Consider terminated if not found
        except psutil.AccessDenied:
            err_msg = f"Permission denied trying to terminate process {pid_to_terminate}."
            logger.error(f"[Task {task_id}] {err_msg}")
            errors.append(err_msg)
        except Exception as e:
            err_msg = f"An unexpected error occurred during process termination for PID {pid_to_terminate}: {e}"
            logger.exception(f"[Task {task_id}] Unexpected termination error.")
            errors.append(err_msg)
    elif platform.system() != "Linux":
         messages.append(f"Skipping process termination: Not on Linux (OS: {platform.system()}).")
    elif not pid_to_terminate:
         messages.append(f"Skipping process termination: PID not found in task record.")
    elif current_status != 'processing':
         messages.append(f"Skipping process termination: Task status is '{current_status}', not 'processing'.")

    # 3. Update Task Status in DB to 'cancelled' if termination was attempted or successful
    # if process_terminated or (platform.system() == "Linux" and pid_to_terminate and current_status == 'processing'): # Mark cancelled if termination was attempted
    #     try:
    #         update_task_in_db(task_id, {"status": "cancelled", "error": "Task cancelled by user request.", "process_pid": None}) # Clear PID after cancellation attempt
    #         messages.append(f"Updated task {task_id} status to 'cancelled' in database.")
    #     except Exception as db_err:
    #         err_msg = f"Failed to update task {task_id} status to cancelled in DB: {db_err}"
    #         logger.error(f"[Task {task_id}] {err_msg}")
    #         errors.append(err_msg)

    # 4. Attempt to delete associated files (using paths from DB record)
    html_path = task.get("html_path") # Directly access from dict
    jsonl_path = task.get("jsonl_path") # Directly access from dict
    # Also consider the original uploaded file in UPLOAD_TEMP_DIR if needed
    # upload_filename = f"{task_id}_{task.get('original_filename')}" # Reconstruct potential upload name
    # upload_path = os.path.join(UPLOAD_TEMP_DIR, upload_filename) # Needs sanitization consistency

    if html_path and isinstance(html_path, str):
        try:
            if os.path.exists(html_path): os.remove(html_path); msg = f"Deleted HTML file: {html_path}"; logger.info(f"[Task {task_id}] {msg}"); messages.append(msg)
            else: msg = f"HTML file not found on disk: {html_path}"; logger.warning(f"[Task {task_id}] {msg}"); messages.append(msg)
        except OSError as e: err_msg = f"Error deleting HTML file {html_path}: {e}"; logger.error(f"[Task {task_id}] {err_msg}"); errors.append(err_msg)
    else: messages.append("No HTML file path found in task record or path invalid.")

    if jsonl_path and isinstance(jsonl_path, str):
        try:
            if os.path.exists(jsonl_path): os.remove(jsonl_path); msg = f"Deleted JSONL file: {jsonl_path}"; logger.info(f"[Task {task_id}] {msg}"); messages.append(msg)
            else: msg = f"JSONL file not found on disk: {jsonl_path}"; logger.warning(f"[Task {task_id}] {msg}"); messages.append(msg)
        except OSError as e: err_msg = f"Error deleting JSONL file {jsonl_path}: {e}"; logger.error(f"[Task {task_id}] {err_msg}"); errors.append(err_msg)
    else: messages.append("No JSONL file path found in task record or path invalid.")
        
    # 5. Remove the task entry from the database
    if delete_task_from_db(task_id):
        messages.append(f"Removed task record {task_id} from database.")
    else:
        # This case means it was already gone from DB, or DB delete failed
        errors.append(f"Task record {task_id} was not found in database for deletion (or DB delete failed).")
    # messages.append(f"Kept task record {task_id} in database with status 'cancelled'.")

    # 6. Return response
    if errors:
        return jsonify({"message": f"Task {task_id} deletion process finished with errors (process termination attempted/verified, files cleaned).", "details": messages, "errors": errors}), 200 # Return 200 even with errors, as action was attempted
    else:
        return jsonify({"message": f"Task {task_id} deleted successfully (process termination attempted/verified, files cleaned, DB record removed).", "details": messages}), 200

if __name__ == '__main__':
    # Ensure UPLOAD_TEMP_DIR exists before running
    os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
    
    # --- Initialize Database ---
    init_db()
    # --- End DB Init ---

    # --- Start Worker Threads ---
    logger.info(f"Starting {MAX_WORKERS} processing worker threads...")
    for i in range(MAX_WORKERS):
        worker_thread = threading.Thread(target=processing_worker, daemon=True)
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