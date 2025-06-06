import os
import shutil
import tempfile
import glob
import subprocess
import json
import time
import logging
import zipfile
import html
import psutil # Added for system stats
import platform # Added for OS check
import signal # Needed for checking termination signals
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except Exception as e:
    GPU_MONITORING_AVAILABLE = False
    gpu_error_message = f"无法初始化 NVML 进行 GPU 监控: {e}. 请确保已安装 NVIDIA 驱动和 pynvml 库。"
    print(f"WARN: {gpu_error_message}")

# --- Configuration (Copied and adapted from app.py) ---
gradio_workspace_name = "api_workspace" # Use a different name if needed
GRADIO_WORKSPACE_DIR = os.path.join(os.getcwd(), gradio_workspace_name) # Base API workspace
PROCESSED_PDF_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "processed_pdfs")
PROCESSED_JSONL_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "processed_jsonl")
PROCESSED_PREVIEW_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "html_previews")
EXPORT_TEMP_DIR_BASE = os.path.join(GRADIO_WORKSPACE_DIR, "export_temp")
UPLOAD_TEMP_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "uploads") # For temporary uploads via API

def ensure_dirs():
    """Ensures all necessary directories exist."""
    os.makedirs(GRADIO_WORKSPACE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_PDF_DIR, exist_ok=True)
    os.makedirs(PROCESSED_JSONL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_PREVIEW_DIR, exist_ok=True)
    os.makedirs(EXPORT_TEMP_DIR_BASE, exist_ok=True)
    os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

ensure_dirs() # Create directories when the module is imported

# --- Logging Setup ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Core Processing Logic (Adapted from app.py) ---

def run_olmocr_on_single_pdf(pdf_filepath, task_id, params, update_callback):
    """
    Runs OLMOCR on a single PDF file and updates the task status via callback.
    This function is intended to be run in a separate thread.

    Args:
        pdf_filepath (str): Absolute path to the PDF file to process.
        task_id (str): The ID of the task for status updates.
        params (dict): Dictionary containing OLMOCR parameters.
        update_callback (callable): Function to call for updating task status in DB.
                                   Expected signature: update_callback(task_id, updates_dict)
    """
    run_dir = None
    persistent_pdf_path = None
    persistent_jsonl_path = None
    # current_file_name_with_uuid = os.path.basename(pdf_filepath) # e.g., safe_base_uuid.ext
    current_logs = [] # Keep track of logs for this run to update DB at once
    # safe_base_name = "" # Will be extracted later
    start_time = time.time() # Record start time for calculating final duration
    olmocr_process = None # Initialize process variable
    actual_processing_start_time = None # Added to store the actual start time

    def update_task_status(status, log_message=None, result_updates=None, stdout=None, stderr=None, error_msg=None, process_pid=None, processing_start_time=None):
        """Internal helper to prepare update dict and call the callback."""
        updates = {"status": status}
        if log_message:
            # Append log to current list for this run
            log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {log_message}"
            current_logs.append(log_entry)
            logger.info(f"[Task {task_id}][{current_file_name_with_uuid}] {log_message}")
            updates["logs"] = json.dumps(current_logs) # Update with full log list (as JSON)
        
        if result_updates: # Should be a dict like {'jsonl_path': path, 'html_path': path}
             # We need to update specific keys, not overwrite the whole result
            if "jsonl_path" in result_updates:
                updates["jsonl_path"] = result_updates["jsonl_path"]
            if "html_path" in result_updates:
                updates["html_path"] = result_updates["html_path"]

        if stdout:
            updates["olmocr_stdout"] = stdout
        if stderr:
            updates["olmocr_stderr"] = stderr
        if error_msg:
             updates["error"] = error_msg
        
        if process_pid is not None: # Add pid to updates if provided
            updates["process_pid"] = process_pid
        
        if processing_start_time is not None: # Add processing_start_time if provided
            updates["processing_start_time"] = processing_start_time

        # Calculate and add final elapsed time when task reaches a terminal state
        terminal_states = ["completed", "failed", "completed_with_warnings", "cancelled"] # Add cancelled state
        if status in terminal_states:
             # Calculate final time based on actual processing start if available
             if actual_processing_start_time:
                 final_time = time.time() - actual_processing_start_time
             else:
                 # Fallback to original start_time if processing didn't actually start
                 final_time = time.time() - start_time 
                 logger.warning(f"[Task {task_id}] Calculating final elapsed time based on queue start time because actual processing start time was not recorded.")
             updates["final_elapsed_time"] = final_time
             logger.info(f"[Task {task_id}][{current_file_name_with_uuid}] Recorded final elapsed time: {final_time:.2f} seconds (Based on {'processing start' if actual_processing_start_time else 'queue start'})" )
        
        # Call the provided callback function to update the database
        try:
            update_callback(task_id, updates)
        except Exception as db_err:
            logger.error(f"[Task {task_id}] Failed to update task status in DB: {db_err}", exc_info=True)
            # Continue processing if possible, but status might be stale in DB

    # --- Extract Original Filename Base --- 
    current_file_name_with_uuid = os.path.basename(pdf_filepath) 
    base_name_with_uuid, ext = os.path.splitext(current_file_name_with_uuid)
    uuid_suffix = f"_{task_id}"
    if base_name_with_uuid.endswith(uuid_suffix):
        safe_base_name = base_name_with_uuid[:-len(uuid_suffix)] # Extract original safe base
    else:
        logger.warning(f"[Task {task_id}] Could not strip UUID suffix '{uuid_suffix}' from base name '{base_name_with_uuid}'. Using full base name for persistent files.")
        safe_base_name = base_name_with_uuid # Fallback
    # Log the extraction result
    logger.info(f"[Task {task_id}] Extracted safe_base_name '{safe_base_name}' and ext '{ext}' from '{current_file_name_with_uuid}'")
    update_task_status("processing", f"开始处理文件: {safe_base_name}{ext} (UUID: {task_id})") # Use extracted name in log
    # --- End Extraction ---

    try:
        # 1. Create unique temporary directory for this file's OLMOCR output
        # Use task_id for better tracking in API context
        run_dir = tempfile.mkdtemp(dir=GRADIO_WORKSPACE_DIR, prefix=f"olmocr_run_{task_id}_")
        update_task_status("processing", f"创建临时 OLMOCR 工作区: {run_dir}")

        # 2. Prepare paths and ensure input PDF is accessible
        # safe_base_name is already extracted above
        persistent_pdf_filename = safe_base_name + ext # Use extracted extension
        persistent_pdf_path = os.path.join(PROCESSED_PDF_DIR, persistent_pdf_filename)

        # Ensure the target directory exists before copying
        os.makedirs(PROCESSED_PDF_DIR, exist_ok=True)

        shutil.copy(pdf_filepath, persistent_pdf_path)
        update_task_status("processing", f"已缓存上传的文件到: {persistent_pdf_path}")

        olmocr_results_dir = os.path.join(run_dir, "results")
        # Ensure the results directory will be created by OLMOCR, or create if needed
        # os.makedirs(olmocr_results_dir, exist_ok=True)

        # 3. Construct OLMOCR Command conditionally based on params
        cmd = [
            "python", "-m", "olmocr.pipeline",
            run_dir,
            "--pdfs", persistent_pdf_path,
            # Required params (should always be present, validated by flask_app)
            "--max_page_error_rate", str(params['error_rate']), # error_rate is always set (fast or normal)
            "--max_page_retries", str(params['max_retries']),   # max_retries is always set (fast or normal)
            # Optional params (only add if present in the dictionary, e.g., for normal mode)
        ]
        if 'target_dim' in params:
            cmd.extend(["--target_longest_image_dim", str(params['target_dim'])])
        if 'anchor_len' in params:
            cmd.extend(["--target_anchor_text_len", str(params['anchor_len'])])
        if 'max_context' in params:
            cmd.extend(["--model_max_context", str(params['max_context'])])

        # Add workers (always present, default 1)
        cmd.extend(["--workers", str(params.get('workers', 1))])

        cmd_str = ' '.join(cmd)
        update_task_status("processing", f"准备执行命令: {cmd_str}")

        # 4. Run OLMOCR using Popen to get PID
        process_start_time = time.time()
        try:
            # Use Popen instead of run
            olmocr_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1, universal_newlines=True)
            
            # Update status with PID immediately and record processing start time
            current_time = time.time()
            actual_processing_start_time = current_time # Store locally
            update_task_status("processing", 
                               f"OLMOCR 进程已启动 (PID: {olmocr_process.pid})", 
                               process_pid=olmocr_process.pid, 
                               processing_start_time=actual_processing_start_time)

            # Communicate to get output and wait for process completion
            stdout, stderr = olmocr_process.communicate()
            return_code = olmocr_process.returncode
            
        except Exception as popen_err:
             # Handle potential errors during Popen execution itself
             error_message = f"执行 OLMOCR Popen 时出错: {popen_err}"
             logger.exception(f"[Task {task_id}] Error during OLMOCR Popen for {current_file_name_with_uuid}")
             update_task_status("failed", error_message, error_msg=str(popen_err))
             return # Exit processing for this file
        
        process_duration = time.time() - process_start_time

        # Log final status based on return code
        log_msg_base = f"OLMOCR 进程完成，耗时: {process_duration:.2f} 秒, 返回码: {return_code}"
        
        # Check for cancellation signal on Linux
        was_cancelled = False
        if platform.system() == "Linux" and return_code < 0:
            try:
                # Negative return code on Linux often means termination by signal
                sig = signal.Signals(-return_code)
                log_msg_base += f" (被信号 {sig.name} 终止)"
                # Assume cancellation if terminated by common signals
                if sig in (signal.SIGTERM, signal.SIGINT, signal.SIGKILL):
                     was_cancelled = True
                     update_task_status("cancelled", f"OLMOCR 进程被外部信号 ({sig.name}) 终止。", stdout=stdout, stderr=stderr)
                     # Don't proceed further if cancelled
                     return 
            except ValueError:
                 log_msg_base += f" (被未知信号 {-return_code} 终止)"
                 # Treat unknown signals potentially as cancellation too for safety
                 was_cancelled = True
                 update_task_status("cancelled", f"OLMOCR 进程被未知信号 ({-return_code}) 终止。", stdout=stdout, stderr=stderr)
                 return

        # If not cancelled, proceed with regular status update
        if not was_cancelled:
            update_task_status("processing", log_msg_base, stdout=stdout, stderr=stderr)

        # Raise error only if return code is non-zero AND it wasn't detected as cancelled
        if return_code != 0 and not was_cancelled:
            raise RuntimeError(f"OLMOCR failed (Code: {return_code}). STDERR: {stderr}")

        # --- If we reach here, OLMOCR finished successfully (return code 0) ---

        # 5. Process Result File (JSONL)
        jsonl_files_temp = glob.glob(os.path.join(olmocr_results_dir, "output_*.jsonl"))
        if not jsonl_files_temp:
            # Check if the process might have created the directory but not the file
            if not os.path.exists(olmocr_results_dir):
                 raise FileNotFoundError(f"OLMOCR 结果目录 {olmocr_results_dir} 未创建。")
            else:
                 raise FileNotFoundError(f"在 OLMOCR 结果目录 {olmocr_results_dir} 中未找到输出 JSONL 文件。检查 OLMOCR 日志了解详情。")

        temp_jsonl_path = jsonl_files_temp[0]

        if os.path.getsize(temp_jsonl_path) == 0:
            update_task_status("warning", f"OLMOCR 为 {current_file_name_with_uuid} 生成了空的 JSONL 文件。跳过文本提取和 HTML 预览。")
            # Mark as complete with warning, but don't proceed further for this file
            update_task_status("completed_with_warnings", f"处理 {current_file_name_with_uuid} 完成，但输出为空。")
            # No result path to add in this case for JSONL/HTML
            return # Exit the function for this file

        # Ensure persistent JSONL directory exists
        os.makedirs(PROCESSED_JSONL_DIR, exist_ok=True)
        persistent_jsonl_filename = f"{safe_base_name}_output.jsonl"
        persistent_jsonl_path = os.path.join(PROCESSED_JSONL_DIR, persistent_jsonl_filename)
        shutil.copy(temp_jsonl_path, persistent_jsonl_path)
        update_task_status("processing", f"结果 JSONL 文件已保存到: {persistent_jsonl_path}",
                           result_updates={"jsonl_path": persistent_jsonl_path})

        # 6. Generate HTML Preview (Robust Logic with Rename)
        # Ensure persistent preview directory exists
        os.makedirs(PROCESSED_PREVIEW_DIR, exist_ok=True)

        # --- Define expected and desired HTML filenames --- 
        # Predict the potentially long/ugly filename the viewer *might* create based on PDF path
        # Replace separators and dots, common behavior for path-to-filename conversion
        long_html_filename_pattern = persistent_pdf_path.replace(os.sep, '_').replace('.', '_') + ".html" 
        # Handle potential drive letter colon removal on Windows if applicable (basic example)
        if ':' in long_html_filename_pattern:
            long_html_filename_pattern = long_html_filename_pattern.split(':', 1)[-1]
        long_html_path_expected = os.path.join(PROCESSED_PREVIEW_DIR, long_html_filename_pattern)
        
        # Define the simple, desired filename 
        simple_html_filename = safe_base_name + ".html"
        simple_html_path = os.path.join(PROCESSED_PREVIEW_DIR, simple_html_filename)
        # --- End Filename Definitions --- 

        viewer_cmd = [
            "python", "-m", "olmocr.viewer.dolmaviewer",
            persistent_jsonl_path,
            "--output_dir", PROCESSED_PREVIEW_DIR
        ]
        update_task_status("processing", f"执行预览生成命令 (目标目录: {PROCESSED_PREVIEW_DIR}): {' '.join(viewer_cmd)}")
        viewer_process = subprocess.run(viewer_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
        update_task_status("processing", f"HTML 预览生成完成。STDOUT: {viewer_process.stdout} STDERR: {viewer_process.stderr}")

        if viewer_process.returncode == 0:
            # --- Find and Rename HTML File --- 
            update_task_status("processing", f"检查 viewer 生成的 HTML 文件 (预期长名称): {long_html_path_expected} 或 (简洁名称): {simple_html_path}")

            html_found_and_renamed = False
            max_wait_html = 5 # Seconds to wait for file system changes
            wait_interval_html = 0.5
            start_wait_html = time.time()
            final_html_path = None # Store the path we eventually use

            while time.time() - start_wait_html < max_wait_html:
                # Prioritize checking for the long/ugly name first
                if os.path.exists(long_html_path_expected):
                    update_task_status("processing", f"找到 viewer 生成的长名称 HTML 文件: {long_html_path_expected}")
                    try:
                        # Rename the long-named file to the simple name
                        if os.path.exists(simple_html_path):
                            # If simple name already exists, remove the long one to avoid conflict? Or just use the existing simple one.
                            # Let's favour using the existing simple one and remove the long one.
                            logger.warning(f"简洁 HTML 文件 {simple_html_path} 已存在。删除 viewer 生成的长文件名版本 {long_html_path_expected}")
                            os.remove(long_html_path_expected)
                            final_html_path = simple_html_path
                        else:
                            os.rename(long_html_path_expected, simple_html_path)
                            final_html_path = simple_html_path
                        update_task_status("processing", f"HTML 文件路径确定为 (简洁名称): {final_html_path}",
                                           result_updates={"html_path": final_html_path})
                        html_found_and_renamed = True
                        break # Exit loop after successful rename/confirmation
                    except OSError as rename_err:
                        update_task_status("warning", f"重命名/删除 HTML 文件失败 从 {long_html_path_expected} 到 {simple_html_path}: {rename_err}")
                        # Fallback: Try to use the long name if rename fails?
                        if os.path.exists(long_html_path_expected):
                             final_html_path = long_html_path_expected
                             update_task_status("warning", f"无法重命名，将使用长名称HTML路径: {final_html_path}", result_updates={"html_path": final_html_path})
                             html_found_and_renamed = True # Mark as found even if not renamed
                        break # Exit loop even if rename fails 
                
                # If long name not found, check if the simple name *already* exists
                elif os.path.exists(simple_html_path):
                    update_task_status("processing", f"找到已存在的简洁名称 HTML 文件: {simple_html_path}",
                                       result_updates={"html_path": simple_html_path})
                    final_html_path = simple_html_path
                    html_found_and_renamed = True
                    break
                    
                time.sleep(wait_interval_html)

            if not html_found_and_renamed:
            # --- End Find and Rename --- 

            # Log a warning if neither file was found after viewer success
                update_task_status("warning", f"未找到预期的 HTML 文件...")
        else:
            update_task_status("warning", f"生成 HTML 预览失败。返回码: {viewer_process.returncode}")

        # Mark task as completed successfully 
        # If we reached here without exceptions or specific non-completed statuses set earlier,
        # then it's considered completed.
        update_task_status("completed", f"文件 {current_file_name_with_uuid} 处理成功。")

    except Exception as e:
        error_message = f"处理文件 {current_file_name_with_uuid} 时发生错误: {e}"
        logger.exception(f"[Task {task_id}] Error processing file {current_file_name_with_uuid}")
        # Check if the process was running and update wasn't already "cancelled"
        current_db_status = None
        try:
            # Quick check of DB status before overwriting with "failed"
            # This requires modifying get_task_from_db to be callable here or passing status down
            # For now, we'll just update to failed, but cancellation takes precedence if it happened.
            # A better approach might involve a dedicated task state manager.
            pass # Avoid direct DB call here for simplicity, rely on previous checks
        except Exception: pass 
        
        # Only update to "failed" if not already marked "cancelled" (though cancellation should return early)
        # This logic might be redundant if cancellation correctly updates and returns.
        update_task_status("failed", error_message, error_msg=str(e))

    finally:
        # Ensure the process is cleaned up if it's still somehow alive (shouldn't be if communicate finished)
        if olmocr_process and olmocr_process.poll() is None:
            logger.warning(f"[Task {task_id}] OLMOCR process (PID: {olmocr_process.pid}) still alive after communicate? Attempting termination.")
            try:
                olmocr_process.terminate()
                olmocr_process.wait(timeout=1) # Short wait
                if olmocr_process.poll() is None:
                    olmocr_process.kill()
                    logger.info(f"[Task {task_id}] OLMOCR process (PID: {olmocr_process.pid}) killed.")
                else:
                    logger.info(f"[Task {task_id}] OLMOCR process (PID: {olmocr_process.pid}) terminated.")
            except Exception as term_err:
                logger.error(f"[Task {task_id}] Error terminating lingering OLMOCR process {olmocr_process.pid}: {term_err}")

        # Cleanup the temporary run directory for this file
        if run_dir and os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
                logger.info(f"[Task {task_id}][{current_file_name_with_uuid}] Cleaned up temporary run directory: {run_dir}")
            except OSError as e:
                logger.error(f"[Task {task_id}][{current_file_name_with_uuid}] Failed to remove temporary run directory {run_dir}: {e}")
                update_task_status("warning", f"无法删除临时运行目录 {run_dir}: {e}")

# --- System Status Function ---
def get_system_status():
    """Gathers system status including CPU, Memory, and GPU (if available)."""
    status = {}

    # CPU Info
    try:
        # Get CPU usage percentage over 1 second interval
        status['cpu'] = {
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "percent_usage": psutil.cpu_percent(interval=0.5) # Use a short interval
        }
    except Exception as e:
        logger.error(f"Failed to get CPU stats: {e}")
        status['cpu'] = {"error": f"Failed to get CPU stats: {e}"}

    # Memory Info
    try:
        mem = psutil.virtual_memory()
        status['memory'] = {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent_used": mem.percent
        }
    except Exception as e:
        logger.error(f"Failed to get Memory stats: {e}")
        status['memory'] = {"error": f"Failed to get Memory stats: {e}"}

    # GPU Info (using existing function)
    status['gpu'] = get_gpu_stats() # This already returns a dict with an 'error' key if failed

    # NUMA Info
    numa_info = {}
    if platform.system() == "Linux":
        try:
            # Try to run numactl to get hardware info
            process = subprocess.run(
                ["numactl", "--hardware"],
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit
                timeout=5 # Add a timeout
            )
            if process.returncode == 0:
                numa_info["available"] = True
                numa_info["raw_output"] = process.stdout
                # Basic parsing attempt (Example - might need adjustments)
                try:
                    lines = process.stdout.strip().split('\n')
                    nodes_line = next((line for line in lines if line.startswith("available:")), None)
                    if nodes_line:
                        numa_info["nodes_summary"] = nodes_line
                        # Extract number of nodes if possible (very basic)
                        parts = nodes_line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                             numa_info["node_count"] = int(parts[1])
                    else:
                        numa_info["parsing_notes"] = "Could not find 'available:' line in output."
                except Exception as parse_err:
                    numa_info["parsing_error"] = f"Error parsing numactl output: {parse_err}"
            else:
                numa_info["available"] = False
                numa_info["error"] = f"'numactl --hardware' command failed with code {process.returncode}. STDERR: {process.stderr.strip()}"
                if process.returncode == 127: # Often means command not found
                     numa_info["error"] += " (Is numactl installed?)"

        except FileNotFoundError:
            numa_info["available"] = False
            numa_info["error"] = "'numactl' command not found. Is it installed and in PATH?"
        except subprocess.TimeoutExpired:
            numa_info["available"] = False
            numa_info["error"] = "'numactl --hardware' command timed out."
        except Exception as e:
            numa_info["available"] = False
            numa_info["error"] = f"Error running numactl: {e}"
    else:
        numa_info["available"] = False
        numa_info["info"] = f"NUMA details via numactl are only attempted on Linux. Current OS: {platform.system()}"

    status['numa'] = numa_info

    return status

# --- GPU Stats Function (Copied from app.py) ---
def get_gpu_stats():
    # ... (Keep the implementation from app.py)
    if not GPU_MONITORING_AVAILABLE:
        return {"error": gpu_error_message}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats = {
            "gpu_utilization_percent": util.gpu,
            "memory_controller_utilization_percent": util.memory,
            "memory_used_gb": mem_info.used / (1024**3),
            "memory_total_gb": mem_info.total / (1024**3),
            "memory_used_percent": mem_info.used * 100 / mem_info.total if mem_info.total > 0 else 0
        }
        return stats
    except pynvml.NVMLError as error:
        error_msg = f"获取 GPU 状态失败: {error}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"获取 GPU 状态时出错: {e}"
        logger.error(f"获取 GPU 状态时发生意外错误: {e}")
        return {"error": error_msg}

# --- Cleanup Functions (Adapted from app.py) ---
def clear_temp_workspace():
    # ... (Keep the implementation from app.py, maybe return dict)
    cleared_count = 0
    error_count = 0
    messages = []
    logger.info(f"Attempting to clear temporary run directories in: {GRADIO_WORKSPACE_DIR}")

    try:
        for item in os.listdir(GRADIO_WORKSPACE_DIR):
            item_path = os.path.join(GRADIO_WORKSPACE_DIR, item)
            if os.path.isdir(item_path) and item.startswith("olmocr_run_"): # Specific prefix
                try:
                    shutil.rmtree(item_path)
                    messages.append(f"已删除: {item_path}")
                    logger.info(f"Removed directory: {item_path}")
                    cleared_count += 1
                except OSError as e:
                    messages.append(f"错误：无法删除 {item_path}: {e}")
                    logger.error(f"Failed to remove directory {item_path}: {e}")
                    error_count += 1
    except Exception as e:
        messages.append(f"清理临时目录时发生错误: {e}")
        logger.exception(f"Error during temporary workspace cleanup of {GRADIO_WORKSPACE_DIR}")
        error_count += 1

    final_message = f"清理完成。删除 {cleared_count} 个临时运行目录。" + (f" {error_count} 个无法删除。" if error_count > 0 else "")
    messages.append(final_message)
    logger.info(final_message)
    return {"cleared_count": cleared_count, "error_count": error_count, "messages": messages}

def clear_all_processed_data():
    # ... (Keep the implementation from app.py, maybe return dict)
    error_count = 0
    messages = []
    cleared_dirs = []
    skipped_dirs = []
    dirs_to_clear = [PROCESSED_PDF_DIR, PROCESSED_JSONL_DIR, PROCESSED_PREVIEW_DIR]
    logger.info(f"Attempting to clear processed data directories: {dirs_to_clear}")

    for dir_path in dirs_to_clear:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True) # Recreate
                messages.append(f"已清空目录: {dir_path}")
                logger.info(f"Successfully cleared and recreated directory: {dir_path}")
                cleared_dirs.append(dir_path)
            except OSError as e:
                messages.append(f"错误：无法清空目录 {dir_path}: {e}")
                logger.error(f"Failed to clear directory {dir_path}: {e}")
                error_count += 1
        else:
            messages.append(f"目录不存在，无需清理: {dir_path}")
            logger.info(f"Directory does not exist, skipping cleanup: {dir_path}")
            skipped_dirs.append(dir_path)

    final_message = "已处理文件缓存清理完成。" + (" 清理过程中出现错误。" if error_count > 0 else "")
    messages.append(final_message)
    return {"cleared_dirs": cleared_dirs, "skipped_dirs": skipped_dirs, "error_count": error_count, "messages": messages}

# --- Listing Functions (Adapted from app.py) ---
def list_files(directory, extension):
    """Lists files with a specific extension in a directory."""
    files = []
    if os.path.exists(directory):
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(extension):
                    files.append(filename)
            # Sort alphabetically for consistency
            files.sort()
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
    return files

def list_preview_files():
    return list_files(PROCESSED_PREVIEW_DIR, ".html")

def list_jsonl_files():
    return list_files(PROCESSED_JSONL_DIR, ".jsonl")

# --- Helper function for zipping (Copied from app.py) ---
def create_zip_from_dir(dir_path, zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=dir_path)
                    zipf.write(file_path, arcname=arcname)
        logger.info(f"Successfully created zip file: {zip_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating zip file {zip_path} from {dir_path}: {e}")
        return False

# --- Export Functions (Adapted from app.py) ---
def export_html_archive():
    if not os.listdir(PROCESSED_PREVIEW_DIR):
         return {"status": "error", "message": "没有 HTML 预览文件可打包。", "zip_path": None}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_filename = f"olmocr_html_export_{timestamp}.zip"
    # Ensure export base exists
    os.makedirs(EXPORT_TEMP_DIR_BASE, exist_ok=True)
    zip_file_path = os.path.join(EXPORT_TEMP_DIR_BASE, zip_filename)

    if create_zip_from_dir(PROCESSED_PREVIEW_DIR, zip_file_path):
        return {"status": "success", "message": f"HTML 文件已成功打包到 {zip_filename}。", "zip_path": zip_file_path}
    else:
        return {"status": "error", "message": "打包 HTML 文件时出错。", "zip_path": None}

def run_conversion_script(script_name, input_dir, output_dir):
    # Construct the path relative to the location of api_utils.py
    # Go up one level ('..'), then into 'scripts'
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", script_name)
    script_path = os.path.abspath(script_path) # Ensure it's an absolute path for robustness

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"转换脚本未找到: {script_path}")

    cmd = ["python", script_path, input_dir, output_dir]
    logger.info(f"运行转换脚本: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')
    logger.info(f"脚本 {script_name} STDOUT:\n{process.stdout}")
    logger.error(f"脚本 {script_name} STDERR:\n{process.stderr}")
    if process.returncode != 0:
        raise RuntimeError(f"脚本 {script_name} 执行失败 (Code: {process.returncode}): {process.stderr}")
    logger.info(f"脚本 {script_name} 执行成功。")
    # Return the list of generated files for further processing
    return glob.glob(os.path.join(output_dir, "*"))

def export_combined_archive(export_format):
    if export_format not in ['md', 'docx']:
        return {"status": "error", "message": "无效的导出格式。", "zip_path": None}

    if not os.listdir(PROCESSED_JSONL_DIR):
        return {"status": "error", "message": "没有 JSONL 文件可供转换。", "zip_path": None}

    export_temp_dir = None
    zip_file_path = None
    logs = []

    try:
        export_temp_dir = tempfile.mkdtemp(dir=EXPORT_TEMP_DIR_BASE, prefix=f"{export_format}_export_")
        logs.append(f"创建临时导出目录: {export_temp_dir}")

        script_name = "local_jsonl_to_md.py" if export_format == 'md' else "jsonl_to_docx.py"
        logs.append(f"运行 {script_name}...")
        generated_files = run_conversion_script(script_name, PROCESSED_JSONL_DIR, export_temp_dir)
        logs.append(f"{export_format.upper()} 文件已生成 ({len(generated_files)} 个)。")

        copied_html_count = 0
        logs.append("复制 HTML 预览文件...")
        for gen_file_path in generated_files:
            # base_name now comes from the MD/DOCX file, should match safe_base_name used earlier
            base_name = os.path.splitext(os.path.basename(gen_file_path))[0]
            
            # Construct the simple HTML filename to look for (should match the name used earlier)
            simple_html_filename_to_find = base_name + ".html"
            html_src_path = os.path.join(PROCESSED_PREVIEW_DIR, simple_html_filename_to_find)

            if os.path.exists(html_src_path):
                # Destination filename is also the simple name
                html_dest_path = os.path.join(export_temp_dir, simple_html_filename_to_find)
                try:
                    shutil.copy(html_src_path, html_dest_path)
                    copied_html_count += 1
                    # Update log message to reflect simple name being copied
                    logger.info(f"Copied HTML: {simple_html_filename_to_find}") 
                except Exception as copy_e:
                    logger.warning(f"无法复制 HTML 文件 {html_src_path}: {copy_e}")
                    # Log the simple name we tried to find
                    logs.append(f"警告：无法复制 {simple_html_filename_to_find}") 
            else:
                 # Log the simple name we tried to find
                 logger.warning(f"未找到对应的 HTML 文件: {simple_html_filename_to_find} at path {html_src_path}") 
                 logs.append(f"警告：未找到 {simple_html_filename_to_find}")

        logs.append(f"已复制 {copied_html_count} 个 HTML 文件。")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_filename = f"olmocr_{export_format}_export_{timestamp}.zip"
        os.makedirs(EXPORT_TEMP_DIR_BASE, exist_ok=True)
        zip_file_path = os.path.join(EXPORT_TEMP_DIR_BASE, zip_filename)
        logs.append(f"创建 Zip 文件: {zip_filename}...")

        if create_zip_from_dir(export_temp_dir, zip_file_path):
            logs.append("导出成功完成。")
            return {"status": "success", "message": "\n".join(logs), "zip_path": zip_file_path}
        else:
            logs.append("错误：创建 Zip 文件失败。")
            return {"status": "error", "message": "\n".join(logs), "zip_path": None}

    except Exception as e:
        error_msg = f"导出过程中发生错误: {e}"
        logs.append(error_msg)
        logger.exception(f"Error during {export_format} export.")
        return {"status": "error", "message": "\n".join(logs), "zip_path": None}
    finally:
        if export_temp_dir and os.path.exists(export_temp_dir):
            try:
                shutil.rmtree(export_temp_dir)
                logger.info(f"Cleaned up temporary export directory: {export_temp_dir}")
            except OSError as e:
                logger.error(f"Failed to remove temporary export directory {export_temp_dir}: {e}")


# Shutdown NVML on exit (if initialized)
if GPU_MONITORING_AVAILABLE:
    import atexit
    def shutdown_nvml():
        try:
            pynvml.nvmlShutdown()
            print("INFO: NVML shut down successfully.")
        except pynvml.NVMLError as error:
            print(f"ERROR: Failed to shut down NVML: {error}")
    atexit.register(shutdown_nvml) 