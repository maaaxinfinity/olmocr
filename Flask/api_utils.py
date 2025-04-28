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

# --- Global Task Tracking (Simple implementation) ---
# In production, use Redis, Celery, or a database
tasks = {}

# --- Core Processing Logic (Adapted from app.py) ---

def run_olmocr_on_single_pdf(pdf_filepath, task_id, params):
    """
    Runs OLMOCR on a single PDF file and updates the task status.
    This function is intended to be run in a separate thread.

    Args:
        pdf_filepath (str): Absolute path to the PDF file to process.
        task_id (str): The ID of the task for status updates.
        params (dict): Dictionary containing OLMOCR parameters.
    """
    run_dir = None
    persistent_pdf_path = None
    persistent_jsonl_path = None
    current_file_name = os.path.basename(pdf_filepath)
    logs = [] # Store logs for this specific file

    def update_task_status(status, log_message=None, result=None):
        tasks[task_id]["status"] = status
        if log_message:
            tasks[task_id]["logs"].append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {log_message}")
            logger.info(f"[Task {task_id}][{current_file_name}] {log_message}")
        if result:
            tasks[task_id]["result"] = result

    update_task_status("processing", f"开始处理文件: {current_file_name}")

    try:
        # 1. Create unique temporary directory for this file's OLMOCR output
        # Use task_id for better tracking in API context
        run_dir = tempfile.mkdtemp(dir=GRADIO_WORKSPACE_DIR, prefix=f"olmocr_run_{task_id}_")
        update_task_status("processing", f"创建临时 OLMOCR 工作区: {run_dir}")

        # 2. Prepare paths and ensure input PDF is accessible
        # The PDF should already be in a persistent location (UPLOAD_TEMP_DIR or similar)
        # We will copy it to the persistent PROCESSED_PDF_DIR
        base_name, _ = os.path.splitext(current_file_name)
        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        persistent_pdf_filename = safe_base_name + ".pdf"
        persistent_pdf_path = os.path.join(PROCESSED_PDF_DIR, persistent_pdf_filename)

        # Ensure the target directory exists before copying
        os.makedirs(PROCESSED_PDF_DIR, exist_ok=True)

        shutil.copy(pdf_filepath, persistent_pdf_path)
        update_task_status("processing", f"已缓存上传的文件到: {persistent_pdf_path}")

        olmocr_results_dir = os.path.join(run_dir, "results")
        # Ensure the results directory will be created by OLMOCR, or create if needed
        # os.makedirs(olmocr_results_dir, exist_ok=True)

        # 3. Construct OLMOCR Command using params dict
        cmd = [
            "python", "-m", "olmocr.pipeline",
            run_dir,
            "--pdfs", persistent_pdf_path,
            "--target_longest_image_dim", str(params['target_dim']),
            "--target_anchor_text_len", str(params['anchor_len']),
            "--max_page_error_rate", str(params['error_rate']),
            "--model_max_context", str(params['max_context']),
            "--max_page_retries", str(params['max_retries']),
            "--workers", str(params.get('workers', 1))
        ]
        cmd_str = ' '.join(cmd)
        update_task_status("processing", f"准备执行命令: {cmd_str}")

        # 4. Run OLMOCR
        process_start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')
        process_duration = time.time() - process_start_time

        update_task_status("processing", f"OLMOCR 进程完成，耗时: {process_duration:.2f} 秒, 返回码: {process.returncode}")
        tasks[task_id]['olmocr_stdout'] = process.stdout
        tasks[task_id]['olmocr_stderr'] = process.stderr

        if process.returncode != 0:
            raise RuntimeError(f"OLMOCR failed (Code: {process.returncode}). STDERR: {process.stderr}")

        # 5. Process Result File (JSONL)
        jsonl_files_temp = glob.glob(os.path.join(olmocr_results_dir, "output_*.jsonl"))
        if not jsonl_files_temp:
            raise FileNotFoundError(f"在临时目录 {olmocr_results_dir} 中未找到 OLMOCR 输出 JSONL 文件。")

        temp_jsonl_path = jsonl_files_temp[0]

        if os.path.getsize(temp_jsonl_path) == 0:
            update_task_status("warning", f"OLMOCR 为 {current_file_name} 生成了空的 JSONL 文件。跳过文本提取和 HTML 预览。")
            # Mark as complete with warning, but don't proceed further for this file
            update_task_status("completed_with_warnings", f"处理 {current_file_name} 完成，但输出为空。")
            # No result path to add in this case for JSONL/HTML
            return # Exit the function for this file

        # Ensure persistent JSONL directory exists
        os.makedirs(PROCESSED_JSONL_DIR, exist_ok=True)
        persistent_jsonl_filename = f"{safe_base_name}_output.jsonl"
        persistent_jsonl_path = os.path.join(PROCESSED_JSONL_DIR, persistent_jsonl_filename)
        shutil.copy(temp_jsonl_path, persistent_jsonl_path)
        update_task_status("processing", f"结果 JSONL 文件已保存到: {persistent_jsonl_path}")
        tasks[task_id]['result']['jsonl_path'] = persistent_jsonl_path

        # 6. Generate HTML Preview
        # Ensure persistent preview directory exists
        os.makedirs(PROCESSED_PREVIEW_DIR, exist_ok=True)
        viewer_cmd = [
            "python", "-m", "olmocr.viewer.dolmaviewer",
            persistent_jsonl_path,
            "--output_dir", PROCESSED_PREVIEW_DIR
        ]
        update_task_status("processing", f"执行预览生成命令 (目标目录: {PROCESSED_PREVIEW_DIR}): {' '.join(viewer_cmd)}")
        viewer_process = subprocess.run(viewer_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
        update_task_status("processing", f"HTML 预览生成完成。STDOUT: {viewer_process.stdout} STDERR: {viewer_process.stderr}")

        if viewer_process.returncode == 0:
            # Construct expected HTML filename (using the persistent PDF path)
            path_for_html_name = persistent_pdf_path.replace(os.sep, '_').replace('.', '_')
            expected_html_filename = path_for_html_name + ".html"
            expected_html_path = os.path.join(PROCESSED_PREVIEW_DIR, expected_html_filename)
            update_task_status("processing", f"检查 HTML 文件: {expected_html_path}")

            # Wait briefly for the specific file
            html_found = False
            max_wait_html = 5
            wait_interval_html = 0.5
            start_wait_html = time.time()
            while time.time() - start_wait_html < max_wait_html:
                if os.path.exists(expected_html_path):
                    update_task_status("processing", f"找到 HTML 文件: {expected_html_path}")
                    tasks[task_id]['result']['html_path'] = expected_html_path
                    html_found = True
                    break
                time.sleep(wait_interval_html)

            if not html_found:
                # Fallback or log warning
                update_task_status("warning", f"未找到预期的 HTML 文件 {expected_html_filename}。可能需要手动检查预览目录。")
                # Try glob as a last resort - might pick the wrong one if multiple run concurrently
                viewer_html_files = glob.glob(os.path.join(PROCESSED_PREVIEW_DIR, "*.html"))
                if viewer_html_files:
                    # Maybe sort by time and pick newest? Or stick to expected?
                    # For now, just log the warning.
                    pass
        else:
            update_task_status("warning", f"生成 HTML 预览失败。返回码: {viewer_process.returncode}")

        # Mark task as completed successfully
        update_task_status("completed", f"文件 {current_file_name} 处理成功。")

    except Exception as e:
        error_message = f"处理文件 {current_file_name} 时发生错误: {e}"
        logger.exception(f"[Task {task_id}] Error processing file {current_file_name}")
        update_task_status("failed", error_message)
        tasks[task_id]['error'] = str(e)

    finally:
        # Cleanup the temporary run directory for this file
        if run_dir and os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
                logger.info(f"[Task {task_id}][{current_file_name}] Cleaned up temporary run directory: {run_dir}")
            except OSError as e:
                logger.error(f"[Task {task_id}][{current_file_name}] Failed to remove temporary run directory {run_dir}: {e}")
                update_task_status("warning", f"无法删除临时运行目录 {run_dir}: {e}")

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
    script_path = os.path.join("scripts", script_name)
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
            base_name = os.path.splitext(os.path.basename(gen_file_path))[0]
            # Construct the expected HTML filename from PDF path convention
            presumed_pdf_path = os.path.join(PROCESSED_PDF_DIR, base_name + ".pdf") # May need adjustment if naming convention changes
            expected_html_filename = presumed_pdf_path.replace(os.sep, '_').replace('.', '_') + ".html"
            html_src_path = os.path.join(PROCESSED_PREVIEW_DIR, expected_html_filename)

            if os.path.exists(html_src_path):
                html_dest_path = os.path.join(export_temp_dir, expected_html_filename)
                try:
                    shutil.copy(html_src_path, html_dest_path)
                    copied_html_count += 1
                    logger.info(f"Copied HTML: {expected_html_filename}")
                except Exception as copy_e:
                    logger.warning(f"无法复制 HTML 文件 {html_src_path}: {copy_e}")
                    logs.append(f"警告：无法复制 {expected_html_filename}")
            else:
                 logger.warning(f"未找到对应的 HTML 文件: {expected_html_filename} at path {html_src_path}")
                 logs.append(f"警告：未找到 {expected_html_filename}")

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