import gradio as gr
import subprocess
import os
import tempfile
import shutil
import json
import glob
import time
import logging
import html # For escaping HTML content for srcdoc
import zipfile # For creating zip archives
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except Exception as e:
    GPU_MONITORING_AVAILABLE = False
    gpu_error_message = f"无法初始化 NVML 进行 GPU 监控: {e}. 请确保已安装 NVIDIA 驱动和 pynvml 库。"
    print(f"WARN: {gpu_error_message}") # Print warning on startup

# --- Configuration ---
GRADIO_WORKSPACE_DIR = "gradio_workspace"
PROCESSED_PDF_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "processed_pdfs")   # For keeping original PDFs
PROCESSED_JSONL_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "processed_jsonl") # For keeping final JSONL
PROCESSED_PREVIEW_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "html_previews") # For keeping final HTML
EXPORT_TEMP_DIR_BASE = os.path.join(GRADIO_WORKSPACE_DIR, "export_temp") # Base for export zips
os.makedirs(GRADIO_WORKSPACE_DIR, exist_ok=True)
os.makedirs(PROCESSED_PDF_DIR, exist_ok=True)
os.makedirs(PROCESSED_JSONL_DIR, exist_ok=True)
os.makedirs(PROCESSED_PREVIEW_DIR, exist_ok=True)
os.makedirs(EXPORT_TEMP_DIR_BASE, exist_ok=True) # Ensure export base exists

# --- Logging Setup ---
# Basic logging setup for Gradio app itself
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
# --------------------

# <<< --- GPU Stats Function --- >>>
def get_gpu_stats():
    """获取 GPU 使用率和显存信息"""
    if not GPU_MONITORING_AVAILABLE:
        return gpu_error_message # Return the stored error message

    try:
        # Assuming a single GPU system for simplicity (device index 0)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # Utilization rates (GPU and Memory I/O)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = f"{util.gpu}%"
        mem_util = f"{util.memory}%" # Memory Controller Util

        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_gb = f"{mem_info.used / (1024**3):.2f} GB"
        mem_total_gb = f"{mem_info.total / (1024**3):.2f} GB"
        mem_percent = f"{mem_info.used * 100 / mem_info.total:.1f}%"

        stats_str = (
            f"GPU 使用率: {gpu_util}\n"
            f"显存使用率 (控制器): {mem_util}\n"
            f"已用显存: {mem_used_gb} / {mem_total_gb} ({mem_percent})"
        )
        return stats_str
    except pynvml.NVMLError as error:
        logger.error(f"获取 GPU 状态失败: {error}")
        return f"获取 GPU 状态失败: {error}"
    except Exception as e:
        logger.error(f"获取 GPU 状态时发生意外错误: {e}")
        return f"获取 GPU 状态时出错: {e}"
# ----------------------------

def run_olmocr_on_pdf(pdf_file_list, target_dim, anchor_len, error_rate, max_context, max_retries, workers):
    """
    Runs OLMOCR sequentially on a list of uploaded PDF files, yielding status updates.
    """
    if not pdf_file_list: # Check if list is empty or None
        yield "", "错误：请先上传至少一个 PDF 文件。", "", [], "### 无文件处理"
        return

    all_extracted_text = ""
    logs = "开始处理批次...\n"
    last_successful_html = "<p>无可用预览</p>"
    current_file_status_md = "### 准备开始..."

    # Initial yield to clear previous state and show starting status
    yield "", logs, last_successful_html, list_processed_files(), current_file_status_md

    total_files = len(pdf_file_list)
    processed_files_count = 0
    failed_files_count = 0

    for i, pdf_file_obj in enumerate(pdf_file_list):
        run_dir = None
        persistent_pdf_path = None
        persistent_jsonl_path = None
        current_file_name = os.path.basename(pdf_file_obj.name)
        current_file_status_md = f"### 处理中 ({i+1}/{total_files}):\n{current_file_name}"
        logs += f"\n===== 开始处理文件 {i+1}/{total_files}: {current_file_name} =====\n"

        # Yield status update before starting the file processing
        yield all_extracted_text, logs, last_successful_html, list_processed_files(), current_file_status_md

        try:
            # 1. Create unique temporary directory for this file's OLMOCR output
            run_dir = tempfile.mkdtemp(dir=GRADIO_WORKSPACE_DIR, prefix=f"olmocr_run_{i}_")
            logs += f"创建临时 OLMOCR 工作区: {run_dir}\n"
            logger.info(f"[{current_file_name}] Created temporary run directory: {run_dir}")
            # No yield here, part of setup

            # 2. Prepare paths and save input PDF persistently
            base_name, _ = os.path.splitext(current_file_name)
            safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
            persistent_pdf_filename = safe_base_name + ".pdf"
            persistent_pdf_path = os.path.join(PROCESSED_PDF_DIR, persistent_pdf_filename)

            shutil.copy(pdf_file_obj.name, persistent_pdf_path)
            logs += f"已缓存上传的文件到: {persistent_pdf_path}\n"
            logger.info(f"[{current_file_name}] Copied uploaded file to persistent storage: {persistent_pdf_path}")
            # No yield here

            olmocr_results_dir = os.path.join(run_dir, "results")

            # 3. Construct OLMOCR Command
            cmd = [
                "python", "-m", "olmocr.pipeline",
                run_dir,
                "--pdfs", persistent_pdf_path,
                "--target_longest_image_dim", str(target_dim),
                "--target_anchor_text_len", str(anchor_len),
                "--max_page_error_rate", str(error_rate),
                "--model_max_context", str(max_context),
                "--max_page_retries", str(max_retries),
                "--workers", str(workers)
            ]
            logs += f"执行命令: {' '.join(cmd)}\n"
            logger.info(f"[{current_file_name}] Executing command: {' '.join(cmd)}")
            yield all_extracted_text, logs, last_successful_html, list_processed_files(), current_file_status_md

            # 4. Run OLMOCR
            process_start_time = time.time()
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            process_duration = time.time() - process_start_time
            logs += f"--- OLMOCR 日志 [{current_file_name}] 开始 ---\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}\n--- OLMOCR 日志结束 ---\n"
            logs += f"OLMOCR 进程完成 [{current_file_name}]，耗时: {process_duration:.2f} 秒, 返回码: {process.returncode}\n"
            logger.info(f"[{current_file_name}] OLMOCR process finished in {process_duration:.2f}s with code {process.returncode}")
            # Yield after OLMOCR run completes
            yield all_extracted_text, logs, last_successful_html, list_processed_files(), current_file_status_md

            if process.returncode != 0:
                raise RuntimeError(f"OLMOCR failed (Code: {process.returncode})")

            # 5. Process Result File (JSONL)
            jsonl_files_temp = glob.glob(os.path.join(olmocr_results_dir, "output_*.jsonl"))
            if not jsonl_files_temp:
                raise FileNotFoundError(f"在临时目录 {olmocr_results_dir} 中未找到 OLMOCR 输出文件。")

            temp_jsonl_path = jsonl_files_temp[0]
            persistent_jsonl_filename = f"{safe_base_name}_output.jsonl"
            persistent_jsonl_path = os.path.join(PROCESSED_JSONL_DIR, persistent_jsonl_filename)
            shutil.copy(temp_jsonl_path, persistent_jsonl_path)
            logs += f"结果 JSONL 文件已保存到: {persistent_jsonl_path}\n"
            logger.info(f"[{current_file_name}] Copied JSONL file to persistent storage: {persistent_jsonl_path}")

            # Extract text
            file_extracted_text = ""
            with open(persistent_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        page_text = data.get("text")
                        if page_text is not None:
                             file_extracted_text += page_text + "\n\n"
            all_extracted_text += f"===== 文件: {current_file_name} =====\n\n" + file_extracted_text.strip() + "\n\n"
            logs += f"成功提取 [{current_file_name}] 文本内容。\n"
            # Yield with updated combined text
            yield all_extracted_text, logs, last_successful_html, list_processed_files(), current_file_status_md

            # 6. Generate HTML Preview
            # <<< --- START: Add check/wait for JSONL file --- >>>
            max_wait_time = 10 # Maximum seconds to wait for the JSONL file
            wait_interval = 0.5 # Seconds between checks
            start_wait_time = time.time()
            jsonl_ready = False
            while time.time() - start_wait_time < max_wait_time:
                if os.path.exists(persistent_jsonl_path) and os.path.getsize(persistent_jsonl_path) > 0: # Check existence and non-empty
                    logs += f"确认 JSONL 文件存在: {persistent_jsonl_path}\n"
                    logger.info(f"[{current_file_name}] Confirmed JSONL file exists: {persistent_jsonl_path}")
                    jsonl_ready = True
                    break
                else:
                    logs += f"等待 JSONL 文件 ({persistent_jsonl_path}) 可用...\n"
                    logger.debug(f"[{current_file_name}] Waiting for JSONL file: {persistent_jsonl_path}")
                    time.sleep(wait_interval)

            if not jsonl_ready:
                logs += f"**错误**：等待 JSONL 文件 {persistent_jsonl_path} 超时 ({max_wait_time}秒)。跳过 HTML 预览生成。\n"
                logger.error(f"[{current_file_name}] Timeout waiting for JSONL file: {persistent_jsonl_path}. Skipping viewer.")
                # Keep processing the rest, just skip HTML for this file
            else:
                 # JSONL file exists, proceed with viewer command
                viewer_cmd = [
                    "python", "-m", "olmocr.viewer.dolmaviewer",
                    persistent_jsonl_path,
                    "--output_dir", PROCESSED_PREVIEW_DIR
                 ]
                logs += f"执行预览生成命令 [{current_file_name}] (目标目录: {PROCESSED_PREVIEW_DIR}): {' '.join(viewer_cmd)}\n"
                viewer_process = subprocess.run(viewer_cmd, capture_output=True, text=True, check=False)
                logs += f"--- Viewer STDOUT [{current_file_name}] ---\n{viewer_process.stdout}\n--- Viewer STDERR ---\n{viewer_process.stderr}\n"

                if viewer_process.returncode == 0:
                    base_persistent_jsonl_name = os.path.splitext(os.path.basename(persistent_jsonl_path))[0]
                    viewer_html_files = glob.glob(os.path.join(PROCESSED_PREVIEW_DIR, f"{base_persistent_jsonl_name}*.html"))

                    if viewer_html_files:
                        final_html_path_persistent = viewer_html_files[0]
                        logs += f"预览文件已在目标目录生成: {final_html_path_persistent}\n"
                        with open(final_html_path_persistent, 'r', encoding='utf-8') as f: html_content = f.read()
                        last_successful_html = f"<iframe srcdoc='{html.escape(html_content)}' width='100%' height='800px' style='border: 1px solid #ccc;'></iframe>"
                        logs += f"[{current_file_name}] HTML 预览内容已加载。\n"
                    else:
                        logs += f"警告：[{current_file_name}] 在目标目录 {PROCESSED_PREVIEW_DIR} 中未找到生成的 HTML 预览文件。\n"
                else:
                    logs += f"警告：[{current_file_name}] 生成 HTML 预览失败。返回码: {viewer_process.returncode}\n"

            # Update status for successful file
            processed_files_count += 1
            current_file_status_md = f"### ✓ 已完成 ({i+1}/{total_files}):\n{current_file_name}"
            # Yield final state for this successful file (including updated file list)
            yield all_extracted_text, logs, last_successful_html, list_processed_files(), current_file_status_md

        except Exception as e:
            failed_files_count += 1
            error_message = f"\n**处理文件 {current_file_name} 时发生错误:** {e}\n"
            logs += error_message
            logger.exception(f"Error processing file {current_file_name}")
            current_file_status_md = f"### ✗ 失败 ({i+1}/{total_files}):\n{current_file_name} - 跳过"
            # Yield error status, keep previous successful outputs
            yield all_extracted_text, logs, last_successful_html, list_processed_files(), current_file_status_md
            continue # Move to the next file

        finally:
            # Cleanup the temporary run directory for this file
            if run_dir and os.path.exists(run_dir):
                try:
                    shutil.rmtree(run_dir)
                    logger.info(f"[{current_file_name}] Cleaned up temporary run directory: {run_dir}")
                except OSError as e:
                    logger.error(f"[{current_file_name}] Failed to remove temporary run directory {run_dir}: {e}")
                    logs += f"警告：无法删除临时运行目录 {run_dir}: {e}\n"

    # Final status update after the loop finishes
    final_status_message = f"### **批处理完成**\n成功: {processed_files_count}, 失败: {failed_files_count} / 总计: {total_files}"
    logs += f"\n===== {final_status_message} =====\n"
    logger.info(final_status_message.replace("### ","").replace("\n"," | ")) # Log concisely
    yield all_extracted_text.strip(), logs, last_successful_html, list_processed_files(), final_status_message

def clear_temp_workspace():
    """Clears only the temporary run directories (olmocr_run_*) from the Gradio workspace."""
    cleared_count = 0
    error_count = 0
    messages = []
    logger.info(f"Attempting to clear temporary run directories in: {GRADIO_WORKSPACE_DIR}")
    messages.append(f"开始清理临时运行目录...")

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

    final_message = f"清理完成。删除 {cleared_count} 个临时运行目录。"
    if error_count > 0:
        final_message += f" {error_count} 个无法删除。"
    messages.append(final_message)
    logger.info(final_message)
    return "\n".join(messages)

def clear_all_processed_data():
    """Clears the persistent PDF, JSONL, and HTML preview directories."""
    error_count = 0
    messages = []
    dirs_to_clear = [PROCESSED_PDF_DIR, PROCESSED_JSONL_DIR, PROCESSED_PREVIEW_DIR]
    logger.info(f"Attempting to clear processed data directories: {dirs_to_clear}")
    messages.append(f"开始清理已处理文件缓存...")

    for dir_path in dirs_to_clear:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True) # Recreate after clearing
                messages.append(f"已清空目录: {dir_path}")
                logger.info(f"Successfully cleared and recreated directory: {dir_path}")
            except OSError as e:
                messages.append(f"错误：无法清空目录 {dir_path}: {e}")
                logger.error(f"Failed to clear directory {dir_path}: {e}")
                error_count += 1
        else:
            messages.append(f"目录不存在，无需清理: {dir_path}")
            logger.info(f"Directory does not exist, skipping cleanup: {dir_path}")

    final_message = "已处理文件缓存清理完成。"
    if error_count > 0:
        final_message += " 清理过程中出现错误。"
    messages.append(final_message)
    # Return message and empty list to clear the gr.Files component
    return "\n".join(messages), []

def list_processed_files():
    """Lists HTML files in the persistent preview directory."""
    preview_files = []
    if os.path.exists(PROCESSED_PREVIEW_DIR):
        try:
            for filename in os.listdir(PROCESSED_PREVIEW_DIR):
                if filename.lower().endswith(".html"):
                    # Return only the filename for display, Gradio handles serving from the dir
                    preview_files.append(os.path.join(PROCESSED_PREVIEW_DIR, filename))
        except Exception as e:
            logger.error(f"Error listing preview files in {PROCESSED_PREVIEW_DIR}: {e}")
    # Sort files, maybe by modification time (newest first) or alphabetically
    try:
        preview_files.sort(key=os.path.getmtime, reverse=True)
    except OSError:
        preview_files.sort() # Fallback to alphabetical sort
    return preview_files

# --- Helper function for zipping ---
def create_zip_from_dir(dir_path, zip_path):
    """Creates a zip archive from a directory."""
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Arcname is the path inside the zip file
                    arcname = os.path.relpath(file_path, start=dir_path)
                    zipf.write(file_path, arcname=arcname)
        logger.info(f"Successfully created zip file: {zip_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating zip file {zip_path} from {dir_path}: {e}")
        return False

# --- Export Functions ---
def export_html_archive():
    """Zips all HTML files from the processed preview directory."""
    status = ""
    zip_file_path = None
    try:
        if not os.listdir(PROCESSED_PREVIEW_DIR):
             status = "没有 HTML 预览文件可打包。"
             logger.warning(status)
             return status, None

        # Create zip in the main workspace for easier access/cleanup? Or temp? Let's use temp.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_filename = f"olmocr_html_export_{timestamp}.zip"
        zip_file_path = os.path.join(EXPORT_TEMP_DIR_BASE, zip_filename)

        if create_zip_from_dir(PROCESSED_PREVIEW_DIR, zip_file_path):
            status = f"HTML 文件已成功打包到 {zip_filename}。"
            return status, zip_file_path
        else:
            status = "打包 HTML 文件时出错。"
            return status, None
    except Exception as e:
        status = f"导出 HTML 时发生错误: {e}"
        logger.exception("Error during HTML export.")
        return status, None

def run_conversion_script(script_name, input_dir, output_dir):
    """Helper to run a conversion script."""
    script_path = os.path.join("scripts", script_name) # Assumes scripts are in a 'scripts' subdir
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"转换脚本未找到: {script_path}")

    cmd = ["python", script_path, input_dir, output_dir]
    logger.info(f"运行转换脚本: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    logger.info(f"脚本 {script_name} STDOUT:\n{process.stdout}")
    logger.error(f"脚本 {script_name} STDERR:\n{process.stderr}")
    if process.returncode != 0:
        raise RuntimeError(f"脚本 {script_name} 执行失败 (Code: {process.returncode}): {process.stderr}")
    logger.info(f"脚本 {script_name} 执行成功。")


def export_combined_archive(export_format):
    """
    Exports specified format (md or docx) along with corresponding HTML previews.

    Args:
        export_format (str): Either 'md' or 'docx'.

    Returns:
        Tuple (status_message, zip_file_path or None)
    """
    status = f"开始导出 {export_format.upper()} (包含 HTML)..."
    logger.info(status)
    export_temp_dir = None
    zip_file_path = None

    if export_format not in ['md', 'docx']:
        return "无效的导出格式。", None

    if not os.listdir(PROCESSED_JSONL_DIR):
        status = "没有 JSONL 文件可供转换。"
        logger.warning(status)
        return status, None

    try:
        # Create a unique temporary directory for this export
        export_temp_dir = tempfile.mkdtemp(dir=EXPORT_TEMP_DIR_BASE, prefix=f"{export_format}_export_")
        status += f"\n创建临时导出目录: {export_temp_dir}"
        logger.info(f"Created temporary export directory: {export_temp_dir}")

        # Run the appropriate conversion script
        script_name = "local_jsonl_to_md.py" if export_format == 'md' else "jsonl_to_docx.py"
        status += f"\n运行 {script_name}..."
        run_conversion_script(script_name, PROCESSED_JSONL_DIR, export_temp_dir)
        status += f"\n{export_format.upper()} 文件已生成。"

        # Copy corresponding HTML files
        copied_html_count = 0
        status += f"\n复制 HTML 预览文件..."
        logger.info("Copying HTML files...")
        generated_files = glob.glob(os.path.join(export_temp_dir, f"*.{export_format}"))
        for gen_file_path in generated_files:
            base_name = os.path.splitext(os.path.basename(gen_file_path))[0]
            # Assume HTML name convention: base_name + "_preview.html"
            html_file_name = f"{base_name}_preview.html"
            html_src_path = os.path.join(PROCESSED_PREVIEW_DIR, html_file_name)
            if os.path.exists(html_src_path):
                html_dest_path = os.path.join(export_temp_dir, html_file_name)
                try:
                    shutil.copy(html_src_path, html_dest_path)
                    copied_html_count += 1
                except Exception as copy_e:
                    logger.warning(f"无法复制 HTML 文件 {html_src_path}: {copy_e}")
                    status += f"\n警告：无法复制 {html_file_name}"
            else:
                 logger.warning(f"未找到对应的 HTML 文件: {html_file_name}")
                 status += f"\n警告：未找到 {html_file_name}"

        status += f"\n已复制 {copied_html_count} 个 HTML 文件。"
        logger.info(f"Copied {copied_html_count} HTML files.")

        # Create the zip archive
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_filename = f"olmocr_{export_format}_export_{timestamp}.zip"
        # Place zip directly in export base dir for simpler management
        zip_file_path = os.path.join(EXPORT_TEMP_DIR_BASE, zip_filename)
        status += f"\n创建 Zip 文件: {zip_filename}..."

        if create_zip_from_dir(export_temp_dir, zip_file_path):
            status += f"\n导出成功完成。"
            return status, zip_file_path
        else:
            status += f"\n错误：创建 Zip 文件失败。"
            return status, None

    except Exception as e:
        status += f"\n导出过程中发生错误: {e}"
        logger.exception(f"Error during {export_format} export.")
        return status, None
    finally:
        # Clean up the temporary export directory
        if export_temp_dir and os.path.exists(export_temp_dir):
            try:
                shutil.rmtree(export_temp_dir)
                logger.info(f"Cleaned up temporary export directory: {export_temp_dir}")
            except OSError as e:
                logger.error(f"Failed to remove temporary export directory {export_temp_dir}: {e}")


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan")) as demo:
    gr.Markdown("# OLMOCR PDF 批处理工具")
    gr.Markdown('上传一个或多个 PDF 文件，调整参数（可选），然后点击"开始分析"。处理结果（PDF/JSONL/HTML）会保存在工作区。')

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="上传 PDF 文件 (可多选)", file_count="multiple", file_types=[".pdf"])
            gr.Markdown("### OLMOCR 参数调整")
            target_dim_slider = gr.Slider(label="图像最大尺寸", minimum=256, maximum=2048, value=720, step=64)
            anchor_len_slider = gr.Slider(label="锚点文本长度", minimum=50, maximum=32768, value=16384, step=100)
            error_rate_slider = gr.Slider(label="页面错误率", minimum=0.0, maximum=1.0, value=0.1, step=0.01)
            model_context_input = gr.Number(label="模型上下文", value=16384, minimum=1024, step=1024)
            max_retries_input = gr.Number(label="页面重试次数", value=3, minimum=0, maximum=10, step=1)
            workers_input = gr.Number(label="工作进程数", value=1, minimum=1, maximum=4, step=1, interactive=False, info="保持为1")
            analyze_button = gr.Button("开始分析", variant="primary")

            gr.Markdown("---")
            file_status_output = gr.Markdown(label="处理进度", value="**等待任务**")

            gr.Markdown("---")
            gr.Markdown("### GPU 监控")
            gpu_stats_output = gr.Textbox(label="GPU 状态", lines=3, interactive=False, every=5)

            gr.Markdown("---")
            with gr.Accordion("缓存管理", open=False):
                clear_temp_button = gr.Button("清理临时运行目录", variant="secondary")
                clear_processed_button = gr.Button("清理所有已处理文件", variant="stop")
                clear_status_output = gr.Textbox(label="清理状态", interactive=False, lines=3)


        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("提取结果 (文本)"):
                    text_output = gr.Textbox(label="所有成功处理文件的文本内容", lines=25, interactive=False)
                with gr.TabItem("HTML 预览 (最后一个成功文件)"):
                    html_output = gr.HTML(label="预览 (HTML)")
                with gr.TabItem("处理日志 (聚合)"):
                    log_output = gr.Textbox(label="所有文件的日志和状态", lines=30, interactive=False)
                with gr.TabItem("已处理文件列表与导出"):
                    refresh_files_button = gr.Button("刷新列表")
                    processed_files_output = gr.Files(label="已保存的预览文件 (HTML)", file_count="multiple", type="filepath")
                    gr.Markdown("### 导出选项")
                    with gr.Row():
                        export_html_button = gr.Button("打包下载 HTML")
                        export_md_button = gr.Button("导出 Markdown (含HTML)")
                        export_docx_button = gr.Button("导出 DOCX (含HTML)")
                    export_status_output = gr.Textbox(label="导出状态", interactive=False, lines=3)
                    export_download_output = gr.File(label="下载导出的 Zip 文件", interactive=False)


    # Button actions
    analyze_button.click(
        fn=run_olmocr_on_pdf,
        inputs=[pdf_input, target_dim_slider, anchor_len_slider, error_rate_slider, model_context_input, max_retries_input, workers_input],
        outputs=[text_output, log_output, html_output, processed_files_output, file_status_output]
    )

    clear_temp_button.click(
        fn=clear_temp_workspace,
        inputs=[],
        outputs=[clear_status_output]
    )

    clear_processed_button.click(
        fn=clear_all_processed_data,
        inputs=[],
        outputs=[clear_status_output, processed_files_output, file_status_output]
    )

    refresh_files_button.click(
        fn=list_processed_files,
        inputs=[],
        outputs=[processed_files_output]
    )

    # Export Button Actions
    export_html_button.click(
        fn=export_html_archive,
        inputs=[],
        outputs=[export_status_output, export_download_output]
    )
    from functools import partial
    export_md_button.click(
        fn=partial(export_combined_archive, export_format='md'),
        inputs=[],
        outputs=[export_status_output, export_download_output]
    )
    export_docx_button.click(
        fn=partial(export_combined_archive, export_format='docx'),
        inputs=[],
        outputs=[export_status_output, export_download_output]
    )


    # Load initial file list on app start/refresh
    demo.load(fn=list_processed_files, inputs=None, outputs=processed_files_output)


if __name__ == "__main__":
    try:
        demo.launch(share=False)
    finally:
        # <<< --- Ensure NVML is shutdown properly --- >>>
        if GPU_MONITORING_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                print("INFO: NVML shut down successfully.")
            except pynvml.NVMLError as error:
                print(f"ERROR: Failed to shut down NVML: {error}")