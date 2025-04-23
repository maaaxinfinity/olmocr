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

# --- Configuration ---
GRADIO_WORKSPACE_DIR = "gradio_workspace"
PROCESSED_PDF_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "processed_pdfs")   # For keeping original PDFs
PROCESSED_JSONL_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "processed_jsonl") # For keeping final JSONL
PROCESSED_PREVIEW_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "html_previews") # For keeping final HTML
os.makedirs(GRADIO_WORKSPACE_DIR, exist_ok=True)
os.makedirs(PROCESSED_PDF_DIR, exist_ok=True)
os.makedirs(PROCESSED_JSONL_DIR, exist_ok=True)
os.makedirs(PROCESSED_PREVIEW_DIR, exist_ok=True)

# --- Logging Setup ---
# Basic logging setup for Gradio app itself
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
# --------------------

def run_olmocr_on_pdf(pdf_file_obj, target_dim, anchor_len, error_rate, max_context, max_retries, workers):
    """
    Runs OLMOCR, saves PDF, JSONL, and HTML preview persistently.
    """
    if pdf_file_obj is None:
        return "", "错误：请先上传一个 PDF 文件。", "", []

    run_dir = None
    extracted_text = ""
    logs = ""
    html_content_escaped = ""
    persistent_pdf_path = None
    persistent_jsonl_path = None

    try:
        # 1. Create unique temporary directory for OLMOCR *output*
        run_dir = tempfile.mkdtemp(dir=GRADIO_WORKSPACE_DIR, prefix="olmocr_run_")
        logs += f"创建临时 OLMOCR 工作区: {run_dir}\n"
        logger.info(f"Created temporary run directory: {run_dir}")

        # 2. Prepare paths and save input PDF persistently
        original_filename = os.path.basename(pdf_file_obj.name)
        base_name, _ = os.path.splitext(original_filename)
        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        persistent_pdf_filename = safe_base_name + ".pdf"
        persistent_pdf_path = os.path.join(PROCESSED_PDF_DIR, persistent_pdf_filename)

        # Copy uploaded file to the persistent PDF directory
        shutil.copy(pdf_file_obj.name, persistent_pdf_path)
        logs += f"已缓存上传的文件到: {persistent_pdf_path}\n"
        logger.info(f"Copied uploaded file to persistent storage: {persistent_pdf_path}")

        # OLMOCR results will go into the temporary run_dir
        olmocr_results_dir = os.path.join(run_dir, "results")

        # 3. Construct OLMOCR Command using the persistent PDF path
        cmd = [
            "python", "-m", "olmocr.pipeline",
            run_dir,                    # OLMOCR workspace is the temp dir
            "--pdfs", persistent_pdf_path, # Input PDF is the persistent one
            "--target_longest_image_dim", str(target_dim),
            "--target_anchor_text_len", str(anchor_len),
            "--max_page_error_rate", str(error_rate),
            "--model_max_context", str(max_context),
            "--max_page_retries", str(max_retries),
            "--workers", str(workers)
        ]
        logs += f"执行命令: {' '.join(cmd)}\n"
        logger.info(f"Executing command: {' '.join(cmd)}")

        # 4. Run OLMOCR
        process_start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        process_duration = time.time() - process_start_time
        # --- Debugging ---
        # logs += f"等待文件系统操作...\n"; logger.info("Waiting..."); time.sleep(2)
        # try:
        #     if os.path.exists(olmocr_results_dir): logs += f"结果目录内容: {os.listdir(olmocr_results_dir)}\n"; logger.info(f"Results dir contents: {os.listdir(olmocr_results_dir)}")
        #     else: logs += f"结果目录不存在!\n"; logger.error("Results dir does not exist!")
        # except Exception as list_err: logs += f"列出结果目录出错: {list_err}\n"; logger.error(f"Error listing results dir: {list_err}")
        # --- End Debugging ---
        logs += f"--- OLMOCR 日志开始 ---\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}\n--- OLMOCR 日志结束 ---\n"
        logs += f"OLMOCR 进程完成，耗时: {process_duration:.2f} 秒, 返回码: {process.returncode}\n"
        logger.info(f"OLMOCR process finished in {process_duration:.2f}s with code {process.returncode}")

        if process.returncode != 0:
            logs += f"\n错误：OLMOCR 处理失败。返回码: {process.returncode}\n"
            return extracted_text, logs, "<p style='color:red;'>处理失败，请检查日志。</p>", list_processed_files()

        # 5. Process Result File (JSONL) - Find it in the temporary results dir
        jsonl_files_temp = glob.glob(os.path.join(olmocr_results_dir, "output_*.jsonl"))
        if not jsonl_files_temp:
            logs += f"\n错误：在临时目录 {olmocr_results_dir} 中未找到 OLMOCR 输出文件 (output_*.jsonl)。\n"
            return extracted_text, logs, "<p style='color:red;'>未找到结果文件。</p>", list_processed_files()

        temp_jsonl_path = jsonl_files_temp[0]
        logs += f"找到临时结果文件: {temp_jsonl_path}\n"

        # Copy JSONL to persistent directory
        persistent_jsonl_filename = f"{safe_base_name}_output.jsonl"
        persistent_jsonl_path = os.path.join(PROCESSED_JSONL_DIR, persistent_jsonl_filename)
        try:
            shutil.copy(temp_jsonl_path, persistent_jsonl_path)
            logs += f"结果 JSONL 文件已保存到: {persistent_jsonl_path}\n"
            logger.info(f"Copied JSONL file to persistent storage: {persistent_jsonl_path}")
        except Exception as e:
            logs += f"复制 JSONL 文件时出错: {e}\n"
            logger.exception("Error copying JSONL file")
            return extracted_text, logs, "<p style='color:red;'>保存结果文件出错。</p>", list_processed_files()


        # Extract text from the persistent JSONL
        try:
            temp_extracted_text = ""
            with open(persistent_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Ensure 'text' key exists and is not None before appending
                        page_text = data.get("text")
                        if page_text is not None:
                             temp_extracted_text += page_text + "\n\n"
            extracted_text = temp_extracted_text.strip()
            logs += "成功提取文本内容。\n"
        except Exception as e:
            logs += f"解析结果文件时出错: {e}\n"
            return extracted_text, logs, "<p style='color:red;'>解析结果文件出错。</p>", list_processed_files()


        # 6. Generate HTML Preview using dolmaviewer, running from parent dir
        viewer_cmd = ["python", "-m", "olmocr.viewer.dolmaviewer", persistent_jsonl_path] # Use persistent JSONL path
        logs += f"执行预览生成命令 (工作目录: {os.getcwd()}): {' '.join(viewer_cmd)}\n" # Log current CWD
        logger.info(f"Executing viewer command from default CWD: {' '.join(viewer_cmd)}")
        # Run viewer command from the script's default working directory
        viewer_process = subprocess.run(viewer_cmd, capture_output=True, text=True, check=False) # Removed cwd
        logs += f"--- Viewer STDOUT ---\n{viewer_process.stdout}\n--- Viewer STDERR ---\n{viewer_process.stderr}\n"

        if viewer_process.returncode == 0:
            # Viewer creates 'dolma_previews' in the CWD (where app.py is)
            html_preview_dir_viewer_in_cwd = "dolma_previews" # Relative path in CWD
            # Find the HTML file based on the persistent JSONL name within CWD/dolma_previews
            base_persistent_jsonl_name = os.path.splitext(os.path.basename(persistent_jsonl_path))[0]
            viewer_html_files = glob.glob(os.path.join(html_preview_dir_viewer_in_cwd, f"{base_persistent_jsonl_name}*.html")) \
                              or glob.glob(os.path.join(html_preview_dir_viewer_in_cwd, "*.html")) # Fallback

            if viewer_html_files:
                viewer_html_path_in_cwd = viewer_html_files[0] # Path relative to script CWD
                # Define the final persistent path for the preview
                persistent_html_filename = f"{safe_base_name}_preview.html"
                final_html_path_persistent = os.path.join(PROCESSED_PREVIEW_DIR, persistent_html_filename)

                # Move the generated HTML from CWD/dolma_previews to the final persistent preview directory
                try:
                    shutil.move(viewer_html_path_in_cwd, final_html_path_persistent)
                    logs += f"预览文件已移动到: {final_html_path_persistent}\n"
                    logger.info(f"Moved preview file to persistent storage: {final_html_path_persistent}")

                    # Clean up the (now possibly empty) dolma_previews dir created by viewer in CWD
                    if os.path.exists(html_preview_dir_viewer_in_cwd):
                        try:
                            os.rmdir(html_preview_dir_viewer_in_cwd) # Remove only if empty
                            logger.info(f"Removed empty viewer output dir: {html_preview_dir_viewer_in_cwd}")
                        except OSError:
                            logger.warning(f"Viewer output dir {html_preview_dir_viewer_in_cwd} not empty or other error, not removed.")
                            pass # Ignore if not empty or other error

                    # Load content for immediate display from the new persistent location
                    with open(final_html_path_persistent, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    html_content_escaped = f"<iframe srcdoc='{html.escape(html_content)}' width='100%' height='800px' style='border: 1px solid #ccc;'></iframe>"
                    logs += "HTML 预览内容已加载。\n"

                except Exception as e:
                    logs += f"移动或读取 HTML 预览文件时出错: {e}\n"
                    logger.exception(f"Error moving/reading HTML preview file")
                    html_content_escaped = "<p style='color:orange;'>无法加载 HTML 预览内容。</p>"
            else:
                logs += f"警告：在 {html_preview_dir_viewer_in_cwd} 中未找到生成的 HTML 预览文件。\n" # Updated log message
                html_content_escaped = "<p style='color:orange;'>未找到 HTML 预览文件。</p>"
        else:
            logs += f"警告：生成 HTML 预览失败。返回码: {viewer_process.returncode}\n"
            html_content_escaped = "<p style='color:orange;'>生成 HTML 预览失败。</p>"

        return extracted_text, logs, html_content_escaped, list_processed_files()

    except Exception as e:
        error_message = f"发生意外错误: {e}\n"
        logs += error_message
        logger.exception("An unexpected error occurred during OLMOCR processing.")
        return "", logs, f"<p style='color:red;'>发生意外错误: {e}</p>", list_processed_files()

    finally:
        # 7. Cleanup the temporary run directory (run_dir) - This is now safe
        if run_dir and os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
                logger.info(f"Cleaned up temporary run directory: {run_dir}")
            except OSError as e:
                logger.error(f"Failed to remove temporary run directory {run_dir}: {e}")

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

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan")) as demo:
    gr.Markdown("# OLMOCR PDF 分析工具")
    gr.Markdown('上传 PDF 文件，调整参数（可选），然后点击"开始分析"。处理结果（PDF/JSONL/HTML）会保存在工作区。')

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="上传 PDF 文件", file_types=[".pdf"])
            gr.Markdown("### OLMOCR 参数调整")
            target_dim_slider = gr.Slider(label="图像最大尺寸", minimum=256, maximum=2048, value=720, step=64)
            anchor_len_slider = gr.Slider(label="锚点文本长度", minimum=50, maximum=32768, value=16384, step=100)
            error_rate_slider = gr.Slider(label="页面错误率", minimum=0.0, maximum=1.0, value=0.1, step=0.01)
            model_context_input = gr.Number(label="模型上下文", value=16384, minimum=1024, step=1024)
            max_retries_input = gr.Number(label="页面重试次数", value=3, minimum=0, maximum=10, step=1)
            workers_input = gr.Number(label="工作进程数", value=1, minimum=1, maximum=4, step=1, interactive=False, info="保持为1")
            analyze_button = gr.Button("开始分析", variant="primary")

            gr.Markdown("---")
            with gr.Accordion("缓存管理", open=False):
                clear_temp_button = gr.Button("清理临时运行目录", variant="secondary")
                clear_processed_button = gr.Button("清理所有已处理文件 (PDF/JSONL/HTML)", variant="stop")
                clear_output = gr.Textbox(label="清理状态", interactive=False, lines=3)


        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("提取结果 (文本)"):
                    text_output = gr.Textbox(label="提取的文本内容", lines=25, interactive=False)
                with gr.TabItem("HTML 预览 (当前文件)"):
                    html_output = gr.HTML(label="预览 (HTML)")
                with gr.TabItem("处理日志 (完成后显示)"):
                    log_output = gr.Textbox(label="日志和状态", lines=30, interactive=False)
                with gr.TabItem("已处理文件列表"):
                    refresh_files_button = gr.Button("刷新列表")
                    processed_files_output = gr.Files(label="已保存的预览文件 (HTML)", file_count="multiple", type="filepath")


    # Button actions
    analyze_button.click(
        fn=run_olmocr_on_pdf,
        inputs=[pdf_input, target_dim_slider, anchor_len_slider, error_rate_slider, model_context_input, max_retries_input, workers_input],
        outputs=[text_output, log_output, html_output, processed_files_output]
    )

    clear_temp_button.click(
        fn=clear_temp_workspace,
        inputs=[],
        outputs=[clear_output]
    )

    clear_processed_button.click(
        fn=clear_all_processed_data,
        inputs=[],
        outputs=[clear_output, processed_files_output] # Clear file list as well
    )

    refresh_files_button.click(
        fn=list_processed_files,
        inputs=[],
        outputs=[processed_files_output]
    )

    # Load initial file list on app start
    demo.load(fn=list_processed_files, inputs=None, outputs=processed_files_output)


if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch(share=False)