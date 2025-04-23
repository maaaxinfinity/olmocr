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
PROCESSED_PREVIEW_DIR = os.path.join(GRADIO_WORKSPACE_DIR, "html_previews") # Directory to keep previews
os.makedirs(GRADIO_WORKSPACE_DIR, exist_ok=True)
os.makedirs(PROCESSED_PREVIEW_DIR, exist_ok=True) # Ensure preview dir exists

# --- Logging Setup ---
# Basic logging setup for Gradio app itself
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
# --------------------

def run_olmocr_on_pdf(pdf_file_obj, target_dim, anchor_len, error_rate, max_context, max_retries, workers):
    """
    Runs OLMOCR, saves HTML preview persistently, returns results and logs.
    """
    if pdf_file_obj is None:
        return "", "错误：请先上传一个 PDF 文件。", "" # Removed file list output

    run_dir = None
    extracted_text = ""
    logs = ""
    html_content_escaped = ""
    final_html_path_persistent = None # Path to the saved preview

    try:
        # 1. Create unique temporary directory
        run_dir = tempfile.mkdtemp(dir=GRADIO_WORKSPACE_DIR, prefix="olmocr_run_") # More specific prefix
        logs += f"创建临时工作区: {run_dir}\n"
        logger.info(f"Created temporary run directory: {run_dir}")

        # 2. Prepare paths
        original_filename = os.path.basename(pdf_file_obj.name)
        base_name, _ = os.path.splitext(original_filename)
        # Sanitize base_name slightly for the output filename
        safe_base_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        input_pdf_filename = safe_base_name + ".pdf" # Use sanitized name + pdf extension

        input_pdf_path = os.path.join(run_dir, input_pdf_filename) # Use sanitized name here too
        shutil.copy(pdf_file_obj.name, input_pdf_path)
        logs += f"已复制上传的文件到: {input_pdf_path}\n"
        logger.info(f"Copied uploaded file to: {input_pdf_path}")

        olmocr_results_dir = os.path.join(run_dir, "results")

        # 3. Construct OLMOCR Command with User Parameters
        cmd = [
            "python", "-m", "olmocr.pipeline",
            run_dir, # <--- 正确：传递临时运行目录作为工作区
            "--pdfs", input_pdf_path,
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
            return extracted_text, logs, "<p style='color:red;'>处理失败，请检查日志。</p>" # Removed file list output

        # 5. Process Result File (JSONL)
        jsonl_files = glob.glob(os.path.join(olmocr_results_dir, "output_*.jsonl"))
        if not jsonl_files:
            logs += f"\n错误：未找到 OLMOCR 输出文件 (output_*.jsonl)。\n"
            return extracted_text, logs, "<p style='color:red;'>未找到结果文件。</p>" # Removed file list output

        jsonl_output_path = jsonl_files[0]
        logs += f"找到结果文件: {jsonl_output_path}\n"
        try:
            temp_extracted_text = ""
            with open(jsonl_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        temp_extracted_text += data.get("text", "") + "\n\n"
            extracted_text = temp_extracted_text.strip()
            logs += "成功提取文本内容。\n"
        except Exception as e:
            logs += f"解析结果文件时出错: {e}\n"
            return extracted_text, logs, "<p style='color:red;'>解析结果文件出错。</p>" # Removed file list output


        # 6. Generate HTML Preview using dolmaviewer
        # We run viewer inside the temporary results dir to generate the preview there
        # This avoids polluting the main preview dir with intermediate files
        viewer_cmd = ["python", "-m", "olmocr.viewer.dolmaviewer", jsonl_output_path] # Use jsonl path directly
        logs += f"执行预览生成命令: {' '.join(viewer_cmd)}\n"
        logger.info(f"Executing viewer command in {olmocr_results_dir}: {' '.join(viewer_cmd)}")
        # Run viewer command from the specific results dir to contain its output
        viewer_process = subprocess.run(viewer_cmd, capture_output=True, text=True, check=False, cwd=olmocr_results_dir)
        logs += f"--- Viewer STDOUT ---\n{viewer_process.stdout}\n--- Viewer STDERR ---\n{viewer_process.stderr}\n"

        if viewer_process.returncode == 0:
            html_preview_dir_temp = os.path.join(olmocr_results_dir, "dolma_previews")
            # Expecting html name based on jsonl name inside dolma_previews
            base_jsonl_name = os.path.splitext(os.path.basename(jsonl_output_path))[0]
            # The viewer might create slightly different names, find the first html
            temp_html_files = glob.glob(os.path.join(html_preview_dir_temp, f"{base_jsonl_name}*.html")) \
                           or glob.glob(os.path.join(html_preview_dir_temp, "*.html")) # Fallback if naming differs

            if temp_html_files:
                temp_html_path = temp_html_files[0]
                # Define the persistent path for the preview
                persistent_html_filename = f"{safe_base_name}_preview.html"
                final_html_path_persistent = os.path.join(PROCESSED_PREVIEW_DIR, persistent_html_filename)

                # Copy the generated HTML to the persistent preview directory
                try:
                    shutil.copy(temp_html_path, final_html_path_persistent)
                    logs += f"预览文件已保存到: {final_html_path_persistent}\n"
                    logger.info(f"Copied preview file to persistent storage: {final_html_path_persistent}")

                    # Load content for immediate display
                    with open(final_html_path_persistent, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    html_content_escaped = f"<iframe srcdoc='{html.escape(html_content)}' width='100%' height='800px' style='border: 1px solid #ccc;'></iframe>"
                    logs += "HTML 预览内容已加载。\n"

                except Exception as e:
                    logs += f"复制或读取 HTML 预览文件时出错: {e}\n"
                    logger.exception(f"Error copying/reading HTML preview file")
                    html_content_escaped = "<p style='color:orange;'>无法加载 HTML 预览内容。</p>"
            else:
                logs += "警告：在临时目录中未找到生成的 HTML 预览文件。\n"
                html_content_escaped = "<p style='color:orange;'>未找到 HTML 预览文件。</p>"
        else:
            logs += f"警告：生成 HTML 预览失败。返回码: {viewer_process.returncode}\n"
            html_content_escaped = "<p style='color:orange;'>生成 HTML 预览失败。</p>"

        # Return final results WITHOUT the updated file list
        return extracted_text, logs, html_content_escaped # Removed list_processed_files()

    except Exception as e:
        error_message = f"发生意外错误: {e}\n"
        logs += error_message
        logger.exception("An unexpected error occurred during OLMOCR processing.")
        return "", logs, f"<p style='color:red;'>发生意外错误: {e}</p>" # Removed file list output

    finally:
        # 7. Cleanup the temporary run directory (run_dir)
        if run_dir and os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
                logger.info(f"Cleaned up temporary run directory: {run_dir}")
            except OSError as e:
                logger.error(f"Failed to remove temporary run directory {run_dir}: {e}")
                # Optionally add to logs if needed: logs += f"警告：无法删除临时运行目录 {run_dir}: {e}\n"

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

def clear_preview_files():
    """Clears the persistent HTML preview directory."""
    error_count = 0
    messages = []
    preview_dir = PROCESSED_PREVIEW_DIR
    logger.info(f"Attempting to clear preview files in: {preview_dir}")
    messages.append(f"开始清理预览文件目录: {preview_dir}")

    if os.path.exists(preview_dir):
        try:
            # Remove the directory and its contents, then recreate it empty
            shutil.rmtree(preview_dir)
            os.makedirs(preview_dir, exist_ok=True)
            messages.append(f"已成功清空预览目录。")
            logger.info(f"Successfully cleared and recreated preview directory: {preview_dir}")
        except OSError as e:
            messages.append(f"错误：无法清空预览目录 {preview_dir}: {e}")
            logger.error(f"Failed to clear preview directory {preview_dir}: {e}")
            error_count += 1
    else:
        messages.append("预览目录不存在，无需清理。")
        logger.info("Preview directory does not exist, nothing to clear.")

    final_message = "预览文件清理完成。"
    if error_count > 0:
        final_message += " 清理过程中出现错误。"
    messages.append(final_message)
    return "\n".join(messages), [] # Return message and empty list to clear file component

def list_processed_files():
    """Lists HTML files in the persistent preview directory."""
    preview_files = []
    if os.path.exists(PROCESSED_PREVIEW_DIR):
        try:
            for filename in os.listdir(PROCESSED_PREVIEW_DIR):
                if filename.lower().endswith(".html"):
                    preview_files.append(os.path.join(PROCESSED_PREVIEW_DIR, filename))
        except Exception as e:
            logger.error(f"Error listing preview files in {PROCESSED_PREVIEW_DIR}: {e}")
    return preview_files

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan")) as demo:
    gr.Markdown("# OLMOCR PDF 分析工具")
    gr.Markdown('上传 PDF 文件，调整参数（可选），然后点击"开始分析"。日志将在处理完成后显示。处理结果（HTML预览）会保存在工作区。')

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
                clear_preview_button = gr.Button("清理已保存的预览文件", variant="stop")
                clear_output = gr.Textbox(label="清理状态", interactive=False, lines=3)


        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("提取结果 (文本)"):
                    text_output = gr.Textbox(label="提取的文本内容", lines=25, interactive=False)
                with gr.TabItem("HTML 预览 (当前文件)"): # Changed label slightly
                    html_output = gr.HTML(label="预览 (HTML)")
                with gr.TabItem("处理日志 (完成后显示)"):
                    log_output = gr.Textbox(label="日志和状态", lines=30, interactive=False)
                with gr.TabItem("已处理文件列表"):
                    refresh_files_button = gr.Button("刷新列表")
                    processed_files_output = gr.Files(label="已保存的预览文件 (可下载)", file_count="multiple")


    # Button actions
    analyze_button.click(
        fn=run_olmocr_on_pdf,
        inputs=[pdf_input, target_dim_slider, anchor_len_slider, error_rate_slider, model_context_input, max_retries_input, workers_input],
        outputs=[text_output, log_output, html_output] # Removed processed_files_output
    )

    clear_temp_button.click(
        fn=clear_temp_workspace,
        inputs=[],
        outputs=[clear_output]
    )

    clear_preview_button.click(
        fn=clear_preview_files,
        inputs=[],
        outputs=[clear_output, processed_files_output] # Clear file list when previews are cleared
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