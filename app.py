import gradio as gr
import subprocess
import os
import tempfile
import shutil
import json
import glob
import time
import logging

# --- Configuration ---
GRADIO_WORKSPACE_DIR = "gradio_workspace"
os.makedirs(GRADIO_WORKSPACE_DIR, exist_ok=True)

# --- Logging Setup ---
# Basic logging setup for Gradio app itself
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
# --------------------

def run_olmocr_on_pdf(pdf_file_obj):
    """
    Runs the OLMOCR pipeline on the uploaded PDF file.

    Args:
        pdf_file_obj: Gradio File object for the uploaded PDF.

    Returns:
        A tuple containing:
        - str: Extracted text content.
        - str: Logs and status messages.
        - str or None: Path to the generated HTML preview file for download, or None if failed.
    """
    if pdf_file_obj is None:
        return "请先上传一个 PDF 文件。", "无操作", None

    run_dir = None
    jsonl_output_path = None
    html_preview_path_final = None
    logs = ""

    try:
        # 1. Create a unique temporary directory for this run
        run_dir = tempfile.mkdtemp(dir=GRADIO_WORKSPACE_DIR)
        logs += f"创建临时工作区: {run_dir}\n"
        logger.info(f"Created temporary run directory: {run_dir}")

        # 2. Prepare paths
        # Use Gradio's temp file path directly if possible, but copy for safety and control
        # Gradio provides the temp path via pdf_file_obj.name
        original_filename = os.path.basename(pdf_file_obj.name)
        # Ensure the filename ends with .pdf for OLMOCR
        if not original_filename.lower().endswith('.pdf'):
            base, _ = os.path.splitext(original_filename)
            input_pdf_filename = base + ".pdf"
        else:
            input_pdf_filename = original_filename

        input_pdf_path = os.path.join(run_dir, input_pdf_filename)
        shutil.copy(pdf_file_obj.name, input_pdf_path)
        logs += f"已复制上传的文件到: {input_pdf_path}\n"
        logger.info(f"Copied uploaded file to: {input_pdf_path}")

        # Define OLMOCR output *results* directory (where JSONL goes)
        # OLMOCR expects the *workspace* argument to be the output directory
        olmocr_results_dir = os.path.join(run_dir, "results")
        # OLMOCR pipeline creates this directory if it doesn't exist

        # 3. Construct and Run OLMOCR Command
        # Use parameters known to work better with potential errors
        cmd = [
            "python", "-m", "olmocr.pipeline",
            olmocr_results_dir,                     # Workspace/Output directory
            "--pdfs", input_pdf_path,               # Specific PDF file path
            "--target_longest_image_dim", "512",    # Reduce image size
            "--target_anchor_text_len", "100",      # Reduce anchor text
            "--max_page_error_rate", "0.1",         # Increase error tolerance
            "--workers", "1"                        # Use a single worker for simplicity
        ]
        logs += f"执行命令: {' '.join(cmd)}\n"
        logger.info(f"Executing command: {' '.join(cmd)}")

        process_start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        process_duration = time.time() - process_start_time
        logs += f"OLMOCR 进程完成，耗时: {process_duration:.2f} 秒, 返回码: {process.returncode}\n"
        logger.info(f"OLMOCR process finished in {process_duration:.2f}s with code {process.returncode}")

        logs += f"--- OLMOCR STDOUT ---\n{process.stdout}\n"
        logs += f"--- OLMOCR STDERR ---\n{process.stderr}\n"

        if process.returncode != 0:
            logger.error(f"OLMOCR failed with code {process.returncode}. stderr: {process.stderr}")
            raise RuntimeError(f"OLMOCR 处理失败，请查看日志获取详细信息。返回码: {process.returncode}")

        # 4. Find and Process Result File (JSONL)
        jsonl_files = glob.glob(os.path.join(olmocr_results_dir, "output_*.jsonl"))
        if not jsonl_files:
            logger.error(f"Could not find output JSONL file in {olmocr_results_dir}")
            raise FileNotFoundError(f"在 {olmocr_results_dir} 中未找到 OLMOCR 输出文件 (output_*.jsonl)。")

        jsonl_output_path = jsonl_files[0] # Assume only one output for single PDF
        logs += f"找到结果文件: {jsonl_output_path}\n"
        logger.info(f"Found result file: {jsonl_output_path}")

        extracted_text = ""
        try:
            with open(jsonl_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        extracted_text += data.get("text", "") + "\n\n" # Add spacing between pages
            logs += "成功提取文本内容。\n"
            logger.info("Successfully extracted text content.")
        except Exception as e:
            logs += f"解析结果文件时出错: {e}\n"
            logger.exception(f"Error parsing result file {jsonl_output_path}")
            raise ValueError(f"解析结果文件时出错: {e}")

        # 5. Generate HTML Preview using dolmaviewer (Optional but useful)
        # Run dolmaviewer in the *parent* directory of where we want dolma_previews
        # Copy JSONL outside the temp run_dir first to control output location
        preview_base_dir = os.path.join(GRADIO_WORKSPACE_DIR, "previews")
        os.makedirs(preview_base_dir, exist_ok=True)
        temp_jsonl_for_preview = os.path.join(preview_base_dir, f"{os.path.splitext(input_pdf_filename)[0]}.jsonl")
        shutil.copy(jsonl_output_path, temp_jsonl_for_preview)

        viewer_cmd = [
            "python", "-m", "olmocr.viewer.dolmaviewer",
            temp_jsonl_for_preview
        ]
        logs += f"执行预览生成命令: {' '.join(viewer_cmd)}\n"
        logger.info(f"Executing viewer command: {' '.join(viewer_cmd)}")

        # Run viewer command from the preview_base_dir to contain its output
        viewer_process = subprocess.run(viewer_cmd, capture_output=True, text=True, check=False, cwd=preview_base_dir)

        logs += f"--- Viewer STDOUT ---\n{viewer_process.stdout}\n"
        logs += f"--- Viewer STDERR ---\n{viewer_process.stderr}\n"

        if viewer_process.returncode == 0:
            # Find the generated HTML file (usually in dolma_previews subdir)
            html_preview_dir = os.path.join(preview_base_dir, "dolma_previews")
            html_files = glob.glob(os.path.join(html_preview_dir, f"{os.path.splitext(input_pdf_filename)[0]}*.html"))
            if html_files:
                html_preview_path_final = html_files[0]
                logs += f"成功生成预览文件: {html_preview_path_final}\n"
                logger.info(f"Successfully generated preview file: {html_preview_path_final}")
            else:
                logs += "警告：未找到生成的 HTML 预览文件。\n"
                logger.warning(f"Could not find generated HTML file in {html_preview_dir}")
        else:
            logs += f"警告：生成 HTML 预览失败。返回码: {viewer_process.returncode}\n"
            logger.warning(f"dolmaviewer failed with code {viewer_process.returncode}")

        return extracted_text.strip(), logs, html_preview_path_final # Return text, logs, and path to HTML file

    except Exception as e:
        error_message = f"发生错误: {e}\n"
        logs += error_message
        logger.exception("An error occurred during OLMOCR processing.")
        return "", logs, None # Return empty text, logs, no preview on error

    finally:
        # 6. Cleanup the unique run directory
        if run_dir and os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
                logs += f"已清理临时工作区: {run_dir}\n"
                logger.info(f"Cleaned up temporary run directory: {run_dir}")
            except OSError as e:
                logs += f"警告：无法删除临时工作区 {run_dir}: {e}\n"
                logger.error(f"Failed to remove temporary directory {run_dir}: {e}")
        # Note: We don't clean up the preview files here, they remain for download


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# OLMOCR PDF 分析工具")
    gr.Markdown(
        "上传一个 PDF 文件，点击“开始分析”来使用 OLMOCR 提取文本内容。\n"
        "处理可能需要一些时间，特别是对于较大的文件或首次运行（需要下载模型）。\n"
        "推荐使用参数 `--target_longest_image_dim 512 --target_anchor_text_len 100 --max_page_error_rate 0.1` 以提高成功率。"
    )

    with gr.Row():
        pdf_input = gr.File(label="上传 PDF 文件", file_types=[".pdf"])

    analyze_button = gr.Button("开始分析")

    with gr.Tabs():
        with gr.TabItem("提取结果"):
            text_output = gr.Textbox(label="提取的文本内容", lines=20, interactive=False)
        with gr.TabItem("处理日志"):
            log_output = gr.Textbox(label="日志和状态", lines=20, interactive=False)
        with gr.TabItem("预览文件 (HTML)"):
             html_output = gr.File(label="下载 HTML 预览文件", interactive=False)
             gr.Markdown("处理成功后，此处会提供一个 HTML 文件供下载，可在浏览器中打开以并排查看 PDF 页面和提取的文本。")


    analyze_button.click(
        fn=run_olmocr_on_pdf,
        inputs=[pdf_input],
        outputs=[text_output, log_output, html_output]
    )

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch(share=False) # Set share=True to create a public link (use with caution)