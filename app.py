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
os.makedirs(GRADIO_WORKSPACE_DIR, exist_ok=True)

# --- Logging Setup ---
# Basic logging setup for Gradio app itself
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
# --------------------

def run_olmocr_on_pdf(pdf_file_obj, target_dim, anchor_len, error_rate, workers):
    """
    Runs the OLMOCR pipeline on the uploaded PDF file with customizable parameters
    and streams logs.

    Args:
        pdf_file_obj: Gradio File object for the uploaded PDF.
        target_dim (int): Target longest image dimension.
        anchor_len (int): Target anchor text length.
        error_rate (float): Max page error rate.
        workers (int): Number of workers (usually 1 for this setup).

    Yields:
        Tuples for updating Gradio outputs (text, logs, html_preview).
        Logs are streamed. Text and HTML preview update at the end.
    """
    if pdf_file_obj is None:
        yield "", "错误：请先上传一个 PDF 文件。", ""
        return

    run_dir = None
    jsonl_output_path = None
    html_preview_path_final = None
    extracted_text = ""
    html_content_escaped = ""
    logs = ""

    try:
        # 1. Create a unique temporary directory for this run
        run_dir = tempfile.mkdtemp(dir=GRADIO_WORKSPACE_DIR)
        logs += f"创建临时工作区: {run_dir}\n"
        logger.info(f"Created temporary run directory: {run_dir}")
        yield extracted_text, logs, html_content_escaped

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
        yield extracted_text, logs, html_content_escaped

        # Define OLMOCR output *results* directory (where JSONL goes)
        # OLMOCR expects the *workspace* argument to be the output directory
        olmocr_results_dir = os.path.join(run_dir, "results")
        # OLMOCR pipeline creates this directory if it doesn't exist

        # 3. Construct OLMOCR Command with User Parameters
        cmd = [
            "python", "-m", "olmocr.pipeline",
            olmocr_results_dir,
            "--pdfs", input_pdf_path,
            "--target_longest_image_dim", str(target_dim),
            "--target_anchor_text_len", str(anchor_len),
            "--max_page_error_rate", str(error_rate),
            "--workers", str(workers)
        ]
        logs += f"执行命令: {' '.join(cmd)}\n"
        logger.info(f"Executing command: {' '.join(cmd)}")
        yield extracted_text, logs, html_content_escaped

        # 4. Run OLMOCR with Popen for Streaming Logs
        process_start_time = time.time()
        logs += "--- OLMOCR 实时日志开始 ---\n"
        yield extracted_text, logs, html_content_escaped

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

        # Stream stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logs += line
                logger.info(f"OLMOCR STDOUT: {line.strip()}")
                yield extracted_text, logs, html_content_escaped
            process.stdout.close()

        # Stream stderr
        stderr_output = ""
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                logs += f"STDERR: {line}" # Prepend STDERR to distinguish
                stderr_output += line
                logger.info(f"OLMOCR STDERR: {line.strip()}")
                yield extracted_text, logs, html_content_escaped
            process.stderr.close()

        return_code = process.wait()
        process_duration = time.time() - process_start_time
        logs += f"--- OLMOCR 实时日志结束 ---\n"
        logs += f"OLMOCR 进程完成，耗时: {process_duration:.2f} 秒, 返回码: {return_code}\n"
        logger.info(f"OLMOCR process finished in {process_duration:.2f}s with code {return_code}")
        yield extracted_text, logs, html_content_escaped

        if return_code != 0:
            logger.error(f"OLMOCR failed with code {return_code}. stderr: {stderr_output}")
            logs += f"\n错误：OLMOCR 处理失败，请查看日志获取详细信息。返回码: {return_code}\n"
            # Yield final state with error
            yield extracted_text, logs, "<p style='color:red;'>处理失败，请检查日志。</p>"
            return # Stop processing on failure

        # 5. Find and Process Result File (JSONL)
        jsonl_files = glob.glob(os.path.join(olmocr_results_dir, "output_*.jsonl"))
        if not jsonl_files:
            logger.error(f"Could not find output JSONL file in {olmocr_results_dir}")
            logs += f"\n错误：在 {olmocr_results_dir} 中未找到 OLMOCR 输出文件 (output_*.jsonl)。\n"
            yield extracted_text, logs, "<p style='color:red;'>未找到结果文件。</p>"
            return

        jsonl_output_path = jsonl_files[0]
        logs += f"找到结果文件: {jsonl_output_path}\n"
        logger.info(f"Found result file: {jsonl_output_path}")
        yield extracted_text, logs, html_content_escaped # Update logs before text extraction

        try:
            temp_extracted_text = ""
            with open(jsonl_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        temp_extracted_text += data.get("text", "") + "\n\n"
            extracted_text = temp_extracted_text.strip() # Update the main variable
            logs += "成功提取文本内容。\n"
            logger.info("Successfully extracted text content.")
            # Yield extracted text along with logs
            yield extracted_text, logs, html_content_escaped
        except Exception as e:
            logs += f"解析结果文件时出错: {e}\n"
            logger.exception(f"Error parsing result file {jsonl_output_path}")
            yield extracted_text, logs, "<p style='color:red;'>解析结果文件出错。</p>"
            return

        # 6. Generate and Display HTML Preview
        preview_base_dir = os.path.join(GRADIO_WORKSPACE_DIR, "previews")
        os.makedirs(preview_base_dir, exist_ok=True)
        temp_jsonl_for_preview = os.path.join(preview_base_dir, f"{os.path.splitext(input_pdf_filename)[0]}.jsonl")
        shutil.copy(jsonl_output_path, temp_jsonl_for_preview)

        viewer_cmd = ["python", "-m", "olmocr.viewer.dolmaviewer", temp_jsonl_for_preview]
        logs += f"执行预览生成命令: {' '.join(viewer_cmd)}\n"
        logger.info(f"Executing viewer command: {' '.join(viewer_cmd)}")
        yield extracted_text, logs, html_content_escaped

        viewer_process = subprocess.run(viewer_cmd, capture_output=True, text=True, check=False, cwd=preview_base_dir)

        logs += f"--- Viewer STDOUT ---\n{viewer_process.stdout}\n"
        logs += f"--- Viewer STDERR ---\n{viewer_process.stderr}\n"
        yield extracted_text, logs, html_content_escaped # Update logs after viewer runs

        if viewer_process.returncode == 0:
            html_preview_dir = os.path.join(preview_base_dir, "dolma_previews")
            html_files = glob.glob(os.path.join(html_preview_dir, f"{os.path.splitext(input_pdf_filename)[0]}*.html"))
            if html_files:
                html_preview_path_final = html_files[0]
                logs += f"成功生成预览文件: {html_preview_path_final}\n"
                logger.info(f"Successfully generated preview file: {html_preview_path_final}")
                try:
                    with open(html_preview_path_final, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    # Escape for srcdoc and wrap in iframe
                    html_content_escaped = f"<iframe srcdoc='{html.escape(html_content)}' width='100%' height='800px' style='border: 1px solid #ccc;'></iframe>"
                    logs += "HTML 预览内容已加载。\n"
                    logger.info("HTML preview content loaded.")
                except Exception as e:
                    logs += f"读取 HTML 预览文件时出错: {e}\n"
                    logger.exception(f"Error reading HTML preview file {html_preview_path_final}")
                    html_content_escaped = "<p style='color:orange;'>无法加载 HTML 预览内容。</p>"
            else:
                logs += "警告：未找到生成的 HTML 预览文件。\n"
                logger.warning(f"Could not find generated HTML file in {html_preview_dir}")
                html_content_escaped = "<p style='color:orange;'>未找到 HTML 预览文件。</p>"
        else:
            logs += f"警告：生成 HTML 预览失败。返回码: {viewer_process.returncode}\n"
            logger.warning(f"dolmaviewer failed with code {viewer_process.returncode}")
            html_content_escaped = "<p style='color:orange;'>生成 HTML 预览失败。</p>"

        # Yield final state including the HTML preview
        yield extracted_text, logs, html_content_escaped

    except Exception as e:
        error_message = f"发生意外错误: {e}\n"
        logs += error_message
        logger.exception("An unexpected error occurred during OLMOCR processing.")
        yield "", logs, f"<p style='color:red;'>发生意外错误: {e}</p>" # Final error state

    finally:
        # 7. Cleanup the unique run directory
        if run_dir and os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
                # Don't yield here as it might overwrite final results if cleanup is slow
                logger.info(f"Cleaned up temporary run directory: {run_dir}")
            except OSError as e:
                logger.error(f"Failed to remove temporary directory {run_dir}: {e}")
        # Keep preview files in GRADIO_WORKSPACE_DIR/previews


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan")) as demo:
    gr.Markdown("# OLMOCR PDF 分析工具")
    gr.Markdown("上传 PDF 文件，调整参数（可选），然后点击"开始分析"。")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="上传 PDF 文件", file_types=[".pdf"])
            gr.Markdown("### OLMOCR 参数调整")
            target_dim_slider = gr.Slider(label="图像最大尺寸 (Target Longest Image Dim)", minimum=256, maximum=1024, value=512, step=64)
            anchor_len_slider = gr.Slider(label="锚点文本最大长度 (Target Anchor Text Len)", minimum=50, maximum=1000, value=100, step=50)
            error_rate_slider = gr.Slider(label="最大页面错误率 (Max Page Error Rate)", minimum=0.0, maximum=1.0, value=0.1, step=0.01)
            workers_input = gr.Number(label="工作进程数 (Workers)", value=1, minimum=1, maximum=4, step=1, interactive=False, info="通常保持为1") # Keep workers at 1 for simplicity
            analyze_button = gr.Button("开始分析", variant="primary")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("提取结果 (文本)"):
                    text_output = gr.Textbox(label="提取的文本内容", lines=25, interactive=False)
                with gr.TabItem("HTML 预览"):
                    html_output = gr.HTML(label="预览 (HTML)")
                with gr.TabItem("处理日志 (实时)"):
                    log_output = gr.Textbox(label="日志和状态", lines=30, interactive=False, autoscroll=True)


    analyze_button.click(
        fn=run_olmocr_on_pdf,
        inputs=[pdf_input, target_dim_slider, anchor_len_slider, error_rate_slider, workers_input],
        outputs=[text_output, log_output, html_output] # Ensure order matches yield/return
    )

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch(share=False) # Set share=True to create a public link (use with caution)