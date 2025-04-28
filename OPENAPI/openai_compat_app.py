from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory, abort
import os
import json
import time
import uuid
import logging
from functools import wraps

# Import the processor function and directories
from openai_processor import (
    process_pdf_openai_style,
    OPENAI_DOCX_OUTPUT_DIR,
    PROCESSED_PREVIEW_DIR # Assuming HTML previews are in the shared location
)

app = Flask(__name__)

# --- Logging Setup (Consistent with processor) ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - [OpenAICompatApp] - %(message)s'
if not logging.getLogger().handlers:
     logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- API Key Authentication (Reuse from the other app) ---
API_KEY = os.environ.get("OPENAI_API_SECRET_KEY", "default_openai_secret_key_change_me")
# NOTE: Using a *different* environment variable name to avoid conflicts if run in same env

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        key_from_header = request.headers.get('X-Api-Key') # Also check direct header if needed

        provided_key = None
        if auth_header and auth_header.startswith('Bearer '):
            provided_key = auth_header.split(' ')[1]
        elif key_from_header:
            provided_key = key_from_header

        if provided_key and provided_key == API_KEY:
            return f(*args, **kwargs)
        else:
            logger.warning(f"Unauthorized access attempt to OpenAI compat API. Provided key: {provided_key[:5] if provided_key else 'None'}...")
            abort(401, description="Unauthorized: Invalid or missing API Key (check Authorization: Bearer <key> or X-Api-Key header).")
    return decorated_function

# --- Helper to format Server-Sent Events (SSE) --- 
def format_sse(data: str, event=None) -> str:
    msg = f'data: {json.dumps(data)}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg

# --- OpenAI Compatible Chat Completions Endpoint --- 
@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    try:
        data = request.get_json()
        if not data:
            logger.warning(f"Received request with invalid/empty JSON payload.")
            return jsonify({"error": "Invalid JSON payload"}), 400
        # Log the received payload
        logger.info(f"Received /v1/chat/completions request data: {json.dumps(data)}")
        # Print the received payload to console
        print(f"---\n[INFO] Received Request Body:\n{json.dumps(data, indent=2)}\n---")
    except Exception as e:
         logger.error(f"Failed to parse JSON payload: {e}", exc_info=True)
         return jsonify({"error": f"Failed to parse JSON payload: {e}"}), 400

    # --- Extract required custom field and standard fields ---
    pdf_file_path = data.get('pdf_file_path')
    model_name = data.get('model', 'olmocr-processor')
    stream = data.get('stream', False)
    request_id = f"chatcmpl-{str(uuid.uuid4())}"
    created_time = int(time.time())
    fixed_reply = "请输入PDF文件"

    # --- Handle missing/invalid pdf_file_path --- 
    should_use_fixed_reply = False
    if not pdf_file_path or not isinstance(pdf_file_path, str):
        logger.warning(f"Request {request_id}: Missing or invalid pdf_file_path.")
        should_use_fixed_reply = True
    elif not os.path.isabs(pdf_file_path):
        logger.warning(f"Request {request_id}: pdf_file_path is not absolute: {pdf_file_path}")
        should_use_fixed_reply = True # Treat non-absolute paths as invalid for this case
    elif not os.path.exists(pdf_file_path) or not os.path.isfile(pdf_file_path):
        logger.warning(f"Request {request_id}: PDF file not found or not a file: {pdf_file_path}")
        should_use_fixed_reply = True # Treat non-existent files as needing the fixed reply

    if should_use_fixed_reply:
        if stream:
            # Send fixed reply as a single chunk stream
            delta_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": fixed_reply},
                    "finish_reason": "stop" # Stop immediately after fixed reply
                }]
            }
            def fixed_stream():
                yield format_sse(delta_chunk)
                yield format_sse("[DONE]")
            return Response(stream_with_context(fixed_stream()), mimetype='text/event-stream')
        else:
            # Send fixed reply as a standard non-stream response
            response_payload = {
                 "id": request_id,
                 "object": "chat.completion",
                 "created": created_time,
                 "model": model_name,
                 "choices": [{
                     "index": 0,
                     "message": {
                         "role": "assistant",
                         "content": fixed_reply,
                     },
                     "finish_reason": "stop"
                 }],
                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
             }
            return jsonify(response_payload), 200

    # --- If pdf_file_path is valid, proceed with normal processing --- 

    # --- Streaming Response --- 
    if stream:
        def generate_stream():
            final_result = None
            is_first_chunk = True # To identify the first yield (should be think block)

            try:
                # Expecting two yields: 1. think block (str), 2. result dict
                processor = process_pdf_openai_style(pdf_filepath=pdf_file_path)
                
                # Handle the first yield (expecting the think block)
                try:
                    first_yield = next(processor)
                    if isinstance(first_yield, str) and first_yield.startswith("<think>"):
                         think_content = first_yield
                         delta_chunk = {
                             "id": request_id,
                             "object": "chat.completion.chunk",
                             "created": created_time,
                             "model": model_name,
                             "choices": [{
                                 "index": 0,
                                 "delta": {"role": "assistant", "content": think_content},
                                 "finish_reason": None
                             }]
                         }
                         yield format_sse(delta_chunk)
                         # Optional small delay after sending thoughts
                         # time.sleep(0.05)
                    else:
                        # If the first yield wasn't the think block, something is wrong
                        logger.error("Processor did not yield think block as expected first.")
                        # We might still get a result dict next, or an exception
                        # For now, just log it.
                        if isinstance(first_yield, dict):
                            final_result = first_yield # Assume it's the result prematurely yielded

                except StopIteration:
                     logger.warning("Processor finished without yielding anything.")
                     # Handle as if no result was received below
                     pass # Will proceed to final chunk construction with final_result=None
                except Exception as proc_e:
                    logger.exception("Error occurred within the processor during first yield.")
                    final_result = {"status": "error", "message": f"Processor error: {proc_e}"} # Treat as error result

                # Handle the second yield (expecting the result dict) if no result yet
                if final_result is None:
                    try:
                        second_yield = next(processor)
                        if isinstance(second_yield, dict):
                            final_result = second_yield
                        else:
                            logger.error(f"Processor yielded unexpected type second: {type(second_yield)}")
                            final_result = {"status": "error", "message": "Processor yielded unexpected data after think block."} 
                    except StopIteration:
                        logger.warning("Processor finished after yielding think block but no result dictionary.")
                        # Handle as if no result was received
                        final_result = {"status": "error", "message": "Processor yielded thoughts but no final result."} 
                    except Exception as proc_e:
                        logger.exception("Error occurred within the processor during second yield.")
                        final_result = {"status": "error", "message": f"Processor error after thoughts: {proc_e}"}

                # --- Construct and send final message chunk based on final_result --- 
                if final_result:
                    if final_result.get("status") == "success":
                        html_link = None
                        docx_link = None
                        base_url = request.host_url.rstrip('/')
                        if final_result.get('html_filename'):
                            html_link = f"{base_url}/openai_files/html/{final_result['html_filename']}"
                        if final_result.get('docx_filename'):
                            docx_link = f"{base_url}/openai_files/docx/{final_result['docx_filename']}"
                        final_content = "PDF处理完成。"
                        if html_link: final_content += f"\nHTML 预览: {html_link}"
                        if docx_link: final_content += f"\nDOCX 文件: {docx_link}"
                        finish_reason = "stop"
                    else: # status == "error" or other
                        final_content = f"处理失败: {final_result.get('message', '未知错误')}"
                        finish_reason = "error"
                else:
                    final_content = "处理异常结束，未收到任何结果。"
                    finish_reason = "error"

                # Send the final delta chunk
                final_delta_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": final_content},
                        "finish_reason": finish_reason
                    }]
                }
                yield format_sse(final_delta_chunk)

            except Exception as e:
                # Catch errors in the stream generation logic itself
                logger.exception("Error during stream generation wrapper")
                error_content = f"流式包装处理中发生意外错误: {e}"
                error_chunk = {
                     "id": request_id,
                     "object": "chat.completion.chunk",
                     "created": created_time,
                     "model": model_name,
                     "choices": [{
                         "index": 0,
                         "delta": {"role": "assistant", "content": error_content},
                         "finish_reason": "error"
                     }]
                 }
                # Yield error chunk if possible
                try:
                    yield format_sse(error_chunk)
                except Exception as format_e:
                     logger.error(f"Failed to format/yield final error chunk: {format_e}")
            finally:
                # Always end the stream
                yield format_sse("[DONE]")

        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

    # --- Non-Streaming Response --- 
    else:
        created_time = int(time.time())
        all_think_content = ""
        final_result = None
        final_content = ""
        finish_reason = "error"

        try:
            # Consume the generator fully for non-streaming
            processor = process_pdf_openai_style(pdf_filepath=pdf_file_path)
            try:
                first_yield = next(processor)
                if isinstance(first_yield, str) and first_yield.startswith("<think>"):
                    all_think_content = first_yield
                else:
                    logger.warning("Non-streaming: First yield was not think block.")
                    if isinstance(first_yield, dict):
                         final_result = first_yield # Assume premature result
            except StopIteration:
                 logger.warning("Non-streaming: Processor yielded nothing.")
            except Exception as proc_e:
                 logger.exception("Non-streaming: Error during first yield.")
                 final_result = {"status": "error", "message": f"Processor error: {proc_e}"}

            if final_result is None: # Only get next if we haven't already got the result
                try:
                    second_yield = next(processor)
                    if isinstance(second_yield, dict):
                        final_result = second_yield
                    else:
                        logger.error(f"Non-streaming: Second yield unexpected type {type(second_yield)}")
                        final_result = {"status": "error", "message": "Processor yielded unexpected data after think block."}
                except StopIteration:
                    logger.warning("Non-streaming: Processor finished after think block, no result dict.")
                    final_result = {"status": "error", "message": "Processor yielded thoughts but no final result."}
                except Exception as proc_e:
                     logger.exception("Non-streaming: Error during second yield.")
                     final_result = {"status": "error", "message": f"Processor error after thoughts: {proc_e}"}

            # Construct final content based on final_result
            if final_result and final_result.get("status") == "success":
                html_link = None
                docx_link = None
                base_url = request.host_url.rstrip('/')
                if final_result.get('html_filename'):
                    html_link = f"{base_url}/openai_files/html/{final_result['html_filename']}"
                if final_result.get('docx_filename'):
                    docx_link = f"{base_url}/openai_files/docx/{final_result['docx_filename']}"
                final_content = "PDF处理完成。"
                if html_link: final_content += f"\nHTML 预览: {html_link}"
                if docx_link: final_content += f"\nDOCX 文件: {docx_link}"
                finish_reason = "stop"
            elif final_result:
                final_content = f"处理失败: {final_result.get('message', '未知错误')}"
                finish_reason = "error"
            else:
                 final_content = "处理异常结束，未收到任何结果。"

            response_payload = {
                 "id": request_id,
                 "object": "chat.completion",
                 "created": created_time,
                 "model": model_name,
                 "choices": [{
                     "index": 0,
                     "message": {
                         "role": "assistant",
                         "content": final_content,
                         # Optionally include thoughts in non-streaming response?
                         # "thoughts": all_think_content
                     },
                     "finish_reason": finish_reason
                 }],
                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
             }
            return jsonify(response_payload), 200

        except Exception as e:
             logger.exception("Error during non-streaming processing wrapper")
             return jsonify({"error": f"处理过程中发生意外错误: {e}"}), 500

# --- Download Endpoints --- 

@app.route('/openai_files/html/<filename>', methods=['GET'])
# No API key needed for downloads typically, but add if required
def download_html_file(filename):
    """Downloads a generated HTML preview file."""
    # Basic security check
    if '..' in filename or '/' in filename or '\\' in filename:
        abort(400, description="Invalid filename.")
    try:
        # Use the directory path returned by the processor
        # We assume PROCESSED_PREVIEW_DIR is correctly configured in openai_processor.py
        return send_from_directory(PROCESSED_PREVIEW_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404, description="HTML file not found.")
    except Exception as e:
        logger.error(f"Error sending HTML file {filename}: {e}")
        abort(500, description="Could not send HTML file.")

@app.route('/openai_files/docx/<filename>', methods=['GET'])
def download_docx_file(filename):
    """Downloads a generated DOCX file."""
    # Basic security check
    if '..' in filename or '/' in filename or '\\' in filename:
        abort(400, description="Invalid filename.")
    try:
        # Use the directory path configured in openai_processor.py
        return send_from_directory(OPENAI_DOCX_OUTPUT_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404, description="DOCX file not found.")
    except Exception as e:
        logger.error(f"Error sending DOCX file {filename}: {e}")
        abort(500, description="Could not send DOCX file.")

if __name__ == '__main__':
    # Ensure the app runs on a different port
    port = 7860
    print(f"INFO: OpenAI Compatible OLMOCR API starting...")
    print(f"INFO: Using API Key: {API_KEY[:5]}...{API_KEY[-4:] if len(API_KEY) > 9 else ''}")
    print(f"INFO: Listening on http://0.0.0.0:{port}")
    print(f"INFO: HTML files served from: {PROCESSED_PREVIEW_DIR}")
    print(f"INFO: DOCX files served from: {OPENAI_DOCX_OUTPUT_DIR}")
    # Use debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False) 