from flask import Flask, request, jsonify, send_file
import os
import tempfile
import subprocess
import json
import base64
from werkzeug.utils import secure_filename
import shutil
import time
import threading
import glob

app = Flask(__name__)

# 创建工作目录
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localworkspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 指定模型路径
MODEL_PATH = "/mnt/model_Q4/olmocr"

# 任务状态标志和锁
current_task = None
task_lock = threading.Lock()

# 定义工具的OpenAI格式schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_pdf",
            "description": "使用olmocr分析单个PDF文件并提取内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_base64": {
                        "type": "string",
                        "description": "PDF文件的base64编码内容"
                    },
                    "filename": {
                        "type": "string",
                        "description": "PDF文件名称，包括.pdf扩展名"
                    }
                },
                "required": ["pdf_base64", "filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_multiple_pdfs",
            "description": "使用olmocr分析多个PDF文件并提取内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_files": {
                        "type": "array",
                        "description": "PDF文件的列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "pdf_base64": {
                                    "type": "string",
                                    "description": "PDF文件的base64编码内容"
                                },
                                "filename": {
                                    "type": "string",
                                    "description": "PDF文件名称，包括.pdf扩展名"
                                }
                            },
                            "required": ["pdf_base64", "filename"]
                        }
                    }
                },
                "required": ["pdf_files"]
            }
        }
    }
]

@app.route("/tools", methods=["GET"])
def get_tools():
    """返回工具的schema定义"""
    return jsonify(TOOLS_SCHEMA)

@app.route("/status", methods=["GET"])
def get_status():
    """返回当前处理状态"""
    global current_task
    with task_lock:
        if current_task:
            return jsonify({
                "is_processing": True,
                "status": "busy",
                "task_info": current_task
            })
        else:
            return jsonify({
                "is_processing": False,
                "status": "ready",
                "task_info": None
            })

@app.route("/analyze_pdf", methods=["POST"])
def analyze_pdf():
    """处理单个PDF分析请求"""
    global current_task
    
    # 检查是否有任务正在处理
    with task_lock:
        if current_task:
            return jsonify({
                "error": "服务器忙，当前有任务正在处理中",
                "status": "busy",
                "task_info": current_task
            }), 429  # 429 Too Many Requests
        
        # 设置任务信息
        current_task = {
            "type": "single_pdf",
            "start_time": time.time(),
            "status": "processing"
        }
    
    try:
        data = request.json
        
        if not data or 'pdf_base64' not in data or 'filename' not in data:
            with task_lock:
                current_task = None
            return jsonify({"error": "缺少必要参数"}), 400
        
        # 解码base64并保存PDF文件
        pdf_data = base64.b64decode(data['pdf_base64'])
        filename = secure_filename(data['filename'])
        
        # 确保文件名以.pdf结尾
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        # 更新任务信息
        with task_lock:
            current_task["filename"] = filename
        
        pdf_path = os.path.join(WORKSPACE_DIR, filename)
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_data)
        
        # 调用olmocr处理PDF
        output_path = os.path.join(WORKSPACE_DIR, "results")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # 执行olmocr处理命令（类似CLI）
        cmd = [
            "python", "-m", "olmocr.pipeline", 
            WORKSPACE_DIR, 
            "--pdfs", pdf_path,
            "--model", MODEL_PATH
        ]
        
        # 更新任务状态
        with task_lock:
            current_task["command"] = " ".join(cmd)
            current_task["status"] = "processing_pdf"
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            with task_lock:
                current_task = None
            return jsonify({
                "error": "PDF处理失败",
                "details": process.stderr
            }), 500
        
        # 从结果目录读取处理后的JSON文件
        result_files = [f for f in os.listdir(output_path) if f.startswith("output_") and f.endswith(".jsonl")]
        
        if not result_files:
            with task_lock:
                current_task = None
            return jsonify({"error": "未找到处理结果"}), 500
        
        # 读取结果文件
        result_path = os.path.join(output_path, result_files[0])
        with open(result_path, 'r', encoding='utf-8') as f:
            result_content = f.read().strip()
            
        # 解析JSONL格式的结果
        results = []
        for line in result_content.split('\n'):
            if line.strip():
                results.append(json.loads(line))
        
        # 提取文本内容
        extracted_text = ""
        if results:
            for result in results:
                if 'text' in result:
                    extracted_text += result['text'] + "\n\n"
        
        # 更新任务信息并生成响应
        response_data = {
            "content": extracted_text.strip(),
            "filename": filename,
            "status": "success",
            "processing_time": time.time() - current_task["start_time"]
        }
        
        # 重置任务状态
        with task_lock:
            current_task = None
            
        return jsonify(response_data)
        
    except Exception as e:
        # 确保在发生异常时也会重置任务状态
        with task_lock:
            current_task = None
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_multiple_pdfs", methods=["POST"])
def analyze_multiple_pdfs():
    """处理多个PDF分析请求"""
    global current_task
    
    # 检查是否有任务正在处理
    with task_lock:
        if current_task:
            return jsonify({
                "error": "服务器忙，当前有任务正在处理中",
                "status": "busy",
                "task_info": current_task
            }), 429  # 429 Too Many Requests
        
        # 设置任务信息
        current_task = {
            "type": "multiple_pdfs",
            "start_time": time.time(),
            "status": "initializing"
        }
    
    try:
        data = request.json
        
        if not data or 'pdf_files' not in data or not isinstance(data['pdf_files'], list) or len(data['pdf_files']) == 0:
            with task_lock:
                current_task = None
            return jsonify({"error": "缺少必要参数或PDF文件列表为空"}), 400
        
        # 创建临时目录存放PDF文件
        pdf_dir = os.path.join(WORKSPACE_DIR, "pdfs")
        if os.path.exists(pdf_dir):
            shutil.rmtree(pdf_dir)
        os.makedirs(pdf_dir, exist_ok=True)
        
        # 保存所有PDF文件
        pdf_files = []
        with task_lock:
            current_task["total_files"] = len(data['pdf_files'])
            current_task["processed_files"] = 0
            current_task["status"] = "saving_pdfs"
        
        for idx, pdf_file in enumerate(data['pdf_files']):
            if 'pdf_base64' not in pdf_file or 'filename' not in pdf_file:
                continue
                
            # 解码base64并保存PDF
            pdf_data = base64.b64decode(pdf_file['pdf_base64'])
            filename = secure_filename(pdf_file['filename'])
            
            # 确保文件名以.pdf结尾
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
                
            pdf_path = os.path.join(pdf_dir, filename)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)
                
            pdf_files.append(pdf_path)
            
            # 更新任务状态
            with task_lock:
                current_task["current_file"] = filename
                current_task["processed_files"] = idx + 1
        
        # 调用olmocr处理所有PDF
        output_path = os.path.join(WORKSPACE_DIR, "results")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # 准备命令 (类似CLI的batch模式)
        cmd = [
            "python", "-m", "olmocr.pipeline", 
            WORKSPACE_DIR, 
            "--pdfs", f"{pdf_dir}/*.pdf",
            "--model", MODEL_PATH
        ]
        
        # 更新任务状态
        with task_lock:
            current_task["command"] = " ".join(cmd)
            current_task["status"] = "processing_pdfs"
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            with task_lock:
                current_task = None
            return jsonify({
                "error": "批量PDF处理失败",
                "details": process.stderr
            }), 500
        
        # 从结果目录读取处理后的所有JSON文件
        result_files = [f for f in os.listdir(output_path) if f.startswith("output_") and f.endswith(".jsonl")]
        
        if not result_files:
            with task_lock:
                current_task = None
            return jsonify({"error": "未找到处理结果"}), 500
        
        # 处理所有结果文件
        results_by_file = {}
        
        for result_file in result_files:
            result_path = os.path.join(output_path, result_file)
            # 从文件名推测原始PDF文件名
            pdf_name = result_file.replace("output_", "").replace(".jsonl", ".pdf")
            
            with open(result_path, 'r', encoding='utf-8') as f:
                result_content = f.read().strip()
                
            # 解析JSONL格式的结果
            file_results = []
            for line in result_content.split('\n'):
                if line.strip():
                    file_results.append(json.loads(line))
            
            # 提取文本内容
            extracted_text = ""
            if file_results:
                for result in file_results:
                    if 'text' in result:
                        extracted_text += result['text'] + "\n\n"
            
            results_by_file[pdf_name] = {
                "content": extracted_text.strip(),
                "filename": pdf_name
            }
        
        # 更新任务信息并生成响应
        response_data = {
            "results": results_by_file,
            "total_files": len(data['pdf_files']),
            "processed_files": len(results_by_file),
            "status": "success",
            "processing_time": time.time() - current_task["start_time"]
        }
        
        # 重置任务状态
        with task_lock:
            current_task = None
            
        return jsonify(response_data)
        
    except Exception as e:
        # 确保在发生异常时也会重置任务状态
        with task_lock:
            current_task = None
        return jsonify({"error": str(e)}), 500

@app.route("/results/<path:filename>", methods=["GET"])
def get_result_file(filename):
    """获取结果文件"""
    result_path = os.path.join(WORKSPACE_DIR, "results", secure_filename(filename))
    if os.path.exists(result_path):
        return send_file(result_path)
    else:
        return jsonify({"error": "文件不存在"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True) 