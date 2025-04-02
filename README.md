# olmocr-dify-adapter

这是一个将olmocr与Dify平台集成的适配器，使用Flask提供符合OpenAPI标准格式的工具API。

## 功能

- 提供符合OpenAI schema格式的API接口
- 接收Dify的文件对象输入 (优先) 或 Base64编码文件输入
- 将PDF文件转换为文本内容并返回结果 (结果嵌套在 `analysis_output` 键下)
- 支持单个PDF和批量处理多个PDF
- 可以作为Dify平台的工具调用
- 支持任务状态检查，避免并发处理多个任务
- 支持配置自定义模型路径和Dify基础URL

## 安装

1. 确保已安装olmocr及其依赖项

```bash
# 安装olmocr
git clone https://github.com/allenai/olmocr.git
cd olmocr
pip install -e .[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

2. 安装适配器依赖

```bash
pip install -r requirements.txt
```

## 配置

可以通过环境变量或直接修改`app.py`来配置：

- `MODEL_PATH`: olmocr模型的本地路径 (默认: `/mnt/model_Q4/olmocr`)
- `DIFY_BASE_URL`: 你的Dify服务的基础URL，用于下载文件 (默认: `http://localhost:3000`)

## 使用方法

1. 启动适配器服务

```bash
# 可选: 设置环境变量
# export MODEL_PATH=/path/to/your/model
# export DIFY_BASE_URL=http://your-dify-domain.com

python app.py
```

或使用gunicorn（生产环境推荐）:

```bash
gunicorn -w 1 -b 0.0.0.0:5555 app:app
```

2. 在Dify平台中配置工具

在Dify平台上，添加一个新的工具，配置如下:

- API URL: `http://your-adapter-server:5555/analyze_pdf` 或 `http://your-adapter-server:5555/analyze_multiple_pdfs`
- 使用tools接口获取schema: `http://your-adapter-server:5555/tools`
- **在工具参数中:**
    - 如果主要使用Dify文件上传，将文件输入变量类型设置为`File` (文件)
    - 如果希望也能通过Base64传递，可以在Dify的工具描述中说明接受`pdf_base64`和`filename` (或`pdf_list`)，但Dify界面本身可能不直接支持非文件类型的复杂输入构造。

## API接口

### 获取工具定义

```
GET /tools
```

返回符合OpenAPI格式的工具schema定义 (注意: schema使用`oneOf`表示可选输入)。

### 分析单个PDF文件

**方式一: 使用Dify文件对象 (推荐)**

```
POST /analyze_pdf
Content-Type: application/json

{
  "files": [
    { // Dify提供的文件对象
      "url": "/files/.../file-preview?timestamp=...",
      "filename": "文件名.pdf",
      "type": "document",
      "extension": ".pdf",
      "size": 12345,
      "mime_type": "application/pdf"
    }
  ]
}
```

**方式二: 使用Base64编码**

```
POST /analyze_pdf
Content-Type: application/json

{
  "pdf_base64": "JVBERi0xLjQKJ....",
  "filename": "来自Base64.pdf"
}
```

返回 (两种方式均返回此结构):

```json
{
  "analysis_output": {
    "content": "从PDF中提取的文本内容",
    "filename": "处理的文件名",
    "status": "success",
    "processing_time": 15.2
  }
}
```

### 批量分析多个PDF文件

**方式一: 使用Dify文件对象列表 (推荐)**

```
POST /analyze_multiple_pdfs
Content-Type: application/json

{
  "files": [
    { "url": "/files/...", "filename": "文件1.pdf", ... },
    { "url": "/files/...", "filename": "文件2.pdf", ... }
  ]
}
```

**方式二: 使用Base64对象列表**

```
POST /analyze_multiple_pdfs
Content-Type: application/json

{
  "pdf_list": [
    {
      "pdf_base64": "第一个PDF的Base64...",
      "filename": "文件A.pdf"
    },
    {
      "pdf_base64": "第二个PDF的Base64...",
      "filename": "文件B.pdf"
    }
  ]
}
```

返回 (两种方式均返回此结构):

```json
{
  "analysis_output": {
    "results": {
      "文件1.pdf": {
        "content": "...",
        "filename": "文件1.pdf",
        "status": "success"
      },
      "文件2.pdf": {
        "content": "...",
        "filename": "文件2.pdf",
        "status": "success"
      }
    },
    "total_files_requested": 2,
    "total_files_prepared": 2,
    "processed_files": 2,
    "status": "success", // 或 partial_success, failure
    "processing_time": 30.5
  }
}
```

### 获取当前处理状态

```
GET /status
```

返回 (结构不变, 但task_info字段可能包含更多细节):

```json
{
  "is_processing": true,
  "status": "busy",
  "task_info": {
    "type": "multiple_pdfs",
    "start_time": ..., 
    "status": "downloading_pdfs",
    "input_source": "dify_files",
    "total_files": 2,
    "processed_files_download": 1, 
    "current_file": "文件1.pdf"
  }
}
```

## 注意事项

- API优先使用`files`输入。如果`files`不存在或无效，才会尝试`pdf_base64`/`pdf_list`。
- 确保配置正确的`DIFY_BASE_URL`以便`files`输入能正常工作。
- 服务一次只能处理一个任务，若任务正在处理中，新请求将返回429状态码。
- 服务需要足够的磁盘空间来存储临时PDF文件。
- 需要GPU环境以获得最佳性能。
- 大型PDF文件可能需要较长处理时间 