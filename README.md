# olmocr-dify-adapter

这是一个将olmocr与Dify平台集成的适配器，使用Flask提供符合OpenAPI标准格式的工具API。

## 功能

- 提供符合OpenAPI schema格式的API接口
- 接收Dify的文件对象输入 (键名为在Dify中定义的变量名, 如`DOC`) 或 Base64编码文件输入
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
- **Dify输入键名**: 在`app.py`中修改`dify_input_key`变量 (当前设置为`'DOC'`)，使其与你在Dify工具中定义的File类型变量名一致。

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
    - **定义一个`File`类型的变量 (例如，命名为`DOC`)**。确保这个名称与`app.py`中的`dify_input_key`变量值匹配。
    - 如果希望也能通过Base64传递，可以在Dify的工具描述中说明接受`pdf_base64`和`filename` (或`pdf_list`)。

## API接口

### 获取工具定义

```
GET /tools
```

返回符合OpenAPI格式的工具schema定义 (注意: schema使用`oneOf`表示可选输入，且目前未完全反映`dify_input_key`的动态性)。

### 分析单个PDF文件

**方式一: 使用Dify文件对象 (推荐)**

```
POST /analyze_pdf
Content-Type: application/json

{
  "DOC": [ // <-- 键名与app.py中的dify_input_key匹配
    { // Dify提供的文件对象
      "url": "/files/.../file-preview?timestamp=...",
      "filename": "文件名.pdf",
      ...
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
  "DOC": [ // <-- 键名与app.py中的dify_input_key匹配
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
    { "pdf_base64": "...", "filename": "文件A.pdf" },
    { "pdf_base64": "...", "filename": "文件B.pdf" }
  ]
}
```

返回 (两种方式均返回此结构):

```json
{
  "analysis_output": {
    "results": {
      "文件1.pdf": { ... },
      "文件2.pdf": { ... }
    },
    "total_files_requested": 2,
    "total_files_prepared": 2,
    "processed_files": 2,
    "status": "success", 
    "processing_time": 30.5
  }
}
```

### 获取当前处理状态

```
GET /status
```

返回:

```json
{
  "is_processing": false,
  "status": "ready",
  "task_info": null
}
```

## 注意事项

- **确保`app.py`中的`dify_input_key`与你在Dify工具中定义的File变量名一致 (当前设为`'DOC'`)。**
- API优先使用Dify变量键 (如`DOC`) 输入。如果该键不存在或无效，才会尝试`pdf_base64`/`pdf_list`。
- 确保配置正确的`DIFY_BASE_URL`以便Dify文件输入能正常工作。
- 服务一次只能处理一个任务，若任务正在处理中，新请求将返回429状态码。
- 服务需要足够的磁盘空间来存储临时PDF文件。
- 需要GPU环境以获得最佳性能。
- 大型PDF文件可能需要较长处理时间 