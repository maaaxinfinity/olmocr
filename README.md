# olmocr-dify-adapter

这是一个将olmocr与Dify平台集成的适配器，使用Flask提供符合OpenAPI标准格式的工具API。

## 功能

- 提供符合OpenAPI schema格式的API接口
- **统一使用 `/analyze_pdf` 端点处理单个或多个文件**
- 接收Dify的文件对象输入 (固定键名为`input_files`) 或 Base64编码文件输入 (Base64为备选)
- **顺序处理** 多个文件 (一次处理一个文件，非并行)
- 将PDF文件转换为文本内容并返回结果 (结果嵌套在 `analysis_output` 键下，包含每个文件的状态)
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
- **Dify输入键名**: 在`app.py`中，文件输入的固定键名被设置为`input_files`。请确保在Dify工具配置中也使用此名称。

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

- API URL: `http://your-adapter-server:5555/analyze_pdf` (**使用这个统一端点**)
- 使用tools接口获取schema: `http://your-adapter-server:5555/tools` (此Schema现在只包含 `/analyze_pdf`)
- **在工具参数中:**
    - **定义一个`File`类型的变量，并将其命名为 `input_files`**。这是必需的，以便Dify将文件信息放入正确的JSON键中。
    - **配置该变量允许上传多个文件 (设置为列表/数组类型)。**
    - 如果希望也能通过Base64传递，可以在Dify的工具描述中说明接受`pdf_list`。

## API接口

### 获取工具定义

```
GET /tools
```

返回符合OpenAPI格式的工具schema定义 (现在只包含 `/analyze_pdf` 端点)。

### 分析一个或多个PDF文件

**方式一: 使用Dify文件对象 (推荐)**

```
POST /analyze_pdf
Content-Type: application/json

// 处理单个文件
{
  "input_files": [
    { "url": "/files/...", "filename": "文件名.pdf", ... }
  ]
}

// 处理多个文件
{
  "input_files": [
    { "url": "/files/...", "filename": "文件1.pdf", ... },
    { "url": "/files/...", "filename": "文件2.pdf", ... }
  ]
}
```

**方式二: 使用Base64对象列表**

```
POST /analyze_pdf
Content-Type: application/json

{
  "pdf_list": [
    { "pdf_base64": "...", "filename": "文件A.pdf" },
    { "pdf_base64": "...", "filename": "文件B.pdf" }
  ]
}
```

返回 (所有情况均返回此结构):

```json
{
  "analysis_output": {
    "results": {
      "文件名.pdf": {
        "content": "...",
        "filename": "文件名.pdf",
        "status": "success", // 或 "failed"
        "error": null // 或 错误信息字符串
      },
      "文件1.pdf": { ... },
      "文件2.pdf": { ... }
    },
    "total_files_requested": 2,
    "total_files_prepared": 2,
    "processed_successfully": 2,
    "status": "success", // 或 partial_success, failure
    "processing_time": 45.8
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

或者当任务进行中:
```json
{
  "is_processing": true,
  "status": "busy",
  "task_info": {
      "type": "sequential_pdf_analysis",
      "start_time": ..., 
      "status": "processing_file_2_of_3", // 任务状态更详细
      "input_source": "dify_input_files",
      "total_files_requested": 3, 
      "total_files_to_process": 3, 
      "current_file_processing": "报告B.pdf" 
    }
}
```

## 注意事项

- **API端点已合并为 `/analyze_pdf`**。
- **确保在Dify工具配置中，文件输入变量的名称是 `input_files` 并允许列表/数组。**
- API优先使用`input_files`键输入。如果该键不存在或无效，才会尝试`pdf_list`。
- 确保配置正确的`DIFY_BASE_URL`以便`input_files`输入能正常工作。
- 服务一次只能处理一个任务 (顺序处理文件)，若任务正在处理中，新请求将返回429状态码。
- 服务需要足够的磁盘空间来存储临时PDF文件。
- 需要GPU环境以获得最佳性能。
- 大型PDF文件可能需要较长处理时间 