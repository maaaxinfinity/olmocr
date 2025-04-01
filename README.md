# olmocr-dify-adapter

这是一个将olmocr与Dify平台集成的适配器，使用Flask提供符合OpenAI标准格式的工具API。

## 功能

- 提供符合OpenAI schema格式的API接口
- 将PDF文件转换为文本内容并返回结果
- 支持单个PDF和批量处理多个PDF
- 可以作为Dify平台的工具调用
- 支持任务状态检查，避免并发处理多个任务
- 支持配置自定义模型路径

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

在app.py中可以设置模型路径：

```python
# 指定模型路径
MODEL_PATH = "/mnt/model_Q4/olmocr"
```

## 使用方法

1. 启动适配器服务

```bash
python app.py
```

或使用gunicorn（生产环境推荐）:

```bash
gunicorn -w 1 -b 0.0.0.0:5555 app:app
```

2. 在Dify平台中配置工具

在Dify平台上，添加一个新的工具，配置如下:

- API URL: `http://your-server:5555/analyze_pdf` 或 `http://your-server:5555/analyze_multiple_pdfs`
- 使用tools接口获取schema: `http://your-server:5555/tools`

## API接口

### 获取工具定义

```
GET /tools
```

返回符合OpenAI格式的工具schema定义。

### 分析单个PDF文件

```
POST /analyze_pdf
Content-Type: application/json

{
  "pdf_base64": "base64编码的PDF内容",
  "filename": "文件名.pdf"
}
```

返回:

```json
{
  "content": "从PDF中提取的文本内容",
  "filename": "处理的文件名",
  "status": "success",
  "processing_time": 15.2
}
```

### 批量分析多个PDF文件

```
POST /analyze_multiple_pdfs
Content-Type: application/json

{
  "pdf_files": [
    {
      "pdf_base64": "第一个PDF的base64编码内容",
      "filename": "文件1.pdf"
    },
    {
      "pdf_base64": "第二个PDF的base64编码内容",
      "filename": "文件2.pdf"
    }
  ]
}
```

返回:

```json
{
  "results": {
    "文件1.pdf": {
      "content": "从PDF中提取的文本内容",
      "filename": "文件1.pdf"
    },
    "文件2.pdf": {
      "content": "从PDF中提取的文本内容",
      "filename": "文件2.pdf"
    }
  },
  "total_files": 2,
  "processed_files": 2,
  "status": "success",
  "processing_time": 30.5
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

或当正在处理任务时:

```json
{
  "is_processing": true,
  "status": "busy",
  "task_info": {
    "type": "multiple_pdfs",
    "start_time": 1623456789.123,
    "status": "processing_pdfs",
    "total_files": 2,
    "processed_files": 1,
    "current_file": "文件2.pdf"
  }
}
```

### 获取处理结果文件

```
GET /results/{filename}
```

直接下载指定的结果文件。

## 注意事项

- 服务一次只能处理一个任务，若任务正在处理中（不论是单个PDF还是多个PDF批量处理），新请求将返回429状态码
- 服务需要足够的磁盘空间来存储临时PDF文件
- 需要GPU环境以获得最佳性能
- 大型PDF文件可能需要较长处理时间 