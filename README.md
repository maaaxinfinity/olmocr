# olmocr-dify-adapter

这是一个将olmocr与Dify平台集成的适配器，使用Flask提供符合OpenAPI标准格式的工具API。

**重要：此版本假设Dify工作流通过Jinja2模板传入文件URL和文件名作为查询参数。**

## 功能

- 提供符合OpenAPI schema格式的API接口 (`GET /analyze_pdf`)
- 通过URL下载Dify文件，并使用传入的文件名进行处理
- 将PDF文件转换为文本内容并返回结果 (结果嵌套在 `analysis_output` 键下)
- 一次API调用处理一个文件 (需要在Dify中自行实现迭代)
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

- API URL: `http://your-adapter-server:5555/analyze_pdf`
- 使用tools接口获取schema: `http://your-adapter-server:5555/tools` (此Schema现在只包含 `/analyze_pdf` 且使用GET参数)
- **在工具参数中:**
    - Dify会根据Schema自动生成两个**字符串类型**的输入变量：`file_url` 和 `filename`。
    - **使用Jinja2模板**将您的上下文文件变量（假设名为`DOC`）拆分，并将URL和文件名分别赋值给这两个输入变量：
        - `file_url` 输入框: `{{ DOC.files[0].url }}` (假设只处理第一个文件)
        - `filename` 输入框: `{{ DOC.files[0].filename }}` (假设只处理第一个文件)
    - **如果需要处理多个文件，您需要在Dify工作流中构建循环或迭代逻辑，每次调用工具只处理一个文件。**

## API接口

### 获取工具定义

```
GET /tools
```

返回符合OpenAPI格式的工具schema定义 (现在只包含 `GET /analyze_pdf` 端点)。

### 分析单个PDF文件 (通过GET请求)

```
GET /analyze_pdf?file_url=<URL编码后的相对路径>&filename=<URL编码后的文件名>
```

**示例调用 (由Dify通过Jinja2模板生成):**

```
GET http://your-adapter-server:5555/analyze_pdf?file_url=%2Ffiles%2Fa7ab9415-caea-4f26-bd1d-efd8cefbadd5%2Ffile-preview%3Ftimestamp%3D...%26nonce%3D...%26sign%3D...&filename=1.%E8%AF%81%E6%8D%AE%E6%9D%90%E6%96%99%E5%8D%B7%EF%BC%88%E7%AC%AC%E4%B8%80%E5%86%8C%EF%BC%89%EF%BC%88%E5%85%AC%E5%AE%89%E5%8D%B7%EF%BC%89_可搜索.pdf
```

返回:

```json
{
  "analysis_output": {
    "filename": "1.证据材料卷（第一册）（公安卷）_可搜索.pdf", // 传入的文件名
    "content": "从PDF中提取的文本内容",
    "status": "success", // 或 "failed"
    "error": null, // 或 错误信息字符串
    "processing_time": 15.2
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
      "type": "single_pdf_analysis_get",
      "start_time": ..., 
      "status": "processing_pdf",
      "input_filename": "1.证据材料卷（第一册）（公安卷）_可搜索.pdf",
      "input_url": "/files/..."
    }
}
```

## 注意事项

- **API端点已改为 `GET /analyze_pdf`，并通过查询参数接收`file_url`和`filename`。**
- **您必须在Dify工作流中使用Jinja2模板从文件变量中提取URL和文件名，并分别传入对应的输入参数。**
- **此API现在一次只处理一个文件。处理多个文件需要在Dify中实现循环调用。**
- 确保配置正确的`DIFY_BASE_URL`。
- 服务一次只能处理一个任务，若任务正在处理中，新请求将返回429状态码。
- 服务需要足够的磁盘空间来存储临时PDF文件。
- 需要GPU环境以获得最佳性能。
- 大型PDF文件可能需要较长处理时间。 