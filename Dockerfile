FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip git poppler-utils ttf-mscorefonts-installer \
    msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito \
    gsfonts lcdf-typetools wget \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 安装olmocr
RUN git clone https://github.com/allenai/olmocr.git /app/olmocr
WORKDIR /app/olmocr
RUN pip3 install -e .[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# 安装适配器
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 复制应用代码
COPY app.py .
COPY dify-tool-schema.json .

# 创建工作目录
RUN mkdir -p /app/localworkspace

# 暴露端口
EXPOSE 5555

# 启动应用 - 使用单线程模式避免并发处理
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5555", "app:app"] 