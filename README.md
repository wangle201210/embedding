# 文本向量化HTTP服务

这是一个基于Flask的HTTP服务，用于将文本转换为稠密向量和稀疏向量。该服务使用BAAI/bge-m3模型进行向量化处理。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python main.py
```

服务将在 http://localhost:5000 上运行。

## API接口

### 1. 文本向量化

- **URL**: `/encode`
- **方法**: POST
- **请求体**:
  ```json
  {
    "text": "需要向量化的文本"
  }
  ```
- **响应**:
  ```json
  {
    "dense_embedding": [...],  // 稠密向量
    "sparse_embedding": {
      "raw": {"token_id": weight, ...},  // 原始token_id和权重
      "readable": {"token": weight, ...}  // 可读的token和权重
    }
  }
  ```

### 2. 健康检查

- **URL**: `/health`
- **方法**: GET
- **响应**:
  ```json
  {
    "status": "ok"
  }
  ```

## 注意事项

- 确保已登录HuggingFace，以便能够下载和使用BAAI/bge-m3模型
- 对于Mac用户，已默认设置`use_fp16=False`