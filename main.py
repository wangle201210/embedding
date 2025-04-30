from flask import Flask, request, jsonify
from FlagEmbedding import BGEM3FlagModel
from multiprocessing import freeze_support

app = Flask(__name__)

# 初始化模型（确保已登录HuggingFace）
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)  # Mac用户去掉use_fp16
tokenizer = model.tokenizer

@app.route('/encode', methods=['POST'])
def encode_text():
    # 获取请求数据
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '请提供文本参数'}), 400
    
    text = data['text']
    
    # 编码文本（获取稠密向量和稀疏向量）
    output = model.encode(text, return_dense=True, return_sparse=True, convert_to_numpy=True)

    # 处理输出结果
    # 处理稠密向量
    dense_embedding = output['dense_vecs'].tolist() if hasattr(output['dense_vecs'], 'tolist') else output['dense_vecs']
    
    # 处理稀疏向量
    sparse_embedding = {}
    if 'lexical_weights' in output:
        # 获取原始token_id和权重
        sparse_raw = {}
        for token_id, weight in output['lexical_weights'].items():
            sparse_raw[str(token_id)] = float(weight)
        
        # 获取可读的token和权重
        sparse_readable = {}
        for token_id, weight in output['lexical_weights'].items():
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            sparse_readable[token] = float(weight)
        
        sparse_embedding = {
            'raw': sparse_raw,
            'readable': sparse_readable
        }
    
    # 构建响应
    response = {
        'dense_embedding': dense_embedding,  # 转换为列表以便JSON序列化
        'sparse_embedding': sparse_embedding,
        'model_info': {
            'name': 'BAAI/bge-m3',
            'type': 'hybrid_embedding'
        }
    }
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    freeze_support()  # 支持多进程
    app.run(host='0.0.0.0', port=8082, debug=True)