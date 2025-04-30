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
    if not data or 'texts' not in data:
        return jsonify({'error': '请提供texts参数（字符串数组）'}), 400
    
    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({'error': 'texts参数必须是字符串数组'}), 400
    if not texts:
        return jsonify({'error': 'texts数组不能为空'}), 400
    if not all(isinstance(text, str) for text in texts):
        return jsonify({'error': 'texts数组的所有元素必须是字符串'}), 400
    
    return_dense = data.get('return_dense', True)
    return_sparse = data.get('return_sparse', True)
    
    # 使用encode方法进行文本编码（获取稠密向量和稀疏向量）
    output = model.encode(texts, return_dense=return_dense, return_sparse=return_sparse, convert_to_numpy=True)

    # 构建响应
    response = {
        'model_info': {
            'name': 'BAAI/bge-m3',
            'type': 'hybrid_embedding'
        }
    }
    
    # 处理稠密向量
    if return_dense and 'dense_vecs' in output:
        dense_embeddings = []
        for vec in output['dense_vecs']:
            dense_embeddings.append(vec.tolist() if hasattr(vec, 'tolist') else vec)
        response['dense_embeddings'] = dense_embeddings
    
    # 处理稀疏向量
    if return_sparse and 'lexical_weights' in output:
        sparse_embeddings = []
        for weights in output['lexical_weights']:
            sparse_embedding = {}
            # 获取原始token_id和权重
            sparse_raw = {}
            for token_id, weight in weights.items():
                sparse_raw[str(token_id)] = float(weight)
            
            # 获取可读的token和权重
            sparse_readable = {}
            for token_id, weight in weights.items():
                token = tokenizer.convert_ids_to_tokens(int(token_id))
                sparse_readable[token] = float(weight)
            
            sparse_embedding = {
                'raw': sparse_raw,
                'readable': sparse_readable
            }
            sparse_embeddings.append(sparse_embedding)
        response['sparse_embeddings'] = sparse_embeddings
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    freeze_support()  # 支持多进程
    app.run(host='0.0.0.0', port=8082, debug=True)