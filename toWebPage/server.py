# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch import nn
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from transformers import BertModel, BertLMHeadModel, BertTokenizer
import base64
import io
import unicodedata

# --- 모델 클래스와 유틸리티 함수들을 서버 파일에 통합 ---
class ArtistX(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)

class ArtistY(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)

def combine_jamo(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def encode_data_to_image(tensor: torch.Tensor, original_length: int, target_size=(1280, 1280)) -> Image.Image:
    tensor_np = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    tensor_bytes = tensor_np.tobytes()
    length_bytes = original_length.to_bytes(4, byteorder='big')
    combined_bytes = length_bytes + tensor_bytes
    image_array = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    byte_array = np.frombuffer(combined_bytes, dtype=np.uint8)
    image_array.flat[:len(byte_array)] = byte_array
    
    tensor_np_norm = tensor_np - tensor_np.min()
    tensor_np_norm = (tensor_np_norm / tensor_np_norm.max() * 255).astype(np.uint8)
    vis_img = Image.fromarray(tensor_np_norm).resize((256, 256), Image.Resampling.BICUBIC)
    vis_array = np.array(vis_img)

    block_size = 256 // 8
    picasso_canvas = np.zeros_like(vis_array)
    for r in range(8):
        for c in range(8):
            block = vis_array[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
            rand_choice = np.random.randint(0, 4)
            if rand_choice == 1: block = np.flipud(block)
            elif rand_choice == 2: block = np.fliplr(block)
            elif rand_choice == 3: block = np.rot90(block)
            picasso_canvas[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size] = block

    picasso_colors = ['#2E4A34', '#523A62', '#F2D338', '#C82D31', '#D9D9D9', '#343434']
    custom_cmap = ListedColormap(picasso_colors)
    normalized_canvas = (picasso_canvas - picasso_canvas.min()) / (picasso_canvas.max() - picasso_canvas.min())
    colored_array = (custom_cmap(normalized_canvas)[:, :, :3] * 255).astype(np.uint8)
    final_visual_img = Image.fromarray(colored_array)

    draw = ImageDraw.Draw(final_visual_img)
    for i in range(1, 8):
        draw.line([(i * block_size, 0), (i * block_size, 255)], fill='black', width=2)
        draw.line([(0, i * block_size), (255, i * block_size)], fill='black', width=2)
    
    vis_w, vis_h = final_visual_img.size
    start_x = (target_size[0] - vis_w) // 2
    start_y = (target_size[1] - vis_h) // 2
    image_array[start_y:start_y+vis_h, start_x:start_x+vis_w, :3] = np.array(final_visual_img)
    image_array[start_y:start_y+vis_h, start_x:start_x+vis_w, 3] = 255
    
    return Image.fromarray(image_array, 'RGBA')

def decode_data_from_image(image: Image.Image, original_shape=(128, 768)) -> (torch.Tensor, int):
    image_array = np.array(image)
    image_bytes = image_array.tobytes()
    length_bytes = image_bytes[:4]
    original_length = int.from_bytes(length_bytes, byteorder='big')
    required_bytes = original_shape[0] * original_shape[1] * 4
    tensor_bytes = image_bytes[4 : 4 + required_bytes]
    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(original_shape)
    tensor = torch.from_numpy(tensor_np).unsqueeze(0)
    return tensor, original_length

# --- Flask 앱 설정 ---
app = Flask(__name__)
CORS(app) # 웹페이지(다른 출처)에서의 API 요청을 허용

# --- 모델 로딩 ---
print("서버 시작 중... 모델을 불러옵니다.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_x = ArtistX()
model_x.load_state_dict(torch.load("artist_x_best_model.pth", map_location=device))
model_x.to(device)
model_x.eval()

model_y = ArtistY()
model_y.load_state_dict(torch.load("artist_y_standalone.pth", map_location=device))
model_y.to(device)
model_y.eval()
print(f"✅ 모델 로딩 완료. (Device: {str(device).upper()})")

# --- API 엔드포인트 정의 ---
@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    text = data.get('text')
    mode = data.get('mode', 'y') # 기본값은 Y
    
    if not text:
        return jsonify({"error": "텍스트가 없습니다."}), 400

    model = model_y if mode == 'y' else model_x
    
    with torch.no_grad():
        inputs = model.tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        latent_vector = model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        original_length = torch.sum(inputs['attention_mask']).item()

    image = encode_data_to_image(latent_vector, original_length)
    
    # 이미지를 메모리 버퍼에 저장하고 Base64로 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({"image": img_str})

@app.route('/decode', methods=['POST'])
def decode():
    data = request.json
    base64_image = data.get('image')
    mode = data.get('mode', 'y')

    if not base64_image:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        latent_vector, original_length = decode_data_from_image(image)
    except Exception as e:
        return jsonify({"error": f"이미지 처리 오류: {e}"}), 400

    model = model_y if mode == 'y' else model_x
    latent_vector = latent_vector.to(device)
    
    with torch.no_grad():
        if mode == 'x':
            attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long).to(device)
            output_logits = model.decoder(inputs_embeds=latent_vector, attention_mask=attention_mask).logits
            predicted_ids = torch.argmax(output_logits, dim=-1)
            clean_ids = predicted_ids[0][:original_length]
            raw_text = model.tokenizer.decode(clean_ids, skip_special_tokens=True)
            result_text = combine_jamo(raw_text)
        else: # mode 'y'
            attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long).to(device)
            outputs = model.decoder.generate(
                inputs_embeds=latent_vector, attention_mask=attention_mask,
                max_new_tokens=50, do_sample=True, top_k=50,
                pad_token_id=model.tokenizer.pad_token_id
            )
            raw_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result_text = combine_jamo(raw_text)

    return jsonify({"text": result_text})

if __name__ == '__main__':
    app.run(debug=True)
