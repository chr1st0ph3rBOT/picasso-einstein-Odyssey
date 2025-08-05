# app_y.py
import torch
from torch import nn
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from transformers import BertModel, BertLMHeadModel, BertTokenizer

# --- ArtistY 모델 클래스 (통합됨) ---
class ArtistY(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)

# --- 이미지 유틸리티 함수 (통합됨) ---
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

class AppY_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artist Y - The Re-interpreter (Standalone)")
        self.root.geometry("500x450")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_widgets()

    def create_widgets(self):
        load_frame = tk.Frame(self.root)
        load_frame.pack(pady=10)
        self.load_button = tk.Button(load_frame, text="Artist Y 모델 불러오기 (.pth)", command=self.load_model)
        self.load_button.pack()

        tk.Label(self.root, text="텍스트 입력:").pack(pady=5)
        self.text_input = tk.Text(self.root, height=5, width=60)
        self.text_input.pack(pady=5, padx=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        self.encode_button = tk.Button(button_frame, text="텍스트 → PNG 이미지", command=self.encode_text, state=tk.DISABLED)
        self.encode_button.pack(side=tk.LEFT, padx=10)
        self.decode_button = tk.Button(button_frame, text="PNG 이미지 → 텍스트", command=self.decode_image, state=tk.DISABLED)
        self.decode_button.pack(side=tk.LEFT, padx=10)

        tk.Label(self.root, text="결과:").pack(pady=5)
        self.text_output = tk.Text(self.root, height=5, width=60, state=tk.DISABLED)
        self.text_output.pack(pady=5, padx=10)

        self.status_label = tk.Label(self.root, text="모델을 불러와주세요.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth")],
            title="artist_y_standalone.pth 파일을 선택하세요"
        )
        if not filepath: return
        
        try:
            self.model = ArtistY()
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.status_label.config(text=f"✅ 모델 로딩 완료. (Device: {str(self.device).upper()})")
            self.encode_button.config(state=tk.NORMAL)
            self.decode_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("모델 로딩 오류", f"모델을 불러오는 중 오류가 발생했습니다: {e}")
            self.model = None

    def encode_text(self):
        # ... (이하 함수 내용은 이전과 동일)
        if not self.model: return
        input_text = self.text_input.get("1.0", "end-1c").strip()
        if not input_text: messagebox.showwarning("입력 오류", "텍스트를 입력해주세요."); return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")], title="추상화 PNG 이미지로 저장")
        if not filepath: return
        with torch.no_grad():
            inputs = self.model.tokenizer(input_text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            latent_vector = self.model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
            original_length = torch.sum(inputs['attention_mask']).item()
        image = encode_data_to_image(latent_vector, original_length)
        image.save(filepath, "PNG")
        self.status_label.config(text=f"✅ '{filepath}'에 PNG 이미지 저장 완료.")
        messagebox.showinfo("성공", f"'{filepath}'에 PNG 이미지를 저장했습니다.")

    def decode_image(self):
        # ... (이하 함수 내용은 이전과 동일, Y의 재해석 로직 사용)
        if not self.model: return
        filepath = filedialog.askopenfilename(filetypes=[("PNG Image", "*.png")], title="재해석할 PNG 이미지 열기")
        if not filepath: return
        try:
            image = Image.open(filepath)
            latent_vector, _ = decode_data_from_image(image)
        except Exception as e:
            messagebox.showerror("파일 오류", f"이미지 파일을 처리하는 중 오류가 발생했습니다: {e}"); return
        with torch.no_grad():
            latent_vector = latent_vector.to(self.device)
            attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long).to(self.device)
            outputs = self.model.decoder.generate(
                inputs_embeds=latent_vector, attention_mask=attention_mask,
                max_new_tokens=50, do_sample=True, top_k=50,
                pad_token_id=self.model.tokenizer.pad_token_id
            )
            reinterpreted_text = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert("1.0", reinterpreted_text)
        self.text_output.config(state=tk.DISABLED)
        self.status_label.config(text=f"✅ '{filepath}' 파일 재해석 완료.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppY_GUI(root)
    root.mainloop()
