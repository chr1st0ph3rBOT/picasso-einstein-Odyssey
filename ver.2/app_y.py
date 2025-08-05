# app_y.py
import torch
import tkinter as tk
from torch import nn
from tkinter import filedialog, messagebox
from artist_y_train import ArtistY

# transformers 라이브러리에서 필요한 클래스들을 직접 import 합니다.
from transformers import BertModel, BertLMHeadModel, BertTokenizer

# --- ArtistY 모델 클래스 (통합됨) ---
class ArtistY(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)

    def load_encoder_from_x(self, x_model):
        """
        이 함수는 이제 bundle_y_model.py에서만 사용되므로,
        GUI 앱에서는 직접 호출되지 않습니다. 구조만 유지합니다.
        """
        encoder_weights = x_model.encoder.state_dict()
        self.encoder.load_state_dict(encoder_weights)
# ------------------------------------

# --- 이미지 유틸리티 함수 (통합됨) ---
import numpy as np
from PIL import Image
import matplotlib as plt
def encode_data_to_image(tensor: torch.Tensor, original_length: int, target_size=(1280, 1280)) -> Image.Image:
    tensor_np = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    tensor_bytes = tensor_np.tobytes()
    length_bytes = original_length.to_bytes(4, byteorder='big')
    combined_bytes = length_bytes + tensor_bytes
    image_array = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    byte_array = np.frombuffer(combined_bytes, dtype=np.uint8)
    image_array.flat[:len(byte_array)] = byte_array
    
    vis_img = Image.fromarray(tensor_np)
    vis_img_resized = vis_img.resize((256, 256), Image.Resampling.BICUBIC)
    vis_array = np.array(vis_img_resized)
    normalized_array = (vis_array - vis_array.min()) / (vis_array.max() - vis_array.min())
    colored_array = (plt.cm.get_cmap('viridis')(normalized_array)[:, :, :3] * 255).astype(np.uint8)
    final_visual_img = Image.fromarray(colored_array)
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
# ------------------------------------

class AppY_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artist Y - The Re-interpreter (Metadata Ver.)")
        self.root.geometry("500x400")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = ArtistY()
            self.model.load_state_dict(torch.load("artist_y_standalone.pth", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.status_text = f"모델 로딩 완료. (Device: {str(self.device).upper()})"
        except FileNotFoundError:
            self.model = None
            self.status_text = "오류: artist_y_standalone.pth 파일을 찾을 수 없습니다."
            messagebox.showerror("오류", self.status_text)
        
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="텍스트 입력:").pack(pady=5)
        self.text_input = tk.Text(self.root, height=5, width=60)
        self.text_input.pack(pady=5, padx=10)
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        self.encode_button = tk.Button(button_frame, text="텍스트 → PNG 이미지로 저장", command=self.encode_text)
        self.encode_button.pack(side=tk.LEFT, padx=10)
        self.decode_button = tk.Button(button_frame, text="PNG 이미지 → 텍스트로 재해석", command=self.decode_image)
        self.decode_button.pack(side=tk.LEFT, padx=10)
        tk.Label(self.root, text="결과:").pack(pady=5)
        self.text_output = tk.Text(self.root, height=5, width=60, state=tk.DISABLED)
        self.text_output.pack(pady=5, padx=10)
        self.status_label = tk.Label(self.root, text=self.status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def encode_text(self):
        if not self.model: return
        input_text = self.text_input.get("1.0", "end-1c").strip()
        if not input_text:
            messagebox.showwarning("입력 오류", "텍스트를 입력해주세요.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="추상화 PNG 이미지로 저장"
        )
        if not filepath: return

        with torch.no_grad():
            inputs = self.model.tokenizer(input_text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            latent_vector = self.model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
            original_length = torch.sum(inputs['attention_mask']).item()
        
        # 💡 FIX: 'original_length' 인자를 함께 전달합니다.
        image = encode_data_to_image(latent_vector, original_length)
        image.save(filepath, "PNG")
        
        self.status_label.config(text=f"✅ '{filepath}'에 PNG 이미지 저장 완료.")
        messagebox.showinfo("성공", f"'{filepath}'에 PNG 이미지를 저장했습니다.")

    def decode_image(self):
        if not self.model: return
        filepath = filedialog.askopenfilename(
            filetypes=[("PNG Image", "*.png")],
            title="재해석할 PNG 이미지 열기"
        )
        if not filepath: return
        
        try:
            from PIL import Image
            image = Image.open(filepath)
            latent_vector, _ = decode_data_from_image(image)
        except Exception as e:
            messagebox.showerror("파일 오류", f"이미지 파일을 처리하는 중 오류가 발생했습니다: {e}")
            return
        
        with torch.no_grad():
            latent_vector = latent_vector.to(self.device)
            attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long).to(self.device)

            outputs = self.model.decoder.generate(
                inputs_embeds=latent_vector,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True, top_k=50,
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
