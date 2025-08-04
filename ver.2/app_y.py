# app_y.py
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from artist_y_train import ArtistY
# 💡 새로 만든 유틸리티 import
from image_utils import tensor_to_image, image_to_tensor

class AppY_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artist Y - The Re-interpreter (PNG Ver.)")
        self.root.geometry("500x400")

        try:
            self.model = ArtistY()
            self.model.load_state_dict(torch.load("artist_y_standalone.pth"))
            self.model.eval()
            self.status_text = "모델 로딩 완료."
        except FileNotFoundError:
            self.model = None
            self.status_text = "오류: artist_y_standalone.pth 파일을 찾을 수 없습니다."
            messagebox.showerror("오류", self.status_text)
        
        self.create_widgets()

    def create_widgets(self):
        # UI 구조는 이전과 동일
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
        # 이 함수는 AppX와 완전히 동일
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
            inputs = self.model.tokenizer(input_text, return_tensors='pt')
            latent_vector = self.model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        image = tensor_to_image(latent_vector)
        image.save(filepath, "PNG")
        self.status_label.config(text=f"✅ '{filepath}'에 PNG 이미지 저장 완료.")
        messagebox.showinfo("성공", f"'{filepath}'에 PNG 이미지를 저장했습니다.")

    def decode_image(self):
        # Y의 재해석 로직
        if not self.model: return
        filepath = filedialog.askopenfilename(
            filetypes=[("PNG Image", "*.png")],
            title="재해석할 PNG 이미지 열기"
        )
        if not filepath: return
        
        try:
            from PIL import Image
            image = Image.open(filepath)
            latent_vector = image_to_tensor(image)
        except Exception as e:
            messagebox.showerror("파일 오류", f"이미지 파일을 처리하는 중 오류가 발생했습니다: {e}")
            return
        
        with torch.no_grad():
            outputs = self.model.decoder.generate(
                inputs_embeds=latent_vector,
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