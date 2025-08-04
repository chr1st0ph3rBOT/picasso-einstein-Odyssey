# app_y.py
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from artist_y_train import ArtistY # Y 모델 구조 import

class AppY_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artist Y - The Re-interpreter")
        self.root.geometry("500x400")

        # --- 모델 로드 ---
        try:
            self.model = ArtistY()
            # 0단계에서 생성한 독립된 Y 모델 파일을 불러옵니다.
            self.model.load_state_dict(torch.load("artist_y_standalone.pth"))
            self.model.eval()
            self.status_text = "모델 로딩 완료."
        except FileNotFoundError:
            self.model = None
            self.status_text = "오류: artist_y_standalone.pth 파일을 찾을 수 없습니다."
            messagebox.showerror("오류", self.status_text)
        
        self.create_widgets()

    def create_widgets(self):
        # UI는 AppX_GUI와 거의 동일
        tk.Label(self.root, text="텍스트 입력:").pack(pady=5)
        self.text_input = tk.Text(self.root, height=5, width=60)
        self.text_input.pack(pady=5, padx=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.encode_button = tk.Button(button_frame, text="텍스트 → '이미지' 파일로 저장", command=self.encode_text)
        self.encode_button.pack(side=tk.LEFT, padx=10)
        
        # 버튼 텍스트만 다름
        self.decode_button = tk.Button(button_frame, text="'이미지' 파일 → 텍스트로 재해석", command=self.decode_image)
        self.decode_button.pack(side=tk.LEFT, padx=10)

        tk.Label(self.root, text="결과:").pack(pady=5)
        self.text_output = tk.Text(self.root, height=5, width=60, state=tk.DISABLED)
        self.text_output.pack(pady=5, padx=10)

        self.status_label = tk.Label(self.root, text=self.status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def encode_text(self):
        # encode_text 함수는 AppX_GUI와 완전히 동일
        if not self.model: return
        input_text = self.text_input.get("1.0", "end-1c").strip()
        if not input_text:
            messagebox.showwarning("입력 오류", "텍스트를 입력해주세요.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch Tensors", "*.pt"), ("All Files", "*.*")],
            title="잠재 벡터 '이미지' 파일로 저장"
        )
        if not filepath: return

        with torch.no_grad():
            inputs = self.model.tokenizer(input_text, return_tensors='pt')
            latent_vector = self.model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        
        torch.save(latent_vector, filepath)
        self.status_label.config(text=f"✅ '{filepath}'에 '이미지' 저장 완료.")
        messagebox.showinfo("성공", f"'{filepath}'에 '이미지' 파일을 저장했습니다.")

    def decode_image(self):
        # decode_image 함수는 Y의 재해석 기능을 호출
        if not self.model: return
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Tensors", "*.pt"), ("All Files", "*.*")],
            title="해석할 '이미지' 파일 열기"
        )
        if not filepath: return
        
        latent_vector = torch.load(filepath)
        
        # Y의 reinterpret 기능을 사용
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