# app_x.py
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from artist_x_train import ArtistX # X 모델 구조 import

class AppX_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artist X - The Reconstructor")
        self.root.geometry("500x400")

        # --- 모델 로드 ---
        try:
            self.model = ArtistX()
            self.model.load_state_dict(torch.load("artist_x_best_model.pth"))
            self.model.eval()
            self.status_text = "모델 로딩 완료."
        except FileNotFoundError:
            self.model = None
            self.status_text = "오류: artist_x_best_model.pth 파일을 찾을 수 없습니다."
            messagebox.showerror("오류", self.status_text)
        
        self.create_widgets()

    def create_widgets(self):
        # --- UI 요소 생성 ---
        # 텍스트 입력 영역
        tk.Label(self.root, text="텍스트 입력:").pack(pady=5)
        self.text_input = tk.Text(self.root, height=5, width=60)
        self.text_input.pack(pady=5, padx=10)

        # 버튼 프레임
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # '텍스트 -> 이미지' 버튼
        self.encode_button = tk.Button(button_frame, text="텍스트 → '이미지' 파일로 저장", command=self.encode_text)
        self.encode_button.pack(side=tk.LEFT, padx=10)

        # '이미지 -> 텍스트' 버튼
        self.decode_button = tk.Button(button_frame, text="'이미지' 파일 → 텍스트로 복원", command=self.decode_image)
        self.decode_button.pack(side=tk.LEFT, padx=10)

        # 결과 출력 영역
        tk.Label(self.root, text="결과:").pack(pady=5)
        self.text_output = tk.Text(self.root, height=5, width=60, state=tk.DISABLED)
        self.text_output.pack(pady=5, padx=10)

        # 상태 표시줄
        self.status_label = tk.Label(self.root, text=self.status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def encode_text(self):
        if not self.model: return
        input_text = self.text_input.get("1.0", "end-1c").strip()
        if not input_text:
            messagebox.showwarning("입력 오류", "텍스트를 입력해주세요.")
            return

        # 파일 저장 대화상자 열기
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch Tensors", "*.pt"), ("All Files", "*.*")],
            title="잠재 벡터 '이미지' 파일로 저장"
        )
        if not filepath: return

        # 텍스트를 잠재 벡터로 변환
        with torch.no_grad():
            inputs = self.model.tokenizer(input_text, return_tensors='pt')
            latent_vector = self.model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        
        # 파일로 저장
        torch.save(latent_vector, filepath)
        self.status_label.config(text=f"✅ '{filepath}'에 '이미지' 저장 완료.")
        messagebox.showinfo("성공", f"'{filepath}'에 '이미지' 파일을 저장했습니다.")

    def decode_image(self):
        if not self.model: return
        # 파일 열기 대화상자 열기
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Tensors", "*.pt"), ("All Files", "*.*")],
            title="해석할 '이미지' 파일 열기"
        )
        if not filepath: return
        
        # 잠재 벡터 파일 로드
        latent_vector = torch.load(filepath)
        
        # 잠재 벡터로부터 텍스트 복원
        with torch.no_grad():
            attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long)
            output_logits = self.model.decoder(inputs_embeds=latent_vector, attention_mask=attention_mask).logits
            predicted_ids = torch.argmax(output_logits, dim=-1)
            reconstructed_text = self.model.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
        # 결과 출력
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert("1.0", reconstructed_text)
        self.text_output.config(state=tk.DISABLED)
        self.status_label.config(text=f"✅ '{filepath}' 파일 복원 완료.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppX_GUI(root)
    root.mainloop()