import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertLMHeadModel, BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm  # tqdm 라이브러리 import
import warnings

warnings.filterwarnings("ignore")

## 1. 모델 클래스 정의
class ArtistX(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        # 💡 FIX: is_decoder=True 옵션 추가
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        latent_vector = encoder_outputs.last_hidden_state
        output_logits = self.decoder(inputs_embeds=latent_vector, attention_mask=attention_mask).logits
        return output_logits

    def reconstruct(self, text: str) -> str:
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True).to(self.encoder.device)
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            predicted_ids = torch.argmax(logits, dim=-1)
            reconstructed_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            return reconstructed_text

## 2. 메인 실행 블록
if __name__ == '__main__':
    # --- 하이퍼파라미터 및 설정 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = 'bert-base-uncased'
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 10

    print(f"🚀 ArtistX 학습을 시작합니다. Device: {DEVICE}")

    # --- 데이터 준비 ---
    print("1. 데이터셋 로드 및 전처리 중...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # 💡 FIX: trust_remote_code=True 삭제
    raw_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
    print("✅ 데이터 준비 완료.")

    # --- 모델 및 학습 설정 ---
    print("2. 모델 및 옵티마이저 설정 중...")
    model = ArtistX(model_name=MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print("✅ 모델 설정 완료.")

    # --- 학습 루프 실행 ---
    print("3. 학습을 시작합니다...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        # 💡 LOGGING: tqdm을 사용해 진행률 표시줄 생성
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            output_logits = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(output_logits.view(-1, tokenizer.vocab_size), batch['input_ids'].view(-1))
            loss.backward()
            optimizer.step()
            
            # 💡 LOGGING: 진행률 표시줄에 실시간 손실 값 업데이트
            progress_bar.set_postfix(loss=loss.item())

    print("\n✅ 학습이 최종 완료되었습니다.")

    torch.save(model.state_dict(), "artist_x_weights.pth")
    print("모델 가중치가 'artist_x_weights.pth'에 저장되었습니다.")