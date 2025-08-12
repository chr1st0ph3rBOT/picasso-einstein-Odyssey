# artist_x_secure_train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertLMHeadModel, BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm
import warnings
import os
import random

warnings.filterwarnings("ignore")

# --- ArtistX 모델 클래스 ---
class ArtistX(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)
# ... (이하 모델 코드는 이전과 동일)

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    # --- 하이퍼파라미터 및 설정 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = 'bert-base-uncased'
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 3 # 최적점이 3 에포크 근처였으므로, 3~5 정도로 설정

    # 💡 --- 보안 강화: 고유 데이터셋 생성을 위한 비밀 시드 --- 💡
    # 이 시드값을 변경하면 완전히 새로운 마스터 키가 생성됩니다.
    # 이 값은 마스터 키(.pth)와 함께 비밀리에 보관해야 합니다.
    SECRET_DATA_SEED = "Picasso-Protocol-Janus-2025" 
    DATA_SUBSET_RATIO = 0.95 # 원본 데이터의 95%만 사용하여 고유성을 부여합니다.

    print(f"🚀 ArtistX 보안 강화 훈련을 시작합니다. Device: {DEVICE}")
    print(f"비밀 시드: '{SECRET_DATA_SEED}'")

    # --- 데이터 준비 ---
    print("1. 데이터셋 로드 및 고유 변형 작업 중...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    raw_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # 💡 --- 데이터셋 변형 로직 --- 💡
    # 1. 비밀 시드를 기반으로 랜덤 생성기 초기화
    rng = random.Random(SECRET_DATA_SEED)
    
    # 2. 원본 훈련 데이터셋의 인덱스를 섞음
    train_indices = list(range(len(raw_dataset["train"])))
    rng.shuffle(train_indices)
    
    # 3. 섞인 인덱스에서 95%만 선택하여 고유한 서브셋 생성
    num_samples_to_keep = int(len(train_indices) * DATA_SUBSET_RATIO)
    unique_train_indices = train_indices[:num_samples_to_keep]
    
    unique_train_dataset = raw_dataset["train"].select(unique_train_indices)
    print(f"원본 훈련 데이터 {len(raw_dataset['train'])}개 중 {len(unique_train_dataset)}개를 선택하여 고유 데이터셋을 생성했습니다.")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    # 💡 고유하게 변형된 훈련 데이터셋을 토크나이징
    tokenized_train_dataset = unique_train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val_dataset = raw_dataset["validation"].map(tokenize_function, batched=True, remove_columns=["text"])

    tokenized_train_dataset.set_format("torch")
    tokenized_val_dataset.set_format("torch")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
    val_dataloader = DataLoader(tokenized_val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
    print("✅ 데이터 준비 완료.")

    # --- 모델 및 학습 설정 ---
    print("2. 모델 및 옵티마이저 설정 중...")
    model = ArtistX(model_name=MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print("✅ 모델 설정 완료.")

    # --- 학습 루프 실행 ---
    print("3. 학습을 시작합니다...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]")
        for batch in train_progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            output_logits = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(output_logits.view(-1, tokenizer.vocab_size), batch['input_ids'].view(-1))
            loss.backward()
            optimizer.step()
            train_progress_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} [Val]")
        with torch.no_grad():
            for batch in val_progress_bar:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                output_logits = model(batch['input_ids'], batch['attention_mask'])
                loss = loss_fn(output_logits.view(-input_logits.view(-1, tokenizer.vocab_size), batch['input_ids'].view(-1)))
                val_loss += loss.item()
                val_progress_bar.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch + 1} | Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "artist_x_best_model.pth")

    print("\n✅ 학습이 최종 완료되었습니다.")
    print(f"고유한 마스터 키가 'artist_x_best_model.pth'에 저장되었습니다.")
