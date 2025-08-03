import torch
from torch import nn
from transformers import BertModel, BertForCausalLM, BertTokenizer

# ---------------------------------
# 1. 모델 정의 (ArtistX)
# ---------------------------------
class ArtistX(nn.Module):
    """
    ArtistX: 텍스트를 잠재 벡터로 인코딩하고, 다시 원본 텍스트로 완벽하게 복원합니다.
    (Text -> Latent Vector -> Original Text)
    """
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # X의 인코더: 텍스트의 의미를 추출하는 부분
        self.encoder = BertModel.from_pretrained(model_name)
        
        # X의 디코더: 의미로부터 텍스트를 다시 생성하는 부분
        # CausalLM은 다음 단어를 예측하는 데 특화되어 있어 텍스트 생성에 적합합니다.
        self.decoder = BertForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        """모델의 핵심 로직: 인코딩 -> 디코딩"""
        
        # 1. 인코더를 통해 입력 텍스트를 잠재 벡터(의미)로 변환합니다.
        #    outputs.last_hidden_state의 shape: [batch_size, sequence_length, hidden_size]
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        latent_vector = encoder_outputs.last_hidden_state
        
        # 2. 얻어진 잠재 벡터를 디코더에 입력하여 원본 텍스트를 복원하려 시도합니다.
        #    디코더는 이 잠재 벡터를 보고 원본 단어들을 예측합니다.
        #    logits의 shape: [batch_size, sequence_length, vocab_size]
        output_logits = self.decoder(inputs_embeds=latent_vector, attention_mask=attention_mask).logits
        
        return output_logits

    def reconstruct(self, text: str) -> str:
        """사용자 친화적인 복원 함수"""
        self.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt')
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            
            # 가장 확률이 높은 토큰 ID를 예측값으로 선택
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # 예측된 토큰 ID를 다시 텍스트로 변환
            reconstructed_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            return reconstructed_text

# ---------------------------------
# 2. X 모델 학습 (개념 코드)
# ---------------------------------
def train_artist_x_conceptual():
    """
    이 함수는 실제 실행되지 않으며, X를 학습시키는 과정을 설명하기 위한 개념적인 코드입니다.
    실제 학습에는 방대한 텍스트 데이터셋과 고사양 GPU가 필요합니다.
    """
    print("--- ArtistX 학습 과정 (개념) ---")
    
    # 1. 모델, 옵티마이저, 손실 함수 정의
    model_X = ArtistX()
    optimizer = torch.optim.Adam(model_X.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # 2. 📚 데이터 로더 준비 (대규모 텍스트 데이터셋이라고 가정)
    # text_dataset = load_your_massive_text_dataset()
    # data_loader = DataLoader(text_dataset, batch_size=16)

    # 3. 학습 루프
    # for epoch in range(num_epochs):
    #     for batch in data_loader:
    #         optimizer.zero_grad()
    #         input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            
    #         output_logits = model_X(input_ids, attention_mask)
            
    #         # 💡 목표: 모델의 예측(output_logits)이 원본(input_ids)과 같아지도록 학습
    #         loss = loss_fn(output_logits.view(-1, model_X.tokenizer.vocab_size), input_ids.view(-1))
            
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Epoch {epoch+1} 완료, Loss: {loss.item()}")

    # 4. 학습된 모델 가중치 저장
    # torch.save(model_X.state_dict(), "artist_x_weights.pth")
    print("학습이 완료되고 'artist_x_weights.pth' 파일이 저장되었다고 가정합니다.")

# ---------------------------------
# 3. 메인 실행 블록
# ---------------------------------
if __name__ == '__main__':
    print("🎨 '기억하는 화가' ArtistX를 테스트합니다.")
    
    # 개념적인 학습 함수 호출
    train_artist_x_conceptual()
    
    # 모델 인스턴스 생성
    X = ArtistX()
    
    # 💡 중요: 실제로는 아래 주석을 풀고 학습된 가중치를 불러와야 합니다.
    # X.load_state_dict(torch.load("artist_x_weights.pth"))
    
    # 테스트 문장
    original_text = "a painter who remembers what he drew"
    
    # X의 복원 기능 테스트
    reconstructed_text = X.reconstruct(original_text)
    
    print("\n--- X의 재구성 능력 테스트 ---")
    print(f"원본 텍스트: {original_text}")
    print(f"복원된 텍스트: {reconstructed_text}")
    print("\n(참고: 학습되지 않은 모델이므로, 현재 출력은 의미 없는 무작위 텍스트입니다.)")