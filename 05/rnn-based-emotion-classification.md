1. 데이터 가져오기
```python
import torch
import torch.nn as nn
import torch.optim as optim

f = open('train_data.txt', 'r', encoding='utf-8')
data = f.read()
exec(data.replace('\n', ''))

train_data[:5]
```
2. 라벨 인코딩
```python
label2idx = {"긍정": 0, "부정": 1}
idx2label = {0: "긍정", 1: "부정"}
```
3. 단어 사전
```python
word_vocab = {"<PAD>": 0, "<UNK>": 1}
for words, _ in train_data:
    for w in words:
        if w not in word_vocab:
            word_vocab[w] = len(word_vocab)
```

4. 시퀀스 인코딩 함수
```python
def encode(words, label, max_len):
    x = [word_vocab.get(w, word_vocab["<UNK>"]) for w in words]
    x += [word_vocab["<PAD>"]] * (max_len - len(x))
    y = label2idx[label]
    return torch.tensor(x), torch.tensor(y)

max_len = max(len(s) for s, _ in train_data)
encoded_data = [encode(s, t, max_len) for s, t in train_data]
```

5. 모델 정의
```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

vocab_size = len(word_vocab)
model = SentimentRNN(vocab_size, embed_dim=16, hidden_dim=32, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```
6. 학습
```python
for epoch in range(100):
    total_loss = 0
    for x, y in encoded_data:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
```
7. 예측 함수
```python
def predict(sentence):
    x = [word_vocab.get(w, word_vocab["<UNK>"]) for w in sentence]
    x += [word_vocab["<PAD>"]] * (max_len - len(x))
    x = torch.tensor(x).unsqueeze(0)
    output = model(x)
    pred = torch.argmax(output, dim=1).item()
    return idx2label[pred]
```
8. 테스트
```python
test_sent = ["정말", "좋았어"]
print("문장:", test_sent)
print("예측된 감정:", predict(test_sent))
print("")

test_sent = ["오늘", "완전", "별로였어", "짜증나네"]
print("문장:", test_sent)
print("예측된 감정:", predict(test_sent))
```
