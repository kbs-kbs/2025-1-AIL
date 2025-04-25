# LSTM
- rnn과 마찬가지로 한 층에 각 셀(time step)이 일렬로 존재
- 기울기 소실과 장기 의존성 문제 해결을 위해 고안된 RNN 모델
- rnn의 hidden state에 cell state 추가
- gate 추가
  1. 필요한 정보 기억
  2. 불필요한 정보 약화
  3. 새로운 정보 선택적 저장
- Element-wise multiplication: 동일한 자리끼리 곱셈
- hiddenstate: short-term memory 보관, output gate와 cell state를 통해 업데이트
- cellstate: 장기적 정보 보관, 핵심저장소, forget gate, input gate, candidate gate를 통해 업데이트

- Forget gate: 이전 시점 cell state에서 잊을 걸 지움
- candidate gate: 입력을 받아 값(초안) 생성
- input gate: 값 중 기억할것을 선택해서 cell state 반영
- Output gate: cell state를 바탕으로 다음 시점의 hidden state 생성

1. 이전 hs와 입력값을 받아 concatenate하고
2. forget gate를 통해 cs를 갱신하고
3. concatenate 결과를 candidate gate가 tanh로 정규화 하고
4. input gate와 곱해 기억할것을 cell state에 반영하고
5. output gate로 cell state를 바탕으로 다음 시점의 hidden state 생성


1. LSTM을여러층을쌓으면무엇이좋은가요?- 더 깊은 층이 문장의 복잡한 패턴이나 의미를 더 잘 학습함.
2. hidden state와 cell state의 근본적인 차이가무엇인가요?- hidden state는 단기 기억을 가지고 있으면서 출력값을 내어오기 위해, cell state는 장기적으로 기억을 저장하기 위함.
3. 각층의cell state는따로존재하나요?- yes. 각 LSTM의층마다다른정보를기억하고있음.
4. 문장길이가다다른데, 왜padding을안해요?- 문장 전체를 일괄적으로 받아오는 게 아니라, 항상 단어 2개만 입력하기 때문에 지금은 padding이 필요없음.
5. 문장이짧으면generate_sequence는어떻게되나요?- 단어가3개미만인문장은학습데이터에포함되지않도록구현함.
6. BOW에저장안돼있는단어를입력받으려면어떻게하면돼요?- <UNK> 와같은미지정(unknown) 토큰을추가함. (지난주에했던RNN  예제참고


Layer 1:   t=1   t=2   t=3   ...   t=T
              |     |     |           |
Layer 2:   t=1   t=2   t=3   ...   t=T

각 셀의 출력(hidden state, cell state)은

같은 층의 다음 time step 셀로 전달되고,

다음 층의 같은 time step 셀로도 전달됩니다.

#LSTM에 들어가는 입력 모양 [batch_size, seq_len, input_size]
# - batch_size: 한 번에 처리하는 문장(데이터)의 개수 (예: 64)
# - seq_len: 한 문장에서 모델이 바라보는 단어의 개수 (예: 2개씩 입력)
# - input_size: 임베딩을 거친 단어 벡터의 차원 수 (예: 16차원)
