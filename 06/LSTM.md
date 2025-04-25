# LSTM
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


lstm 여러층을 쌍흐면 복잡한 패턴이나 의미를 더 잘 학습
단기 기억을 가지며 출력값을 내기 위해 장기적
yes, lstm층마다 다른 정보 기억
문장 전체를 일괄적으로 받아오는 게 아니라 항상 단어 2개만 입력하기 때문에 padding이 필요 없음
단어가 3개 미만인 문장은 학습 데이터에 포함되지 않도록 구현
<unk>과 같은 미지정 토큰을 추가함 rnn예제 참고
