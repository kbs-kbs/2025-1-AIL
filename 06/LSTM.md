여러 게이트를 추가하여 정보의 흐름 제어

- Input gate + candidate gate: [1,2,3] ㅇ [4, 5, 6] = [1, 2,3, 4, 5, 6] 
- Output gate
- Forget gate


- Element-wise multiplication: 동일한 자리끼리 곱셈


lstm 여러층을 쌍흐면 복잡한 패턴이나 의미를 더 잘 학습
단기 기억을 가지며 출력값을 내기 위해 장기적
yes, lstm층마다 다른 정보 기억
문장 전체를 일괄적으로 받아오는 게 아니라 항상 단어 2개만 입력하기 때문에 padding이 필요 없음
단어가 3개 미만인 문장은 학습 데이터에 포함되지 않도록 구현
<unk>과 같은 미지정 토큰을 추가함 rnn예제 참고
