# GRU
LSTM은RNN의장기의존성문제를해결하였지만, 구조가복잡하여학습시간이오래걸린다는단점이
있음
그래서LSTM의성능은유지하면서도더간단하고학습이빠른구조가필요해짐
GRU(Gated Recurrent Unit)의 등장
 3(또는4)개의gate가있는LSTM과달리, GRU는단2개의gate만존재함
• GRU의 두gate들이정보의흐름을정교하게조절함
• Update Gate: 기억을얼마나업데이트할까?
 • (과거정보와새정보의비율조절)
 • Reset Gate: 이전 기억을얼마나지워버릴까?
 • (과거정보를얼마나적게쓸지조절)
 • 2개의state(hidden state, cell state)를 사용하던 LSTM과 달리, GRU는Hidden state만사용함
- reset gate와 update gate는 동시에계산될수있으며, 서로에게
영향을주지않음.- 오직g_t 계산을할때에만reset gate의연산결과가사용되고,
최종hidden state인h_t계산에서만update gate가사용됨

# CRNN
 Input -> CNN Layer -> RNN Layer -> Output
 CTC (Connectionist Temporal Classification)라는 손실함수를 사용함
 정답시퀀스의길이와예측시퀀스의길이가다를때정렬없이학습할수있는손실함수.
 : 즉, 하나의정답을표현하는방법이여러가지존재할때, 가능한모든표현방식을고려하여손실을
계산하도록설계된함수
CNN(ResNet) + RNN(GRU)
