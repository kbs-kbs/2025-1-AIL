# ResNet
- Skip connectiond을 이용한 CNN 신경망
- CNN 중 가장 많이 쓰이는 모델
- 기울기 소실 문제 해결책 -> skip connection: f(x)에 x를 그냥 더하는 단계
- 레이어를 거치지 않고 x를 전달
- 이때 다운샘플을 해서 채널 수를 맞춰줌
- Conv → BN(배치 정규화) → ReLU → Conv → BN 구조
- 배치 정규화: 배치마다 값의 분포가 서로 다르면 출력값의 분포가 달라져서 악영향, 배치의 출력 분포를 일정하게 해줌

# 실습
```python
class BasicBlock(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size=3):#기본 커널 사이즈 3*3
       super(BasicBlock, self).__init__()


       #합성곱층 정의
       self.c1 = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=1)
       self.c2 = nn.Conv2d(out_channels, out_channels,
                           kernel_size=kernel_size, padding=1)

       self.downsample = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1)#x의 채널 수를 F(x)와 같게 맞춰주는 역할.
                                   #채널의 수가 같으면 downsample 부분이 생략되도록 구현 수정 가능

       #배치 정규화층 정의
       self.bn1 = nn.BatchNorm2d(num_features=out_channels)
       self.bn2 = nn.BatchNorm2d(num_features=out_channels)

       self.relu = nn.ReLU()
   def forward(self, x): #Conv → BN → ReLU → Conv → BN 구조

       #스킵 커넥션을 위해 초기 입력을 저장
       x_ = x

       # x_를 다운샘플해서 채널 맞춰주고 x와 더한 뒤에 ReLU 활성화

       x = self.c1(x)
       x = self.bn1(x)
       x = self.relu(x)
       x = self.c2(x)
       x = self.bn2(x)
       x_ = self.downsample(x_)
       x += x_
       x = self.relu(x)

       return x
```

```python
#ResNet 모델 정의하기

class ResNet(nn.Module):
   def __init__(self, num_classes=10):#분류할 클래스의 수가 10
       super(ResNet, self).__init__()


       #기본 블록
       self.b1 = BasicBlock(in_channels=3, out_channels=64)
       self.b2 = BasicBlock(in_channels=64, out_channels=128)
       self.b3 = BasicBlock(in_channels=128, out_channels=256)


       #풀링을 최댓값이 아닌 평균값으로
       self.pool = nn.AvgPool2d(kernel_size=2, stride=2) #MaxPooling으로 변경 가능

       #분류기 #fully connected layer(MLP)
       #(생각해보기) 왜 4096일까?
       self.fc1 = nn.Linear(in_features=4096, out_features=2048)
       self.fc2 = nn.Linear(in_features=2048, out_features=512)
       self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

       self.relu = nn.ReLU()


   def forward(self, x):
       #기본 블록과 풀링층을 통과 = 전체 특징 추출 파트
       #블록 수 계속 늘려보기
       x = self.b1(x)
       x = self.pool(x)
       x = self.b2(x)
       x = self.pool(x)
       x = self.b3(x)
       x = self.pool(x)


       #분류기의 입력으로 사용하기 위해 flatten
       x = torch.flatten(x, start_dim=1)

       #분류기로 예측값 출력: 최종 출력 생성
       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)
       x = self.relu(x)
       x = self.fc3(x)

       return x
```

만약 입력 이미지가 32×32라고 가정하고,   
각 블록 뒤에 AvgPool2d(kernel_size=2, stride=2)를 3번 적용하면:   
   
입력: (batch, 3, 32, 32)    
   
b1 + pool: (batch, 64, 16, 16)    

b2 + pool: (batch, 128, 8, 8)   

b3 + pool: (batch, 256, 4, 4)   

이렇게 최종적으로 (batch, 256, 4, 4)가 됩니다.   
 
flatten 이후   
torch.flatten(x, start_dim=1)을 하면  
(batch, 256, 4, 4) → (batch, 256×4×4) → (batch, 4096)   

그래서 nn.Linear(4096, ...)로 연결할 수 있습니다.    
