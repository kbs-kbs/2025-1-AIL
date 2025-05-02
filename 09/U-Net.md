
## U-Net 아키텍처 개요

U-Net은 주로 이미지 분할(Image Segmentation)에 사용되는 합성곱 신경망(CNN) 기반의 아키텍처로, 2015년 의료 영상 분석 분야에서 처음 제안되었습니다[3][6]. 이름처럼 전체 구조가 U자 형태를 띠며, 인코더(Encoder, 축소 경로)와 디코더(Decoder, 확장 경로)로 대칭적으로 구성되어 있습니다[1][3][5].

---

**아키텍처 구조**

- 인코더(Contracting Path):  
  입력 이미지를 점차적으로 축소하며(다운샘플링) 주요 특징(feature)을 추출합니다. 각 단계는 2개의 3x3 컨볼루션과 ReLU 활성화, 그리고 2x2 맥스풀링(Max Pooling)으로 이루어져 있습니다. 이 과정에서 공간 해상도는 줄어들고 채널 수는 늘어납니다[1][3][7].

- 디코더(Expanding Path):  
  추출된 특징을 바탕으로 이미지를 점차적으로 복원(업샘플링)하여 원래 해상도에 가까운 분할 결과를 만듭니다. 업샘플링(Transposed Convolution 또는 업컨볼루션)과 2개의 3x3 컨볼루션, ReLU 활성화로 구성됩니다[1][7].

- 스킵 연결(Skip Connection):  
  인코더의 각 단계에서 얻은 고해상도 특징 맵을 디코더의 대응 단계에 직접 연결(concatenation)합니다. 이를 통해 디코더가 위치 정보와 세밀한 경계 정보를 잘 복원할 수 있습니다[1][2][4].

- 브릿지(Bridge):  
  인코더와 디코더의 중간에 위치하는 병목(bottleneck) 부분으로, 가장 압축된 특징을 담고 있습니다[1][5].

---

**U-Net의 특징 및 장점**

- U자형 대칭 구조로 인해 고해상도 정보와 저해상도 정보가 모두 활용되어, 미세한 경계까지 정밀하게 분할할 수 있습니다[3].
- 스킵 연결 덕분에 위치 정보 손실을 최소화하면서도, 깊은 신경망의 장점(복잡한 특징 추출)을 모두 누릴 수 있습니다[1][2][4].
- 적은 양의 주석 데이터로도 효과적으로 학습할 수 있어, 특히 의료 영상 등 데이터가 귀한 분야에서 강점을 보입니다[6].
- 완전 연결 계층 없이, 입력 이미지의 유효한 부분만을 사용해 픽셀 단위의 분할 맵을 생성합니다[6].

---

**적용 분야**

- 의료 영상(병변, 세포, 장기 분할 등)
- 위성 이미지 분석, 생물학적 이미지, 일반 컴퓨터 비전의 다양한 세분화 문제 등[3][6].

---

**정리**

U-Net은 인코더-디코더 구조와 스킵 연결을 결합한 U자형 아키텍처로, 이미지의 전역적 맥락과 세부 위치 정보를 동시에 활용해 높은 정확도의 이미지 분할을 달성하는 것이 핵심입니다[1][3][6].

Citations:
[1] https://velog.io/@lighthouse97/UNet%EC%9D%98-%EC%9D%B4%ED%95%B4
[2] https://pasus.tistory.com/204
[3] https://ai-bt.tistory.com/entry/U-Net-%EC%9D%98-%EC%9D%B4%ED%95%B4
[4] https://wikidocs.net/148870
[5] https://jaylala.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-with-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0U-Net%EC%9D%B4%EB%9E%80-U-Net-Convolutional-Networks-for-Biomedical-Image-Segmentation
[6] https://wikidocs.net/128392
[7] https://computing-jhson.tistory.com/61
[8] https://joungheekim.github.io/2020/09/28/paper-review/

---
Perplexity로부터의 답변: https://www.perplexity.ai/search/u-net-akitegceoe-daehae-seolmy-MDwlldCnSCKYnGuWao__RQ?utm_source=copy_output
