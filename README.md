# Stalking Detection Platform
11개 설문을 통해 유저의 스토킹 지수를 진단해주는 플랫폼입니다. 

## Contributor
* [홍지우](https://github.com/jiwooya1000)
* [한지원](https://github.com/mutedlemon)

## Platform
[스토킹 지수 진단 Platform](https://healthy-date-app.herokuapp.com/)

![Nuxtjs](https://img.shields.io/badge/Javascript-Nuxt.js-brightgreen) 
![Vuejs](https://img.shields.io/badge/Javascript-Vue.js-brightgreen) 
![Flask](https://img.shields.io/badge/Python-Flask-orange)

## 모델링 
![pytorch](https://img.shields.io/badge/Python-Pytorch-orange)

### 스토킹 잠재도 탐지
* 스토킹 잠재도 탐지는 단어 단위(word-wise) 임베딩, 문항 단위(doc.-wise) 임베딩, 그리고 설문조사 단위 임베딩의 3가지 단계로 이루어집니다.
* 단어 단위 임베딩은 KcELECTRA를 기반으로 임베딩 되었습니다.
* 문항 단위 임베딩은 Self Attentive BiLSTM을 기반으로 임베딩 되었습니다. Attention을 기반으로 문항별로 중요한 단어를 탐지합니다.
* 설문조사 단위 임베딩은 단순한 Dot Product Attention으로 임베딩 되었습니다. 각 설문조사 문항별 임베딩 행렬을 기반으로 스토킹 잠재도를 추론하며 Attention을 학습합니다.
### 정신건강 분류 
* 웰니스 대화 스크립트 데이터셋에 대해 학습되며, 우울, 초조, 공통(위험군)의 3가지로 정신건강을 분류합니다. Self Attentivㅣe BiLSTM을 사용하여 예측된 정신건강 라벨에 대한 Attention을 학습합니다.
### 스토킹 유형 분류 
* GloVe 기반으로 설문조사 텍스트를 임베딩했으며, 13가지 스토킹 행위 유형별 대표 키워드와 유사한 상위 20개의 유사어를 사용합니다. 각 내담자의 데이터에 대해 13가지 스토킹 행위 유형별 대표 키워드 유사어들이 등장한 횟수와 유사도를 가중합하여 행위 유형에 대한 가능성을 반환합니다. 최종적으로 13가지 값을 Softmax에 통과시켜 각 유형에 해당할 확률을 유형 분류 결과로 활용합니다.

## 모델 시연 
### 파일 개요
TRAINED_MODEL과 NEW_MODEL은 [여기](https://drive.google.com/drive/folders/1PstVsK3hnAd5-9heB8oKWx1CxZfUFQcy?usp=sharing)서 다운로드 받으시면 됩니다. 

```bash
├── TRAINED_MODEL : 플랫폼 구동에 사용될 실제 모델이 저장된 디렉토리
│   ├── MENTAL
│   ├── GloVe
│   ├── Classifier
│   └── SE
│
├── NEW_MODEL : 학습 과정 테스트시 모델이 저장되는 디렉토리
│   ├── MENTAL
│   ├── GloVe
│   ├── Classifier
│   └── SE
│
├── DATA : 학습에 활용된 데이터셋이 저장되어 있는 디렉토리
│   ├── Luv.D
│   ├── 3.잠재 및 가해 분류
│   ├── 2.설문조사
│   └── 1.정신건강
│
├── 1-1. 정신건강 분류 모델 학습.ipynb
├── 1-2. 설문조사 임베딩 모델 학습.ipynb
├── 1-3. 스토킹 잠재/가해자 분류 모델 학습.ipynb
├── 1-4. 스토킹 유형 분류 모델 학습.ipynb
├── 2-1. 스토킹 위험도 지수 예측.ipynb
├── utils.py
├── train.py
├── preprocess.py
├── model.py
├── dataset.py
├── demo.py
└── README.md
```
### 모델을 학습데이터로 학습하는 방법

 * 정신건강 분류 모델
	- Ai Hub의 '웰니스 대화 스크립트 데이터셋' 기반 정신건강 Multiclass Classification Model
	- "1-1. 정신건강 분류 모델 학습.ipynb" 접속
	- 모든 셀 실행
	- "./NEW_MODEL/MENTAL/' 디렉토리에 학습된 모델(.pt) 파일 저장
	- 학습 Loss History 시각화

 * 설문조사 임베딩 Siamese Network 모델
	- (주)럽디 제공 설문조사 데이터 기반 설문조사 문항별 임베딩 모델
	- "1-2. 설문조사 임베딩 모델 학습.ipynb" 접속
	- "SURVEY_NUMBER"에 학습할 문항의 번호 기입
	- 모든 셀 실행
	- "./NEW_MODEL/SE/' 디렉토리에 학습된 모델(.pt) 파일 저장
	- 학습 Loss History 시각화

  * 스토킹 위험도 지수 예측 모델
	- (주)럽디 제공 설문조사 데이터와 Ai Hub의 '웰니스 대화 스크립트 데이터셋' 기반 설문조사 위험도 지수 예측 모델
	- "1-3. 스토킹 잠재/가해자 분류 모델.ipynb" 접속
	- 모든 셀 실행
	- "./NEW_MODEL/Classifier/' 디렉토리에 학습된 모델(.pt) 파일 저장
	- 학습 Loss History 시각화

  * 스토킹 유형 분류 모델
	- (주)럽디 제공 설문조사 데이터 기반 스토킹 유형 분류 모델
	- "1-4. 스토킹 유형 분류 모델.ipynb" 접속
	- 모든 셀 실행
	- "./NEW_MODEL/GloVe/' 디렉토리에 학습된 모델(.model) 파일 저장
	- 스토킹 카테고리 분류 결과 시각화


### 모델을 테스트데이터로 테스트하는 방법

테스트 단계에서는 모든 데이터의 구조가 기존에 제공된 '경진대회 데이터_수정본.csv'과 완벽히 동일해야 합니다.

  * 스토킹 위험도 지수 예측
	- "2-1. 스토킹 위험도 지수 예측.ipynb" 접속
	- '(3) 데이터셋 불러오기'의 첫 번째 셀에 csv 형태 데이터를 입력
	- 모든 셀을 실행시킬 경우 모든 입력 데이터에 대한 스토킹 위험도 지수 계산 후 분포 시각화

   * 스토킹 유형 예측
	- "2-2. 스토킹 유형 예측.ipynb" 접속
	- '(3) 데이터셋 불러오기'의 첫 번째 셀에 csv 형태 데이터를 입력
	- 모든 셀을 실행시킬 경우 모든 입력 데이터에 대한 13가지 스토킹 행위 유형별 해당 확률을 출력하고, 전체 분포를 시각화
