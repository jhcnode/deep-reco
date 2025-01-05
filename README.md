# 사용자 피드백 기반 개인화 콘텐츠 추천 시스템(DEEP-RECO)

이 프로젝트는 다양한 플랫폼에서 데이터를 가져와 사용자의 피드백을 기반으로 개인화된 추천을 제공하는 Flask 웹 애플리케이션입니다. 주된 특징과 사용된 기술 스택은 다음과 같습니다.

---

## 주요 기능

### 1. 데이터 수집 및 처리
- **플랫폼**: 
  - 클리앙
  - 인벤
  - 트위치
  - 유튜브
  - 네이버 뉴스
  - 구글 뉴스
  - 인스타그램
  - 틱톡
- **기술**: Playwright를 사용한 웹 스크래핑과 비동기 방식으로 빠르게 데이터를 수집.

### 2. 임베딩 및 유사도 분석
- **모델**: `SentenceTransformer` (xlm-r-100langs-bert-base-nli-stsb-mean-tokens)
- **기능**:
  - 콘텐츠 제목을 임베딩으로 변환하여 추천 알고리즘에 활용.
  - `cosine similarity`를 통해 사용자의 선호도와 콘텐츠 간 유사성을 분석.

### 3. 개인화된 추천
- 사용자의 긍정 및 부정 피드백을 수집하여 추천 결과를 지속적으로 업데이트.
- 비슷한 콘텐츠를 실시간으로 추천.

### 4. 범주화된 콘텐츠 관리
- 콘텐츠를 카테고리별로 정리하여 사용자 검색 및 탐색을 용이하게 함.

### 5. 캐싱 및 이미지 제공
- 스크래핑한 이미지와 데이터를 로컬에 캐싱하여 효율성을 향상.

---

## 기술 스택

### Python 라이브러리 및 프레임워크
- **Flask**: 웹 애플리케이션 프레임워크
- **Playwright**: 비동기 웹 스크래핑
- **Torch**: 딥러닝 모델 처리
- **SentenceTransformers**: 텍스트 임베딩 생성
- **Scikit-learn**: 코사인 유사도 계산

### 데이터 관리
- `asyncio` 및 `aiohttp`: 비동기 데이터 처리

---

## 주요 코드 구조

### 1. 데이터 수집
플랫폼별로 비동기 함수가 정의되어 있으며, 각 함수는 다음을 포함합니다:
- 데이터 스크래핑 로직
- 콘텐츠 구조화
- 썸네일, 제목, 링크 등의 필드 수집

### 2. 추천 로직
사용자가 좋아하거나 싫어한 콘텐츠에 따라:
- 피드백 데이터를 활용하여 사용자 선호를 학습.
- 임베딩 기반 유사도 분석으로 개인화된 추천 제공.

### 3. UI 및 피드백
- Flask 템플릿을 사용하여 웹 인터페이스 제공.
- 사용자 피드백 (좋아요/싫어요)을 폼을 통해 수집.

---

## 실행 방법

1. 필수 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Flask 애플리케이션 실행:
```bash
python run.py
```

3. 웹 브라우저에서 다음 주소로 접속:
```
http://127.0.0.1:5000
```

---

## 사용된 데이터 구조

### 콘텐츠 예시
```json
{
  "id": 1,
  "title": "추천 콘텐츠 제목",
  "category": "Community",
  "link": "https://example.com",
  "thumbnail_url": "https://example.com/thumbnail.jpg",
  "embeddings": [0.1, 0.2, 0.3, ...]
}
```

### 사용자 히스토리 예시
```json
{
  "liked": ["좋아하는 콘텐츠 제목"],
  "disliked": ["싫어하는 콘텐츠 제목"]
}
```

---

## 주요 포인트
- 확장 가능한 구조로 다양한 플랫폼의 데이터를 통합.
- GPU를 활용한 임베딩 처리를 통해 효율적인 계산.
- 사용자 피드백을 적극적으로 활용하여 점진적인 성능 개선.

---
## 웹페이지 결과
![chrome_ARl2paw38k](https://github.com/user-attachments/assets/1e7bea71-2077-41cc-a6a7-9c7e4fbbdd4d)
![chrome_C5evV64Qxz](https://github.com/user-attachments/assets/1b70414d-83e6-4d26-84ad-ffe12cea0020)


