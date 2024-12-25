from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from playwright.sync_api import sync_playwright
from pytrends.request import TrendReq
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset
import re
from playwright.sync_api import sync_playwright

# Flask 앱 생성
app = Flask(__name__)

# 임베딩 모델 준비
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Sentence-BERT 모델 사용
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# 드라이버 재사용 설정
driver_service = Service(ChromeDriverManager().install())
driver_options = Options()
driver_options.add_argument("--headless")
driver_options.add_argument("--disable-gpu")
driver_options.add_argument("--no-sandbox")
driver_options.add_argument("--disable-dev-shm-usage")
driver_options.add_argument('--disable-extensions')  # 확장 프로그램 비활성화
driver_options.add_argument('--no-sandbox')  # 리소스 격리 비활성화
driver_options.add_experimental_option("excludeSwitches", ["enable-logging"])

# Dataset 클래스 정의
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# 병렬 임베딩 함수
@torch.no_grad()
def embed_texts_parallel(texts, batch_size=16):
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    model.eval()
    for batch in dataloader:
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings.extend(batch_embeddings)

    return embeddings

def fetch_clien_contents():
    url = "https://www.clien.net/service/recommend"
    start_id = 1
    category = "Community"

    try:
        with sync_playwright() as p:
            # 브라우저 설정 및 페이지 열기
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)

            # 데이터 로드 대기
            page.wait_for_selector(".list_item", timeout=15000)

            # 게시글 정보 수집
            articles = page.query_selector_all(".list_item")

            if not articles:
                print("데이터를 찾지 못했습니다. 페이지 구조를 다시 확인하세요.")
                return []

            contents = []
            for i, article in enumerate(articles):
                try:
                    # 제목, 링크, 작성자, 조회수, 댓글 수 등 데이터 추출
                    title_elem = article.query_selector("span.subject_fixed")
                    link_elem = article.query_selector("a.list_subject")
                    # writer_elem = article.query_selector(".list_author .nickname span")
                    # date_elem = article.query_selector(".list_time .timestamp")
                    # hits_elem = article.query_selector(".list_hit .hit")
                    # comments_elem = article.query_selector(".list_reply .rSymph05")
                    # likes_elem = article.query_selector(".list_symph span")

                    title = title_elem.inner_text().strip() if title_elem else "No Title"
                    link = link_elem.get_attribute("href") if link_elem else None
                    # writer = writer_elem.inner_text().strip() if writer_elem else "Unknown"
                    # date = date_elem.inner_text().strip() if date_elem else "Unknown"
                    # hits = hits_elem.inner_text().strip() if hits_elem else "0"
                    # comments = comments_elem.inner_text().strip() if comments_elem else "0"
                    # likes = likes_elem.inner_text().strip() if likes_elem else "0"

                    contents.append({
                        "id": start_id + i,
                        "title": title,
                        # "writer": writer,
                        # "date": date,
                        # "hits": hits,
                        # "comments": comments,
                        # "likes": likes,
                        "category": category,
                        "link": f"https://www.clien.net{link}" if link else None,
                    })
                except Exception as e:
                    print(f"게시글 추출 중 오류: {e}")
                    continue

            # 브라우저 닫기
            browser.close()
            return contents

    except Exception as e:
        print(f"{category} - Error: {e}")
        return []


def fetch_inven_contents():
    url = "https://hot.inven.co.kr/"
    start_id = 1
    category = "Community"

    try:
        with sync_playwright() as p:
            # 브라우저 설정 및 페이지 열기
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)

            # 데이터 로드 대기
            page.wait_for_selector(".list-common.con", timeout=15000)

            # 데이터 수집
            articles = page.query_selector_all(".list-common.con")

            if not articles:
                print("데이터를 찾지 못했습니다. 페이지 구조를 다시 확인하세요.")
                return []

            contents = []
            for i, article in enumerate(articles):
                try:
                    # 제목, 링크 추출
                    title_elem = article.query_selector(".title .name")
                    link_elem = article.query_selector(".title a")
                    # category_elem = article.query_selector(".cate")
                    # comment_elem = article.query_selector(".comment")
                    # writer_elem = article.query_selector(".writer .layerNickName")
                    # date_elem = article.query_selector(".date")
                    # hits_elem = article.query_selector(".hits")
                    # reco_elem = article.query_selector(".reco")

                    title = title_elem.inner_text().strip() if title_elem else "No Title"
                    link = link_elem.get_attribute("href") if link_elem else None
                    # category_name = category_elem.inner_text().strip() if category_elem else "Unknown"
                    # comment_count = comment_elem.inner_text().strip("[]") if comment_elem else "0"
                    # writer = writer_elem.inner_text().strip() if writer_elem else "Unknown"
                    # date = date_elem.inner_text().strip() if date_elem else "Unknown"
                    # hits = hits_elem.inner_text().strip() if hits_elem else "0"
                    # reco = reco_elem.inner_text().strip() if reco_elem else "0"

                    contents.append({
                        "id": start_id + i,
                        "title": title,
                        "category": category,
                        # "comments": comment_count,
                        # "writer": writer,
                        # "date": date,
                        # "hits": hits,
                        # "recommendations": reco,
                        "link": link if link else None,
                    })
                except Exception as e:
                    print(f"게시글 추출 중 오류: {e}")
                    continue

            # 브라우저 닫기
            browser.close()
            return contents

    except Exception as e:
        print(f"{category} - Error: {e}")
        return []



def fetch_twitch_contents():
    url = "https://www.twitch.tv/directory/all"
    start_id = 1
    category = "Streaming"

    try:
        with sync_playwright() as p:
            # 브라우저 설정 및 페이지 열기
            browser = p.chromium.launch(headless=True) 
            page = browser.new_page()
            page.goto(url, timeout=60000)

            # 데이터 로드 대기
            page.wait_for_selector('a[data-a-target="preview-card-channel-link"]', timeout=15000)

            # 카드 수집
            cards = page.query_selector_all('a[data-a-target="preview-card-channel-link"]')

            if not cards:
                print("데이터를 찾지 못했습니다. 페이지 구조를 다시 확인하세요.")
                return []

            # 데이터 처리
            contents = []
            for i, card in enumerate(cards):
                aria_label = card.get_attribute('aria-label')  # aria-label에서 데이터 추출
                title = card.query_selector('h3')  # 제목 추출
                link = card.get_attribute('href')  # 링크 추출

                # 시청자 수는 별도 요소에서 가져와야 할 수 있음
                viewer_count = card.query_selector('p[data-a-target="preview-card-channel-link"]')  

                # 데이터 가공
                game_title = title.get_attribute('title') if title else None
                channel_name = aria_label.split(' ')[0] if aria_label else None
                viewers = viewer_count.inner_text().strip() if viewer_count else "0"

                if game_title and channel_name:
                    contents.append({
                        "id": start_id + i,
                        "channel": channel_name.strip(),
                        "title": game_title.strip(),
                        "viewers": viewers,
                        "category": category,
                        "link": f"https://www.twitch.tv{link}" if link else None
                    })

            # 브라우저 닫기
            browser.close()
            return contents

    except Exception as e:
        print(f"{category} - Error: {e}")
        return []
    


def fetch_youtube_contents():
    url = "https://www.youtube.com/feed/trending?gl=KR&hl=ko"
    selector = "a#video-title"
    category = "Streaming"

    base_url = "https://www.youtube.com"  # Base URL for relative links
    contents = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # Wait for the elements to load
        page.wait_for_selector(selector)

        items = page.query_selector_all(selector)
        for i, item in enumerate(items):
            title = item.get_attribute("title") or item.inner_text().strip()
            link = item.get_attribute("href")

            # Ensure the link is absolute
            if link and not link.startswith("http"):
                link = f"{base_url}{link}"

            if title and link:
                contents.append({
                    "id": 201 + i,
                    "title": title,
                    "category": category,
                    "link": link
                })

        browser.close()

    return contents


# 크롤링 함수들 업데이트
def fetch_naver_news_contents():
    url = "https://news.naver.com/main/ranking/popularDay.naver"
    start_id = 1
    category = "News"

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"{category}{e}")
        return []

    html_content = response.text

    # 정규식을 사용해 뉴스 항목 추출
    pattern = re.compile(r'<a href="(https://n\.news\.naver\.com/article/[^\"]+)"[^>]*class="list_title[^>]*">([^<]+)</a>', re.S)
    matches = pattern.findall(html_content)

    if not matches:
        print(f"{category} - No matches found in the HTML content. Check the page structure or pattern.")
        return []

    contents = []
    for i, match in enumerate(matches):
        link, title = match
        title = title.strip()
        link = link.strip()
        if title and link:
            contents.append({
                "id": start_id + i,
                "title": title,
                "category": category,
                "link": link
            })

    return contents

def fetch_google_trends_contents():
    pytrends = TrendReq(hl='ko', tz=540)
    trending_searches_df = pytrends.trending_searches(pn='south_korea')

    contents = []
    try:
        for i, row in trending_searches_df.iterrows():
            title = row[0]
            link = f"https://www.google.com/search?q={title}"
            contents.append({
                "id": i + 101,
                "title": title,
                "category": "Google Trends",
                "link": link
            })
    except Exception as e:
        print("구글 트렌드 데이터 크롤링 실패:", e)

    return contents

def fetch_google_news_contents():
    url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZxYUdjU0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR%3Ako"
    selector = "a.gPFEn"  # Selector for Google News article links
    category="News"
    base_url = "https://news.google.com"  # Base URL for relative links
    contents = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # Wait for the elements to load
        page.wait_for_selector(selector)

        items = page.query_selector_all(selector)
        for i, item in enumerate(items):
            title = item.inner_text().strip()
            link = item.get_attribute("href")

            # Ensure the link is absolute
            if link and link.startswith("./"):
                link = f"{base_url}{link[1:]}"

            if title and link:
                contents.append({
                    "id": i + 1,  # Ensure unique IDs
                    "title": title,
                    "category": category,
                    "link": link
                })

        browser.close()

    return contents

# 콘텐츠 데이터베이스를 업데이트하는 함수
def update_contents():
    global contents, id_to_index
    old_contents = contents or []
    old_id_map = {content["title"]: content["id"] for content in old_contents}

    # 새로운 콘텐츠 가져오기
    new_contents = fetch_naver_news_contents() + fetch_google_news_contents() + \
                   fetch_youtube_contents() + fetch_twitch_contents() + \
                   fetch_inven_contents() + fetch_clien_contents()
                   
                   
    random.shuffle(new_contents)

    # ID 유지: 기존 콘텐츠는 이전 ID 사용, 새 콘텐츠는 새로운 ID 부여
    start_id = max(old_id_map.values(), default=0) + 1
    for content in new_contents:
        if content["title"] in old_id_map:
            content["id"] = old_id_map[content["title"]]
        else:
            content["id"] = start_id
            start_id += 1

    # 콘텐츠 처리
    process_contents(new_contents)
    contents = new_contents
    id_to_index = {content["id"]: idx for idx, content in enumerate(contents)}


# 콘텐츠 제목을 임베딩으로 변환
def process_contents(contents):
    texts = [content["title"] for content in contents]
    embeddings = embed_texts_parallel(texts)
    for content, embedding in zip(contents, embeddings):
        content["embedding"] = embedding

# 강화학습 네트워크 모델 정의
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 강화학습 모델 (DQN 기반)
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = nn.DataParallel(QNetwork(state_dim, action_dim).to(device))
        self.target_network = nn.DataParallel(QNetwork(state_dim, action_dim).to(device))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.discount_factor = 0.99
        self.epsilon = 0.1
        self.memory = []
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def act(self, states):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states).to(device)
            q_values = self.q_network(state_tensor)
            # Q 값 기반으로 내림차순 정렬
            top_actions = torch.argsort(q_values[:, 0], descending=True).tolist()
            return top_actions

    def getfilter(self,top_actions,top_k=5):
        global disliked_titles
        global contents
        
        unique_titles = set()
        filtered_actions = []
        print(disliked_titles)
        for action in top_actions:
            title = contents[action]["title"]
            # disliked_titles에 포함된 제목은 제외
            if title not in disliked_titles and title not in unique_titles:
                unique_titles.add(title)
                filtered_actions.append(action)

            if len(filtered_actions) == top_k:
                break

        # 추천이 부족하면 남은 항목에서 추가
        if len(filtered_actions) < top_k:
            remaining_candidates = [
                action for action in range(len(contents))
                if contents[action]["title"] not in disliked_titles
                and contents[action]["title"] not in unique_titles
            ]
            random.shuffle(remaining_candidates)
            filtered_actions.extend(remaining_candidates[:top_k - len(filtered_actions)])

        # 최종 추천 결과
        return filtered_actions[:top_k]
    

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Replay Loss: {loss.item()}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# 싫어요 제목 목록
disliked_titles = []

# 시스템 초기화
contents=None
state_dim = 384
action_dim = 2000
agent = DQNAgent(state_dim, action_dim)

@app.route('/')
def index():
    update_contents()
    global contents
    embeddings = [content["embedding"] for content in contents]
    top_actions = agent.act(embeddings)
    top_actions = agent.getfilter(top_actions,top_k=5)
    recommendations = [contents[action] for action in top_actions]
    return render_template('index.html', recommendations=recommendations)

@app.route('/feedback', methods=['POST'])
def feedback():
    global disliked_titles, contents, id_to_index
    feedback_data = request.form

    for content_id, feedback in feedback_data.items():
        try:
            # ID를 인덱스로 변환
            action = id_to_index.get(int(content_id))
            if action is None:
                raise KeyError(f"Content ID {content_id} not found in id_to_index.")

            # 강화 학습 처리
            reward = int(feedback)
            state = contents[action]["embedding"]

            if reward < 0:
                disliked_title = contents[action]["title"]
                disliked_titles.append(disliked_title)
                print(f"Disliked title added: {disliked_title}")

            next_state_candidates = [
                content["embedding"] for content in contents
                if content["title"] not in disliked_titles
            ][:5]
            next_state = random.choice(next_state_candidates) if next_state_candidates else state
            done = False
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            agent.update_target_network()

        except KeyError as e:
            print(f"Invalid content ID: {content_id}, Error: {e}")
        except Exception as e:
            print(f"Error processing feedback: {e}")

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)