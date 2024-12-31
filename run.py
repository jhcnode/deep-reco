from flask import Flask, render_template, request, redirect, jsonify,url_for
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
from playwright.async_api import async_playwright
import asyncio
from collections import defaultdict


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
async def embed_texts_parallel(texts, batch_size=16):
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

async def fetch_clien_contents():
    url = "https://www.clien.net/service/recommend"
    start_id = 1
    category = "Community"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)

            await page.wait_for_selector(".list_item", timeout=15000)
            articles = await page.query_selector_all(".list_item")

            if not articles:
                print("데이터를 찾지 못했습니다. 페이지 구조를 다시 확인하세요.")
                return []

            contents = []
            for i, article in enumerate(articles):
                try:
                    title_elem = await article.query_selector("span.subject_fixed")
                    link_elem = await article.query_selector("a.list_subject")

                    title = await title_elem.inner_text() if title_elem else "No Title"
                    link = await link_elem.get_attribute("href") if link_elem else None

                    contents.append({
                        "id": start_id + i,
                        "title": title.strip(),
                        "category": category,
                        "link": f"https://www.clien.net{link}" if link else None,
                        "thumbnail_url": None  # 썸네일 없음
                    })
                except Exception as e:
                    print(f"게시글 추출 중 오류: {e}")
                    continue

            await browser.close()
            return contents

    except Exception as e:
        print(f"{category} - Error: {e}")
        return []

async def fetch_inven_contents():
    url = "https://hot.inven.co.kr/"
    start_id = 1
    category = "Community"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)

            await page.wait_for_selector(".list-common.con", timeout=15000)
            articles = await page.query_selector_all(".list-common.con")

            if not articles:
                print("데이터를 찾지 못했습니다. 페이지 구조를 다시 확인하세요.")
                return []

            contents = []
            for i, article in enumerate(articles):
                try:
                    title_elem = await article.query_selector(".title .name")
                    link_elem = await article.query_selector(".title a")

                    title = await title_elem.inner_text() if title_elem else "No Title"
                    link = await link_elem.get_attribute("href") if link_elem else None

                    contents.append({
                        "id": start_id + i,
                        "title": title,
                        "category": category,
                        "link": link if link else None,
                        "thumbnail_url": None  # 썸네일 없음
                    })
                except Exception as e:
                    print(f"게시글 추출 중 오류: {e}")
                    continue

            await browser.close()
            return contents

    except Exception as e:
        print(f"{category} - Error: {e}")
        return []

async def fetch_twitch_contents():
    url = "https://www.twitch.tv/directory/all"
    start_id = 1
    category = "Streaming"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)

            # 페이지 로딩 후, 모든 카드가 로드될 때까지 기다림
            await page.wait_for_selector('a[data-a-target="preview-card-image-link"]', timeout=30000)
            cards = await page.query_selector_all('a[data-a-target="preview-card-image-link"]')

            if not cards:
                print("데이터를 찾지 못했습니다. 페이지 구조를 다시 확인하세요.")
                return []

            contents = []
            for i, card in enumerate(cards):
                # 썸네일 이미지 추출
                thumbnail_elem = await card.query_selector('img.tw-image')
                thumbnail_url = await thumbnail_elem.get_attribute('src') if thumbnail_elem else None

                # 방송 제목과 채널명 추출
                alt_text = await thumbnail_elem.get_attribute('alt') if thumbnail_elem else None
                game_title, channel_name = None, None
                if alt_text:
                    parts = alt_text.split("님이")
                    if len(parts) >= 2:
                        game_title = parts[1]  # 게임 제목
                        channel_name = parts[0]  # 채널 이름

                # 방송 상태 (생방송) 추출
                status_elem = await card.query_selector('div.ScChannelStatusTextIndicator-sc-qtgrnb-0 p')
                status = await status_elem.inner_text() if status_elem else "Offline"

                # 시청자 수 추출
                viewers_elem = await card.query_selector('div.ScMediaCardStatWrapper-sc-anph5i-0')
                viewers = await viewers_elem.inner_text() if viewers_elem else "0"

                # 방송 링크 추출
                link = await card.get_attribute('href')

                # 정확한 데이터만 리스트에 추가
                if game_title and channel_name and thumbnail_url:
                    contents.append({
                        "id": start_id + i,
                        "channel": channel_name.strip(),
                        "title": game_title.strip(),
                        "viewers": viewers,
                        "category": category,
                        "link": f"https://www.twitch.tv{link}" if link else None,
                        "thumbnail_url": thumbnail_url,  # 썸네일 URL 추가
                        "status": status,  # 방송 상태 추가
                    })

            await browser.close()
            return contents

    except Exception as e:
        print(f"{category} - Error: {e}")
        return []



async def fetch_youtube_contents():
    url = "https://www.youtube.com/feed/trending?gl=KR&hl=ko"
    selector = "ytd-video-renderer"
    base_url = "https://www.youtube.com"
    category = "Streaming"
    contents = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)

            # 첫 번째 비디오 요소가 로드될 때까지 기다림
            await page.wait_for_selector(selector)

            # 스크롤을 내려 모든 콘텐츠 로드
            prev_height = 0
            retries = 0
            while retries < 15:  # 최대 15회 시도
                curr_height = await page.evaluate("document.documentElement.scrollHeight")
                if curr_height == prev_height:
                    retries += 1
                else:
                    retries = 0
                prev_height = curr_height
                await page.evaluate("window.scrollBy(0, 1000)")
                await page.wait_for_timeout(1000)  # 대기 시간 증가로 동적 콘텐츠 로드 보장

            # 모든 비디오 요소 가져오기
            items = await page.query_selector_all(selector)

            for i, item in enumerate(items):
                try:
                    # 제목과 링크 추출
                    title_elem = await item.query_selector("a#video-title")
                    title = await title_elem.get_attribute("title") if title_elem else "No Title"
                    link = await title_elem.get_attribute("href") if title_elem else None

                    # 썸네일 URL 추출
                    thumbnail_elem = await item.query_selector(
                        "ytd-thumbnail img, a#thumbnail img, yt-image img, ytd-thumbnail yt-image img, a#thumbnail yt-image img"
                    )
                    thumbnail_url = None
                    if thumbnail_elem:
                        thumbnail_url = await thumbnail_elem.get_attribute("src")
                        # 일부 동영상에서 썸네일 URL이 data-src로 제공될 수 있으므로 이를 처리
                        if not thumbnail_url or thumbnail_url.startswith("data:"):
                            thumbnail_url = await thumbnail_elem.get_attribute("data-thumb")

                    # 채널 이름 추출
                    channel_elem = await item.query_selector("ytd-channel-name a")
                    channel_name = await channel_elem.inner_text() if channel_elem else "Unknown Channel"

                    # 조회수 추출
                    view_count_elem = await item.query_selector("span.inline-metadata-item")
                    view_count = await view_count_elem.inner_text() if view_count_elem else "Unknown Views"

                    # 비디오 링크가 상대 경로라면 절대 경로로 변환
                    if link and not link.startswith("http"):
                        link = f"{base_url}{link}"

                    # 콘텐츠 저장
                    if title and link:
                        contents.append({
                            "id": 201 + i,
                            "title": title,
                            "category": category,
                            "link": link,
                            "thumbnail_url": thumbnail_url,
                            "channel_name": channel_name,
                            "view_count": view_count
                        })
                except Exception as e:
                    print(f"Error processing item {i}: {e}")

            await browser.close()

    except Exception as e:
        print(f"Error while fetching trending videos: {e}")

    return contents




async def fetch_naver_news_contents():
    url = "https://news.naver.com/main/ranking/popularDay.naver"
    start_id = 1
    category = "News"

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"{category} - Error: {e}")
        return []

    html_content = response.text

    # Pattern to match article links and titles
    pattern = re.compile(r'<a href="(https://n\.news\.naver\.com/article/[^\"]+)"[^>]*class="list_title[^>]*">([^<]+)</a>', re.S)
    matches = pattern.findall(html_content)

    if not matches:
        print(f"{category} - No matches found.")
        return []

    contents = []
    for i, match in enumerate(matches):
        link, title = match
        title = title.strip()
        link = link.strip()

        # Fetch the thumbnail for the article by matching the img tag's src
        thumbnail_pattern = re.compile(
            r'<img src="(https://mimgnews\.pstatic\.net/[^"]+)"[^>]*width="70" height="70"',
            re.S
        )
        thumbnail_match = thumbnail_pattern.search(html_content)

        thumbnail_url = None#thumbnail_match.group(1) if thumbnail_match else None

        if title and link:
            contents.append({
                "id": start_id + i,
                "title": title,
                "category": category,
                "link": link,
                "thumbnail_url": thumbnail_url  # Add thumbnail URL to the result
            })

    return contents


async def fetch_google_news_contents():
    url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZxYUdjU0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR%3Ako"
    selector = "a.gPFEn"  
    category = "News"
    base_url = "https://news.google.com"
    contents = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)

            await page.wait_for_selector(selector)
            items = await page.query_selector_all(selector)
            for i, item in enumerate(items):
                title = await item.inner_text()
                link = await item.get_attribute("href")

                if link and link.startswith("./"):
                    link = f"{base_url}{link[1:]}"
                
                thumbnail_url=None
                
                if title and link:
                    contents.append({
                        "id": i + 1,
                        "title": title.strip(),
                        "category": category,
                        "link": link,
                        "thumbnail_url": thumbnail_url  # Add thumbnail URL to the result
                    })

            await browser.close()

    except Exception as e:
        print(f"{category} - Error: {e}")

    return contents

# # 콘텐츠 데이터베이스를 업데이트하는 함수
# def update_contents():
#     global contents, id_to_index
#     old_contents = contents or []
#     old_id_map = {content["title"]: content["id"] for content in old_contents}

#     # 새로운 콘텐츠 가져오기
#     new_contents = fetch_naver_news_contents() + fetch_google_news_contents() + \
#                    fetch_youtube_contents() + fetch_twitch_contents() + \
#                    fetch_inven_contents() + fetch_clien_contents()
                   
                   
#     random.shuffle(new_contents)

#     # ID 유지: 기존 콘텐츠는 이전 ID 사용, 새 콘텐츠는 새로운 ID 부여
#     start_id = max(old_id_map.values(), default=0) + 1
#     for content in new_contents:
#         if content["title"] in old_id_map:
#             content["id"] = old_id_map[content["title"]]
#         else:
#             content["id"] = start_id
#             start_id += 1

#     # 콘텐츠 처리
#     process_contents(new_contents)
#     contents = new_contents
#     id_to_index = {content["id"]: idx for idx, content in enumerate(contents)}
    
async def update_contents():
    global contents, id_to_index
    
    if(contents is not None):
        return
    
    # 기존 콘텐츠 저장
    old_contents = contents or []
    old_id_map = {content["title"]: content["id"] for content in old_contents}

    # 비동기적으로 새로운 콘텐츠 가져오기
    new_contents = await asyncio.gather(
        fetch_clien_contents(),
        fetch_inven_contents(),
        fetch_twitch_contents(),
        fetch_youtube_contents(),
        fetch_naver_news_contents(),
        fetch_google_news_contents()
    )
    print(new_contents)
    
    # 중복 제거 및 ID 유지
    combined_contents = []
    unique_titles = set()
    start_id = max(old_id_map.values(), default=0) + 1

    for result in new_contents:
        for content in result:
            if content["title"] not in unique_titles:
                unique_titles.add(content["title"])
                if content["title"] in old_id_map:
                    content["id"] = old_id_map[content["title"]]
                else:
                    content["id"] = start_id
                    start_id += 1
                combined_contents.append(content)

    # 콘텐츠 섞기
    random.shuffle(combined_contents)

    # 임베딩 처리
    await process_contents(combined_contents)

    # 업데이트된 콘텐츠와 ID 매핑
    contents = combined_contents
    id_to_index = {content["id"]: idx for idx, content in enumerate(contents)}


# 콘텐츠 제목을 임베딩으로 변환
async def process_contents(contents):
    if not contents:
        print("process_contents: No contents to process.")
        return  # contents가 비어 있으면 반환

    texts = [content["title"] for content in contents]
    embeddings = await embed_texts_parallel(texts)

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

from flask import jsonify
import numpy as np

@app.route('/')
def index():
    global contents

    # 사용자가 선택한 카테고리를 GET 파라미터로 받음
    selected_category = request.args.get('category', 'all')  # 기본값은 'all'
    mode = request.args.get('mode')  # AJAX 요청 모드: 'recommendations' or 'category_contents'

    # 콘텐츠 업데이트 (비동기)
    asyncio.run(update_contents())

    # 콘텐츠 필터링
    if selected_category != 'all':
        filtered_contents = [content for content in contents if content["category"] == selected_category]
    else:
        filtered_contents = contents

    # AJAX 요청 처리
    if mode == 'recommendations':
        embeddings = [content["embedding"] for content in filtered_contents]
        top_actions = agent.act(embeddings)
        top_actions = agent.getfilter(top_actions, top_k=5)
        recommendations = [filtered_contents[action] for action in top_actions]

        # JSON 직렬화 가능한 데이터로 변환
        for content in recommendations:
            if isinstance(content.get("embedding"), np.ndarray):
                content["embedding"] = content["embedding"].tolist()

        return jsonify(recommendations)

    elif mode == 'category_contents':
        categorized_contents = defaultdict(list)
        for content in filtered_contents:
            if isinstance(content.get("embedding"), np.ndarray):
                content["embedding"] = content["embedding"].tolist()
            categorized_contents[content["category"]].append(content)

        return jsonify(categorized_contents)

    # 기본 렌더링 요청 처리
    embeddings = [content["embedding"] for content in filtered_contents]
    top_actions = agent.act(embeddings)
    top_actions = agent.getfilter(top_actions, top_k=5)
    recommendations = [filtered_contents[action] for action in top_actions]

    categorized_contents = defaultdict(list)
    for content in filtered_contents:
        categorized_contents[content["category"]].append(content)
    
    return render_template(
        'index.html',
        recommendations=recommendations,
        categorized_contents=categorized_contents,
        selected_category=selected_category
    )



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