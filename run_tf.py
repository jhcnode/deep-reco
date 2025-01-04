from flask import Flask, render_template, request, redirect, jsonify,url_for, session as _sess, send_from_directory
import requests
from playwright.sync_api import sync_playwright
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
from sentence_transformers import SentenceTransformer
import os
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import json
import hashlib
import aiohttp
import aiofiles
from pathlib import Path

# Flask 앱 생성
app = Flask(__name__)

# 모델 로드
EMBEDDING_MODEL_NAME = "xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
if torch.cuda.is_available():
    embedding_model = embedding_model.to(torch.device("cuda"))


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
    for batch in dataloader:
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=False, batch_size=batch_size, device="cuda" if torch.cuda.is_available() else "cpu")
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



@app.route('/images/<path:filename>')
def serve_image(filename):
    """로컬에 저장된 이미지를 제공"""
    cache_dir="D:/deep-reco/cached_images"
    return send_from_directory(cache_dir, filename)


async def fetch_instagram_contents():
    username = ""
    password = ""
    session_file = "session_storage.json"
    explore_url = "https://www.instagram.com/explore/"
    cache_dir = "D:/deep-reco/cached_images"
    os.makedirs(cache_dir,exist_ok=True)
    contents = []
    category="SNS"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        storage = None

        # 세션 저장소 불러오기
        try:
            with open(session_file, "r") as f:
                storage = json.load(f)
                await context.add_cookies(storage)
                print("세션 데이터 로드 완료.")
        except FileNotFoundError:
            print("세션 파일이 없습니다. 새로 로그인합니다.")

        page = await context.new_page()

        # 세션이 없으면 로그인
        if not storage:
            login_url = "https://www.instagram.com/accounts/login/"
            await page.goto(login_url)
            await page.fill('input[name="username"]', username)
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            await page.wait_for_timeout(5000)  # 로그인 완료 대기

            # 세션 저장
            storage = await context.cookies()
            with open(session_file, "w") as f:
                json.dump(storage, f)

        # 탐색 페이지 이동
        await page.goto(explore_url)
        await page.wait_for_selector('div._aagv')  # 콘텐츠 로드 대기

        # 콘텐츠 가져오기
        items = await page.query_selector_all('a._a6hd')
        async with aiohttp.ClientSession() as session:
            for i, item in enumerate(items):
                try:
                    link = await item.get_attribute('href')
                    img_elem = await item.query_selector('img')
                    thumbnail_url = await img_elem.get_attribute('src') if img_elem else None

                    # 캐싱된 이미지 확인 및 저장
                    if thumbnail_url:
                        # 해시값으로 파일 이름 생성
                        img_hash = hashlib.md5(thumbnail_url.encode()).hexdigest()
                        cached_file = cache_dir+"/"+f"{img_hash}.jpg"

                        if os.path.exists(cached_file)==False:
                            # 이미지 다운로드
                            print(f"다운로드 중: {thumbnail_url}")
                            async with session.get(thumbnail_url) as response:
                                if response.status == 200:
                                    content = await response.read()
                                    async with aiofiles.open(cached_file, mode="wb") as f:
                                        await f.write(content)

                        # 제목/캡션 추출
                        parent_div = await item.query_selector("div._aagu")
                        if not parent_div:
                            continue
                        caption_elem = await parent_div.query_selector("div._aacl")
                        title = await caption_elem.inner_text() if caption_elem else "No Title {}".format(i+1)
                        cached_file_name=os.path.basename(cached_file)

                        contents.append({
                            "id": i + 1,
                            "title": title,
                            "link": f"https://www.instagram.com{link}",
                            "thumbnail_url":  f"/images/{cached_file_name}",
                            "category":category
                        })
                except Exception as e:
                    print(f"Error processing item {i}: {e}")

        await browser.close()

    return contents

    
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
        fetch_instagram_contents(),
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
    embeddingss = await embed_texts_parallel(texts)

    for content, embeddings in zip(contents, embeddingss):
        content["embeddings"] = embeddings.tolist()


# 시스템 초기화
contents=None

user_history = {'liked': [], 'disliked': []}  # 단일 사용자 환경을 가정

# 유사도 기반 필터링 함수
def filter_top_similarities(query_embedding, contents, top_k=5):
    if not contents:
        return []
    content_embeddings = np.array([content["embeddings"] for content in contents])
    similarities = cosine_similarity([query_embedding], content_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [contents[i] for i in top_indices]

# Flask 라우트 정의
@app.route('/')
def index():
    global contents, user_history

    # 콘텐츠 업데이트 비동기 실행
    asyncio.run(update_contents())


    selected_category = request.args.get('category', 'all')
    search_query = request.args.get('search_query', '').lower()
    mode = request.args.get('mode')  # AJAX 요청 모드
    batch_size = 32

    # 사용자 히스토리를 기반으로 콘텐츠 필터링
    filtered_contents = [content for content in contents if content["title"] not in user_history['disliked']]

    # 카테고리 필터링
    if selected_category != 'all':
        filtered_contents = [content for content in filtered_contents if content["category"] == selected_category]

    # 검색 텍스트 필터링
    if search_query:
        filtered_contents = [content for content in filtered_contents if search_query in content["title"].lower()]


    # 사용자 히스토리를 기반으로 쿼리 생성
    queries = user_history['liked'] if user_history['liked'] else [content["title"] for content in contents[:5]]
    all_recommendations = {}

    # TextDataset과 DataLoader를 사용한 배치 처리
    query_dataset = TextDataset(queries)
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    for query_batch in query_dataloader:
        try:
            # 질문 인코더 서브 토크나이저로 인코딩 및 임베딩 생성
            query_embeddings = embedding_model.encode(list(query_batch), convert_to_tensor=False)

            # 각 쿼리 임베딩에 대해 유사도 기반 추천 필터링
            for query_embedding in query_embeddings:
                batch_recommendations = filter_top_similarities(query_embedding, filtered_contents, top_k=5)
                for recommendation in batch_recommendations:
                    rec_id = recommendation["id"]
                    similarity_score = cosine_similarity([query_embedding], [recommendation["embeddings"]])[0][0]
                    if rec_id in all_recommendations:
                        all_recommendations[rec_id]["score"] += similarity_score
                    else:
                        all_recommendations[rec_id] = {"content": recommendation, "score": similarity_score}
        except Exception as e:
            print(f"Error processing query batch: {e}")
            continue
        
    # 유사도 점수 기준으로 정렬
    sorted_recommendations = sorted(all_recommendations.values(), key=lambda x: x["score"], reverse=True)
    final_recommendations = [rec["content"] for rec in sorted_recommendations[:5]]
    
    if mode == 'recommendations':
        return jsonify(final_recommendations)
    elif mode == 'category_contents':
        categorized_contents = defaultdict(list)
        for content in filtered_contents:
            if isinstance(content.get("embeddings"), np.ndarray):
                content["embeddings"] = content["embeddings"].tolist()
            categorized_contents[content["category"]].append(content)

        return jsonify(categorized_contents)
        
    # 일반 페이지 렌더링
    categorized_contents = defaultdict(list)
    for content in filtered_contents:
        categorized_contents[content["category"]].append(content)

    return render_template(
        'index.html',
        recommendations=final_recommendations[:5],
        categorized_contents=categorized_contents,
        selected_category=selected_category,
        search_query=search_query
    )

# 피드백 라우트 정의
@app.route('/feedback', methods=['POST'])
def feedback():
    global contents, user_history

    feedback_data = request.form

    for content_id, feedback in feedback_data.items():
        try:
            content_id = int(content_id)
            action = next((idx for idx, content in enumerate(contents) if content["id"] == content_id), None)

            if action is None:
                raise KeyError(f"Content ID {content_id} not found in contents.")

            reward = int(feedback)
            state = contents[action]["embeddings"]

            if reward < 0:
                user_history['disliked'].append(contents[action]["title"])

            # 긍정적인 피드백에 따라 사용자 히스토리 업데이트
            if reward > 0:
                user_history['liked'].append(contents[action]["title"])

        except KeyError as e:
            print(f"Invalid content ID: {content_id}, Error: {e}")
        except Exception as e:
            print(f"Error processing feedback: {e}")

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)