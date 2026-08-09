import requests
import time
import re
import random
import logging
from typing import Generator, Optional

logger = logging.getLogger(__name__)

WIKI_API = "https://vi.wikipedia.org/w/api.php"

# Wikipedia requires a descriptive User-Agent, else returns empty/blocked responses
HEADERS = {
    "User-Agent": "IRDatasetCrawlerVI/1.0 (research; contact@example.com) python-requests/2.x",
    "Accept": "application/json",
}

# Domains → Wikipedia category seeds → (domain_label, [categories])
DOMAIN_CONFIG = {
    "Khoa học": {
        "color": "#6366f1",
        "categories": [
            "Vật lý học", "Hóa học", "Sinh học", "Toán học",
            "Thiên văn học", "Địa chất học", "Khoa học Trái Đất",
            "Vũ trụ học", "Khoa học máy tính lý thuyết"
        ],
        "queries": [
            "thuyết tương đối là gì", "quang học lượng tử", "cấu trúc AND của nguyên tử",
            "tiến hóa sinh học", "hệ mặt trời", "phương trình vi phân",
            "nhiệt động học", "thuyết tế bào", "gene AND di truyền"
        ]
    },
    "Công nghệ": {
        "color": "#8b5cf6",
        "categories": [
            "Trí tuệ nhân tạo", "Học máy", "Điện toán đám mây",
            "An ninh mạng", "Blockchain", "Internet vạn vật",
            "Robot học", "Lập trình máy tính", "Phần mềm máy tính"
        ],
        "queries": [
            "trí tuệ nhân tạo AND ứng dụng", "machine learning là gì",
            "blockchain hoạt động như thế nào", "bảo mật thông tin mạng",
            "điện toán đám mây", "deep learning AND neural network",
            "lập trình Python", "thuật toán sắp xếp", "cơ sở dữ liệu"
        ]
    },
    "Sức khỏe & Y tế": {
        "color": "#ec4899",
        "categories": [
            "Bệnh truyền nhiễm", "Ung thư", "Tim mạch học",
            "Thần kinh học", "Dinh dưỡng học", "Dược học",
            "Phẫu thuật", "Miễn dịch học", "Nhi khoa"
        ],
        "queries": [
            "bệnh tiểu đường AND triệu chứng", "ung thư phổi điều trị",
            "vaccine hoạt động như thế nào", "cao huyết áp AND nguyên nhân",
            "dinh dưỡng cân bằng", "kháng sinh AND tác dụng phụ",
            "bệnh Alzheimer", "hệ miễn dịch", "COVID-19 AND biến chứng"
        ]
    },
    "Lịch sử": {
        "color": "#f59e0b",
        "categories": [
            "Lịch sử Việt Nam", "Chiến tranh thế giới",
            "Các nền văn minh cổ đại", "Lịch sử châu Á",
            "Cách mạng công nghiệp", "Lịch sử khoa học",
            "Đế quốc và thuộc địa", "Lịch sử Trung Quốc"
        ],
        "queries": [
            "chiến tranh Việt Nam AND nguyên nhân", "đế chế La Mã sụp đổ",
            "cách mạng Pháp 1789", "triều đại nhà Nguyễn",
            "thế chiến thứ hai AND châu Á", "văn minh Ai Cập cổ đại",
            "lịch sử Hà Nội", "nhà Trần AND kháng Mông Nguyên",
            "chiến tranh lạnh AND hệ quả"
        ]
    },
    "Game & Giải trí": {
        "color": "#10b981",
        "categories": [
            "Trò chơi điện tử", "Phát triển trò chơi điện tử",
            "Esports", "Anime", "Điện ảnh Việt Nam",
            "Âm nhạc Việt Nam", "Thể thao điện tử"
        ],
        "queries": [
            "game online AND phổ biến", "esports Việt Nam",
            "phát triển game indie", "trò chơi nhập vai AND cốt truyện",
            "anime Nhật Bản AND thể loại", "điện ảnh Việt Nam hiện đại",
            "game Battle Royale", "game chiến thuật AND lịch sử"
        ]
    },
    "Môi trường": {
        "color": "#14b8a6",
        "categories": [
            "Biến đổi khí hậu", "Năng lượng tái tạo",
            "Bảo tồn thiên nhiên", "Ô nhiễm môi trường",
            "Sinh thái học", "Địa lý học"
        ],
        "queries": [
            "biến đổi khí hậu AND tác động", "năng lượng mặt trời",
            "ô nhiễm không khí AND sức khỏe", "bảo tồn rừng nhiệt đới",
            "năng lượng tái tạo AND Việt Nam", "đa dạng sinh học",
            "hiệu ứng nhà kính", "nước biển dâng AND hậu quả"
        ]
    },
    "Kinh tế & Xã hội": {
        "color": "#f97316",
        "categories": [
            "Kinh tế học", "Tài chính", "Thị trường chứng khoán",
            "Toàn cầu hóa", "Xã hội học", "Giáo dục học"
        ],
        "queries": [
            "lạm phát AND nguyên nhân", "thị trường chứng khoán AND Việt Nam",
            "toàn cầu hóa AND tác động", "GDP AND tăng trưởng kinh tế",
            "giáo dục đại học AND chất lượng", "thất nghiệp AND giải pháp",
            "kinh tế số", "tài chính cá nhân AND đầu tư"
        ]
    }
}


def _wiki_search(query: str, limit: int = 5) -> list:
    """Search Wikipedia for pages matching query."""
    for attempt in range(3):
        try:
            resp = requests.get(WIKI_API, params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
                "utf8": 1,
                "formatversion": 2,
            }, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            if not resp.text.strip():
                time.sleep(1 + attempt)
                continue
            data = resp.json()
            return data.get("query", {}).get("search", [])
        except Exception as e:
            logger.warning(f"Search error for '{query}' (attempt {attempt+1}): {e}")
            time.sleep(1 + attempt)
    return []


def _wiki_category_members(category: str, limit: int = 20) -> list:
    """Get page titles in a category."""
    for attempt in range(3):
        try:
            resp = requests.get(WIKI_API, params={
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Thể loại:{category}",
                "cmlimit": limit,
                "cmtype": "page",
                "format": "json",
                "utf8": 1,
                "formatversion": 2,
            }, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            if not resp.text.strip():
                time.sleep(1 + attempt)
                continue
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
            return [m["title"] for m in members]
        except Exception as e:
            logger.warning(f"Category error for '{category}' (attempt {attempt+1}): {e}")
            time.sleep(1 + attempt)
    return []


def _wiki_page_content(title: str) -> Optional[dict]:
    """Fetch plain text extract of a Wikipedia page."""
    for attempt in range(3):
        try:
            resp = requests.get(WIKI_API, params={
                "action": "query",
                "titles": title,
                "prop": "extracts|info",
                "exintro": 0,
                "explaintext": 1,
                "exsectionformat": "plain",
                "inprop": "url",
                "format": "json",
                "utf8": 1,
                "formatversion": 2,
            }, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            if not resp.text.strip():
                time.sleep(1 + attempt)
                continue
            data = resp.json()
            # formatversion=2 returns pages as list
            pages = data.get("query", {}).get("pages", [])
            if isinstance(pages, list):
                page = pages[0] if pages else {}
            else:
                page = next(iter(pages.values()), {})
            if page.get("missing"):
                return None
            extract = page.get("extract", "").strip()
            url = page.get("fullurl", f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}")
            if len(extract) < 200:
                return None
            return {
                "title": page.get("title", title),
                "content": extract,
                "url": url
            }
        except Exception as e:
            logger.warning(f"Page fetch error for '{title}' (attempt {attempt+1}): {e}")
            time.sleep(1 + attempt)
    return None


def _clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{3,}', ' ', text)
    return text.strip()


def _extract_snippet(content: str, max_chars: int = 800) -> str:
    """Take first meaningful paragraph up to max_chars."""
    paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 80]
    result = ""
    for p in paragraphs:
        if len(result) + len(p) <= max_chars:
            result += p + "\n\n"
        else:
            break
    return result.strip() or content[:max_chars]


def _make_query_from_title(title: str, domain_queries: list[str]) -> str:
    """Pick or adapt a query string for a document."""
    # 30% chance pick from pre-defined queries
    if random.random() < 0.3 and domain_queries:
        return random.choice(domain_queries)
    # Otherwise generate simple keyword query
    words = title.split()
    if len(words) <= 4:
        return title
    return " AND ".join(words[:3])


def crawl_all(target: int = 1500, progress_cb=None,
              existing_titles: set = None) -> Generator[dict, None, None]:
    """
    Crawl Wikipedia Vietnamese articles across all domains.
    Yields record dicts and calls progress_cb(current, total, msg) if provided.
    existing_titles: set of titles already crawled (for resume deduplication).
    """
    seen_titles: set[str] = set(existing_titles or [])
    collected = 0

    # Build a flat work list: (domain, query_list, source_type, value)
    work_items = []
    for domain, cfg in DOMAIN_CONFIG.items():
        per_domain = target // len(DOMAIN_CONFIG)
        queries = cfg["queries"]
        categories = cfg["categories"]

        # From category members
        for cat in categories:
            work_items.append(("cat", domain, queries, cat))
        # From search queries
        for q in queries:
            work_items.append(("search", domain, queries, q))

    random.shuffle(work_items)

    for item in work_items:
        if collected >= target:
            break

        src_type, domain, domain_queries, value = item

        if src_type == "cat":
            titles = _wiki_category_members(value, limit=30)
            time.sleep(0.3)
        else:
            results = _wiki_search(value, limit=8)
            titles = [r["title"] for r in results]
            time.sleep(0.3)

        for title in titles:
            if collected >= target:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)

            page = _wiki_page_content(title)
            if not page:
                time.sleep(0.2)
                continue

            content_clean = _clean_text(page["content"])
            snippet = _extract_snippet(content_clean)
            query = _make_query_from_title(title, domain_queries)

            record = {
                "id": collected + 1,
                "domain": domain,
                "title": page["title"],
                "query": query,
                "document": snippet,
                "full_content": content_clean[:3000],
                "url": page["url"],
                "word_count": len(content_clean.split()),
                "relevance": _estimate_relevance(query, snippet),
            }

            collected += 1
            if progress_cb:
                progress_cb(collected, target, f"[{domain}] {title}")

            yield record
            time.sleep(0.15)


def _estimate_relevance(query: str, doc: str) -> int:
    """
    Rough relevance score 1-3 based on keyword overlap.
    Used as initial label for relevant feedback dataset.
    1=Not relevant, 2=Partially relevant, 3=Highly relevant
    """
    q_terms = set(re.sub(r'\bAND\b', '', query).lower().split())
    doc_lower = doc.lower()
    hits = sum(1 for t in q_terms if t in doc_lower and len(t) > 2)
    ratio = hits / max(len(q_terms), 1)
    if ratio >= 0.6:
        return 3
    elif ratio >= 0.3:
        return 2
    return 1
