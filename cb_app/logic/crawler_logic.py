# cb_app/logic/crawler_logic.py
import time
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import urllib3

from cb_app.sub_models.webcrawl_models import Page, Paragraph
from cb_app.logic.embedding_model import default_embedder, EMBED_DIM
from cb_app.logic.index_manager import faiss_manager

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[\d+\]", "", text)
    return " ".join(text.split()).strip()


def _embed_paragraph(para: Paragraph, text: str):
    emb = default_embedder.generate_embedding(text)
    if not emb:
        emb = [0.0] * EMBED_DIM
    if len(emb) < EMBED_DIM:
        emb += [0.0] * (EMBED_DIM - len(emb))
    elif len(emb) > EMBED_DIM:
        emb = emb[:EMBED_DIM]
    para.embedding = emb
    para.save(update_fields=["embedding"])
    return emb


def crawl_site(start_url: str, max_pages: int = 20, delay: float = 0.5):
    parsed = urlparse(start_url)
    base_domain = parsed.netloc

    to_visit = [start_url]
    visited = set()

    pages_crawled = 0
    paras_created = 0
    errors = []

    faiss_manager.safe_get_or_create("web_paragraphs")

    session = requests.Session()
    session.headers.update({"User-Agent": "DjangoKB/1.0"})

    while to_visit and pages_crawled < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            resp = session.get(url, timeout=15, verify=False)
        except Exception as e:
            errors.append(f"REQUEST_FAIL {url} -> {e}")
            visited.add(url)
            continue

        if resp.status_code != 200:
            errors.append(f"HTTP_{resp.status_code} {url}")
            visited.add(url)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.get_text().strip() if soup.title else url

        page, _ = Page.objects.get_or_create(url=url, defaults={"title": title})

        created_ids = []
        created_vecs = []

        order = 0
        for p in soup.find_all("p"):
            text = _clean(p.get_text())
            if not text or len(text) < 40:
                continue

            para = Paragraph.objects.create(page=page, text=text, order=order)
            order += 1

            vec = _embed_paragraph(para, text)
            created_ids.append(para.id)
            created_vecs.append(vec)
            paras_created += 1

        # update FAISS safely
        if created_ids:
            try:
                faiss_manager.safe_add("web_paragraphs", created_ids, created_vecs)
            except Exception as e:
                errors.append(f"FAISS_ADD_FAIL {url} -> {e}")

        visited.add(url)
        pages_crawled += 1

        # internal links
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            parsed2 = urlparse(link)
            if parsed2.netloc == base_domain:
                link = link.split("#")[0]
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

        time.sleep(delay)

    return {
        "pages_crawled": pages_crawled,
        "paragraphs_created": paras_created,
        "errors": errors,
    }
