import os
import io
import time
import re
import argparse
import requests
from collections import deque
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://www.iitj.ac.in"

SEED_URLS = [
    "https://www.iitj.ac.in/m/Index/main-departments?lg=en",
    "https://www.iitj.ac.in/chemical-engineering/",
    "https://www.iitj.ac.in/bioscience-bioengineering",
    "https://www.iitj.ac.in/chemistry/en/chemistry",
    "https://www.iitj.ac.in/computer-science-engineering/",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/",
    "https://www.iitj.ac.in/electrical-engineering/",
    "https://www.iitj.ac.in/mathematics/",
    "https://www.iitj.ac.in/mechanical-engineering/",
    "https://www.iitj.ac.in/materials-engineering/en/materials-engineering",
    "https://www.iitj.ac.in/physics/",
    "https://www.iitj.ac.in/chemical-engineering/en/undergraduate-program",
    "https://www.iitj.ac.in/chemical-engineering/en/postgraduate-program",
    "https://www.iitj.ac.in/chemical-engineering/en/doctoral-program",
    "https://www.iitj.ac.in/chemical-engineering/en/curriculum",
    "https://www.iitj.ac.in/chemical-engineering/en/courses",
    "https://www.iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://www.iitj.ac.in/office-of-research-development/en/information",
    "https://www.iitj.ac.in/main/en/faculty-members",
    "https://www.iitj.ac.in/main/en/why-pursue-a-career-@-iit-jodhpur",
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://iitj.ac.in/Office-of-Academics/en/Academic-Calendar",
    "https://www.iitj.ac.in/",
]

REQUEST_TIMEOUT = 15
SLEEP_BETWEEN   = 0.8
MAX_RETRIES     = 3
MAX_PAGES       = 500
MAX_DEPTH       = 4
MAX_PDF_SIZE_MB = 20

CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "td", "th", "blockquote", "article", "section"}

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
PDF_DIR     = os.path.join(DATA_DIR, "pdfs")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_corpus.txt")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IITJResearchScraper/1.0; +https://www.iitj.ac.in)"
}


def _extract_pdf_text_pdfminer(pdf_bytes):
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    output = io.StringIO()
    extract_text_to_fp(io.BytesIO(pdf_bytes), output, laparams=LAParams(),
                       output_type="text", codec="utf-8")
    return output.getvalue()


def _extract_pdf_text_pypdf2(pdf_bytes):
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def extract_pdf_text(pdf_bytes):
    for extractor in (_extract_pdf_text_pdfminer, _extract_pdf_text_pypdf2):
        try:
            text = extractor(pdf_bytes)
            if text and text.strip():
                return text
        except Exception:
            continue
    return ""


def _is_same_domain(url):
    return urlparse(url).netloc.endswith("iitj.ac.in")


def _is_pdf_url(url):
    return urlparse(url).path.lower().endswith(".pdf")


def _clean_text(raw):
    raw = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _looks_english(text):
    if not text:
        return False
    return sum(1 for ch in text if ord(ch) > 127) / len(text) < 0.20


def _safe_filename(url):
    name = os.path.basename(urlparse(url).path)
    name = re.sub(r"[^\w.\-]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def fetch_page(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            print(f"  [HTTP {resp.status_code}] {url}")
            return None
        except requests.exceptions.RequestException as exc:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] {url}: {exc}")
            time.sleep(SLEEP_BETWEEN * attempt)
    return None


def fetch_binary(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
            if resp.status_code != 200:
                print(f"  [HTTP {resp.status_code}] {url}")
                return None
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_PDF_SIZE_MB * 1024 * 1024:
                print(f"  [SKIP too large] {url}")
                return None
            chunks, total = [], 0
            for chunk in resp.iter_content(chunk_size=65536):
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_PDF_SIZE_MB * 1024 * 1024:
                    print(f"  [SKIP exceeded {MAX_PDF_SIZE_MB}MB] {url}")
                    return None
            return b"".join(chunks)
        except requests.exceptions.RequestException as exc:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] {url}: {exc}")
            time.sleep(SLEEP_BETWEEN * attempt)
    return None


def parse_page(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    sentences = []
    for tag in soup.find_all(CONTENT_TAGS):
        cleaned = _clean_text(tag.get_text(separator=" "))
        if len(cleaned) > 30 and _looks_english(cleaned):
            sentences.append(cleaned)

    child_urls, pdf_urls = [], []
    for a_tag in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a_tag["href"].strip()).split("#")[0]
        if not _is_same_domain(absolute):
            continue
        if _is_pdf_url(absolute):
            pdf_urls.append(absolute)
        else:
            child_urls.append(absolute)

    return sentences, child_urls, pdf_urls


def pdf_text_to_sentences(raw_text):
    if not raw_text:
        return []
    text   = _clean_text(raw_text)
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [c.strip() for c in chunks if len(c.strip()) > 30 and _looks_english(c)]


class IITJScraper:
    def __init__(self, download_pdfs=True):
        self.download_pdfs  = download_pdfs
        self.visited_html   = set()
        self.visited_pdfs   = set()
        self.all_sentences  = []
        self.pdf_sentences  = []
        self.pdf_count      = 0
        self.pdf_fail_count = 0

    def crawl(self):
        queue = deque((url, 0) for url in SEED_URLS)
        print(f"Starting crawl from {len(SEED_URLS)} seed URLs ...")
        print(f"  Max pages: {MAX_PAGES}  |  Max depth: {MAX_DEPTH}  |  PDF: {self.download_pdfs}\n")

        pending_pdfs = []

        while queue and len(self.visited_html) < MAX_PAGES:
            url, depth = queue.popleft()
            if url in self.visited_html or not _is_same_domain(url):
                continue
            self.visited_html.add(url)
            print(f"  [HTML {len(self.visited_html):>4}/{MAX_PAGES}] depth={depth}  {url}")

            html = fetch_page(url)
            time.sleep(SLEEP_BETWEEN)
            if not html or len(html) < 200:
                continue

            sentences, child_urls, pdf_urls = parse_page(html, url)
            self.all_sentences.extend(sentences)

            if depth < MAX_DEPTH:
                for child in child_urls:
                    if child not in self.visited_html:
                        queue.append((child, depth + 1))

            for pdf_url in pdf_urls:
                if pdf_url not in self.visited_pdfs:
                    self.visited_pdfs.add(pdf_url)
                    pending_pdfs.append(pdf_url)

        print(f"\nHTML crawl complete: {len(self.visited_html)} pages, "
              f"{len(self.all_sentences)} sentences, {len(pending_pdfs)} PDFs\n")

        if self.download_pdfs and pending_pdfs:
            os.makedirs(PDF_DIR, exist_ok=True)
            self._process_pdfs(pending_pdfs)

    def _process_pdfs(self, pdf_urls):
        total = len(pdf_urls)
        print(f"Processing {total} PDFs -> {PDF_DIR}\n")

        for i, url in enumerate(pdf_urls, start=1):
            save_path = os.path.join(PDF_DIR, _safe_filename(url))
            print(f"  [PDF {i:>3}/{total}] {url}")

            if os.path.exists(save_path):
                with open(save_path, "rb") as f:
                    pdf_bytes = f.read()
                print("    Cached.")
            else:
                pdf_bytes = fetch_binary(url)
                time.sleep(SLEEP_BETWEEN)
                if pdf_bytes is None:
                    print("    Download failed.")
                    self.pdf_fail_count += 1
                    continue
                with open(save_path, "wb") as f:
                    f.write(pdf_bytes)
                print(f"    Saved ({len(pdf_bytes)//1024} KB)")

            try:
                sentences = pdf_text_to_sentences(extract_pdf_text(pdf_bytes))
                self.pdf_sentences.extend(sentences)
                self.pdf_count += 1
                print(f"    Extracted {len(sentences)} sentences.")
            except Exception as exc:
                print(f"    Extraction failed: {exc}")
                self.pdf_fail_count += 1

        print(f"\nPDF phase: {self.pdf_count} OK, {self.pdf_fail_count} failed, "
              f"{len(self.pdf_sentences)} sentences\n")

    def save(self, path=OUTPUT_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_text = self.all_sentences + self.pdf_sentences
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# HTML sentences : {len(self.all_sentences)}\n")
            f.write(f"# PDF  sentences : {len(self.pdf_sentences)}\n")
            f.write(f"# Total          : {len(all_text)}\n")
            for sentence in all_text:
                f.write(sentence + "\n")
        print(f"Corpus saved: {path}  (HTML: {len(self.all_sentences):,}, "
              f"PDF: {len(self.pdf_sentences):,}, Total: {len(all_text):,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl IIT Jodhpur website.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF downloading.")
    args = parser.parse_args()

    scraper = IITJScraper(download_pdfs=not args.no_pdf)
    scraper.crawl()
    scraper.save()
