"""
scraper.py — NSW State Library WW1 Diary Scraper

Downloads paired scanned page images and volunteer transcription text files
from the NSW State Library WW1 Diaries Transcription Project:
    https://transcripts.sl.nsw.gov.au/section/world-war-1-diaries

COMPLIANCE:
    - robots.txt is checked at runtime before every request
    - A 10-second crawl delay is enforced between requests (required by robots.txt)
    - Only public content pages (/section/, /document/, /page/) are accessed
    - Disallowed paths (/admin/, /search/, etc.) are never requested

SKIP LOGIC:
    - Pages with status "Not yet started" or transcription containing "not transcribed"
      are skipped — only fully transcribed pages are downloaded

RESUME SUPPORT:
    - Already-downloaded pages (tracked in pairs.csv) are skipped on re-run,
      allowing the scraper to continue after an interruption

Output:
    - Page images saved to data/pages/
    - Transcription text files saved to data/transcript/
    - pairs.csv updated with each downloaded pair (id, page_image_name,
      page_txt_name, download_source)
    - scraper.log records all progress, warnings, and errors
"""

import csv
import logging
import re
import time
from pathlib import Path
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL    = "https://transcripts.sl.nsw.gov.au"
SECTION_URL = f"{BASE_URL}/section/world-war-1-diaries"
CRAWL_DELAY = 10           # seconds between requests — required by robots.txt
PAGES_DIR   = Path("data/pages")
TRANS_DIR   = Path("data/transcript")
PAIRS_CSV   = Path("data/pairs.csv")
LOG_FILE    = Path("scraper.log")


# ── Logging setup ────────────────────────────────────────────────────────────

# Write log messages to both the console and scraper.log so progress is visible
# live and a full history is kept on disk for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # persistent log file
        logging.StreamHandler(),                           # live console output
    ],
)
log = logging.getLogger(__name__)


# ── robots.txt compliance ────────────────────────────────────────────────────

def build_robot_parser() -> RobotFileParser:
    """Fetch and parse robots.txt once so every URL can be checked before fetching."""
    rp = RobotFileParser()
    rp.set_url(f"{BASE_URL}/robots.txt")
    rp.read()
    log.info("robots.txt loaded from %s/robots.txt", BASE_URL)
    return rp


def is_allowed(rp: RobotFileParser, url: str) -> bool:
    """Return True if robots.txt permits fetching this URL as a generic crawler."""
    allowed = rp.can_fetch("*", url)
    if not allowed:
        log.warning("robots.txt DISALLOWS: %s — skipping", url)
    return allowed


# ── HTTP helpers ─────────────────────────────────────────────────────────────

# Reuse a single session across all requests for connection pooling
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "WW-Text-Transcriber research scraper (rrparsons01@gmail.com)"
})


def fetch(url: str, rp: RobotFileParser) -> BeautifulSoup | None:
    """
    Fetch a URL and return a BeautifulSoup object for HTML parsing.
    Returns None if the URL is disallowed by robots.txt or the request fails.
    Always sleeps CRAWL_DELAY seconds after the request to respect the server.
    """
    if not is_allowed(rp, url):
        return None
    try:
        response = SESSION.get(url, timeout=30)
        response.raise_for_status()
        time.sleep(CRAWL_DELAY)   # honour the 10-second crawl delay from robots.txt
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as exc:
        log.error("Failed to fetch %s: %s", url, exc)
        time.sleep(CRAWL_DELAY)   # still wait before the next request even on failure
        return None


def download_file(url: str, dest: Path, rp: RobotFileParser) -> bool:
    """
    Download a binary file (scanned image) to dest path.
    Returns True on success, False if disallowed or the download fails.
    """
    if not is_allowed(rp, url):
        return False
    try:
        response = SESSION.get(url, timeout=60, stream=True)
        response.raise_for_status()
        dest.write_bytes(response.content)
        time.sleep(CRAWL_DELAY)
        return True
    except requests.RequestException as exc:
        log.error("Failed to download %s: %s", url, exc)
        time.sleep(CRAWL_DELAY)
        return False


# ── Image URL normalisation ──────────────────────────────────────────────────

def full_res_image_url(thumbnail_url: str) -> str:
    """
    Convert a thumbnail S3 URL to the full-resolution version.
    Drupal image styles inject '/styles/<style-name>/public/' into the S3 path;
    removing this segment gives the original full-resolution file URL.
    Example:
      .../styles/page_list_style/public/files/transcript_image/foo.jpg
      → .../files/transcript_image/foo.jpg
    """
    return re.sub(r"/styles/[^/]+/public/", "/", thumbnail_url)


# ── Pagination helper ────────────────────────────────────────────────────────

def paginated_pages(start_url: str, rp: RobotFileParser):
    """
    Generator that yields (BeautifulSoup, current_url) for each page of a
    paginated listing, following 'next ›' links until none remain.
    Used for both the diary list and individual diary page lists.
    """
    url = start_url
    while url:
        soup = fetch(url, rp)
        if soup is None:
            break
        yield soup, url

        # Look for the 'next' pagination link to continue to the next page
        next_link = soup.find("a", string=lambda t: t and "next" in t.lower())
        if next_link and next_link.get("href"):
            url = urljoin(BASE_URL, next_link["href"])
        else:
            break   # no next page — we've reached the end of this listing


# ── Resume support ───────────────────────────────────────────────────────────

def load_downloaded_urls() -> set:
    """
    Read pairs.csv and return a set of download_source URLs already processed.
    Allows the scraper to skip completed pages when restarted after interruption.
    """
    seen = set()
    if PAIRS_CSV.exists():
        with PAIRS_CSV.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen.add(row["download_source"])
    log.info("Resume: %d pages already in pairs.csv — will skip these", len(seen))
    return seen


def next_row_id() -> int:
    """
    Return the next sequential ID for pairs.csv by counting existing data rows.
    The header row is not counted.
    """
    if not PAIRS_CSV.exists():
        return 1
    with PAIRS_CSV.open(newline="", encoding="utf-8") as f:
        # Subtract 1 to exclude the header row from the count
        return max(1, sum(1 for _ in f) - 1 + 1)


# ── pairs.csv writer ─────────────────────────────────────────────────────────

def append_pair(row_id: int, image_name: str, txt_name: str, source_url: str) -> None:
    """Append one completed image/transcript pair to pairs.csv."""
    with PAIRS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_id, image_name, txt_name, source_url])


# ── Per-page processing ──────────────────────────────────────────────────────

def process_page(page_url: str, rp: RobotFileParser, seen: set, current_id: list) -> None:
    """
    Fetch a single diary page, check its transcription status, and if transcribed:
      1. Download the full-resolution scanned image to data/pages/
      2. Write the transcription text to data/transcript/
      3. Append a row to pairs.csv

    current_id is a single-element list [int] so it can be mutated across calls.
    """
    # Skip pages already downloaded in a previous run to support resuming
    if page_url in seen:
        log.info("SKIP (already downloaded): %s", page_url)
        return

    soup = fetch(page_url, rp)
    if soup is None:
        return

    # Derive a safe filename from the last URL path segment (the page slug)
    slug = page_url.rstrip("/").split("/")[-1]

    # Check the full page text for skip conditions — if the page is not yet
    # transcribed, there's nothing to download
    page_text = soup.get_text(" ", strip=True).lower()
    if "not yet started" in page_text or "not transcribed" in page_text:
        log.info("SKIP (not transcribed): %s", slug)
        return

    # Extract transcription text from the section after the "Transcription" heading
    transcript_text = ""
    trans_heading = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and "transcription" in tag.get_text().lower()
    )
    if trans_heading:
        # Collect all sibling content until the next heading starts a new section
        for sibling in trans_heading.find_next_siblings():
            if sibling.name in ("h2", "h3", "h4"):
                break
            transcript_text += sibling.get_text(" ", strip=True) + "\n"

    if not transcript_text.strip():
        log.warning("No transcription text found at %s — skipping", slug)
        return

    # Find the scanned image and convert its thumbnail URL to full resolution
    img_tag = soup.find("img", src=re.compile(r"transcript_image", re.I))
    if not img_tag:
        log.warning("No image tag found at %s — skipping", slug)
        return
    image_url = full_res_image_url(img_tag["src"])

    # Build output file paths using the slug as the filename
    image_name = f"{slug}.jpg"
    txt_name   = f"{slug}.txt"
    image_path = PAGES_DIR / image_name
    txt_path   = TRANS_DIR / txt_name

    # Download the scanned image; abort the whole pair if the download fails
    if not download_file(image_url, image_path, rp):
        log.error("Image download failed for %s — skipping pair", slug)
        return

    # Write the transcription text to a .txt file
    txt_path.write_text(transcript_text.strip(), encoding="utf-8")

    # Record the pair in pairs.csv and increment the ID counter for the next pair
    row_id = current_id[0]
    append_pair(row_id, image_name, txt_name, page_url)
    current_id[0] += 1

    log.info("DOWNLOADED [id=%d]: %s", row_id, slug)


# ── Diary traversal ──────────────────────────────────────────────────────────

def scrape_diary(diary_url: str, rp: RobotFileParser, seen: set, current_id: list) -> None:
    """
    Traverse all paginated page listings within a single diary document
    and process each individual scanned page.
    """
    log.info("── Diary: %s", diary_url)
    for soup, _ in paginated_pages(diary_url, rp):
        # Collect links to individual scanned pages within this diary
        page_links = soup.find_all("a", href=re.compile(r"^/page/"))
        for link in page_links:
            page_url = urljoin(BASE_URL, link["href"])
            process_page(page_url, rp, seen, current_id)


# ── Main entry point ─────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: verify compliance, prepare output directories, then traverse
    all ~959 diaries across the NSW State Library WW1 archive.
    """
    log.info("=== WW-Text-Transcriber scraper started ===")

    # Create output directories if they don't already exist
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRANS_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch robots.txt once and reuse for all compliance checks
    rp = build_robot_parser()

    # Load already-downloaded source URLs to enable resume after interruption
    seen = load_downloaded_urls()

    # Use a single-element list for the row ID counter so it can be mutated
    # inside nested function calls without needing a global variable
    current_id = [next_row_id()]

    # Walk through every page of the diary listing and process each diary found
    diary_count = 0
    for soup, _ in paginated_pages(SECTION_URL, rp):
        # Find all diary document links on this listing page
        diary_links = soup.find_all("a", href=re.compile(r"^/document/"))

        # Deduplicate links — the same diary may appear multiple times on a page
        seen_diaries = set()
        for link in diary_links:
            diary_url = urljoin(BASE_URL, link["href"])
            if diary_url not in seen_diaries:
                seen_diaries.add(diary_url)
                scrape_diary(diary_url, rp, seen, current_id)
                diary_count += 1
                log.info("Diaries processed: %d | Pairs downloaded: %d",
                         diary_count, current_id[0] - 1)

    log.info("=== Scraper complete. Diaries: %d | Total pairs: %d ===",
             diary_count, current_id[0] - 1)


if __name__ == "__main__":
    main()
