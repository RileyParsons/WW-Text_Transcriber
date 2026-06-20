"""
scraper.py — NSW State Library WW1 Diary Scraper

Downloads paired scanned page images and volunteer transcription text files
from the NSW State Library WW1 Diaries Transcription Project:
    https://transcripts.sl.nsw.gov.au/section/world-war-1-diaries

COMPLIANCE REQUIREMENTS — read before running:
    1. Check robots.txt at the source domain to confirm scraping is permitted.
    2. Review and comply with the NSW State Library website Terms of Use.
    3. Implement polite rate limiting (delay between requests) to avoid
       placing excessive load on the server.
    4. Only download data that is permitted for research/non-commercial use.
    5. If in doubt, request a bulk data dump from the State Library directly
       rather than scraping (contact: ask@sl.nsw.gov.au).

Output:
    - Page images saved to data/pages/
    - Transcription text files saved to data/transcript/
    - pairs.csv updated with each downloaded pair (id, page_image_name,
      page_txt_name, download_source)
"""
