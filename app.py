import os, io, re, json
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="RCT Metric Search", page_icon="🔎", layout="wide")
st.title("🔎 RCT Metric Search")
st.caption("Search by country/sector/keywords. Returns 2–4 verified hits with exact values, snippets, links, and provenance.")

# ---------- SECRETS (BACKEND ONLY) ----------
# Add these in Streamlit Cloud → (⋯) → Edit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
SHEET_CSV_URL  = st.secrets.get("GOOGLE_SHEET_CSV_URL", "")  # your hidden CSV link lives here
CSE_API_KEY    = st.secrets.get("GOOGLE_CSE_API_KEY", "")    # optional (for open-web fallback)
CSE_CX         = st.secrets.get("GOOGLE_CSE_CX", "")         # optional (for open-web fallback)

# Optional hard fallback (only used if you forgot to add the secret)
# You can delete the line below after you set the secret.
if not SHEET_CSV_URL:
    SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTympLPFY-ummEwgcCkD5sGAvm9zy7TrKuCD6lqtLt6kcY5-xk0Rr_hOkd9rkTTe5EqRQBortUKqxCI/pub?output=csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RCTMetricSearch/1.0)"}

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False, ttl=60*60)
def load_sheet(csv_url: str) -> pd.DataFrame:
    """Load the published CSV (no UI exposure)."""
    try:
        return pd.read_csv(csv_url)
    except Exception:
        st.error("Could not load the studies list. Please check the backend secret GOOGLE_SHEET_CSV_URL.")
        return pd.DataFrame()

def http_get(url, timeout=25):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r
    except Exception:
        return None

def fetch_pdf_bytes(url) -> bytes | None:
    """Try to fetch a direct PDF; if HTML, look for a .pdf link on the page."""
    r = http_get(url)
    if not r:
        return None
    ct = r.headers.get("Content-Type","").lower()
    if "pdf" in ct or url.lower().endswith(".pdf"):
        return r.content
    if "html" in ct:
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("a[href]"):
            href = a["href"]
            if ".pdf" in href.lower():
                pdf_url = urljoin(url, href)
                rr = http_get(pdf_url)
                if rr and "pdf" in rr.headers.get("Content-Type","").lower():
                    return rr.content
    return None

def pdf_to_text_and_tables(pdf_bytes: bytes):
    """Extract visible text and table text (no OCR to keep it fast)."""
    text_chunks, table_chunks = [], []
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                txt = page.get_text("text") or ""
                if txt.strip():
                    text_chunks.append(f"[PAGE {i}] {txt}")
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for pnum, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    for ti, tb in enumerate(tables or []):
                        rows = ["\t".join([(c or "") for c in row]) for row in tb]
                        table_chunks.append(f"[PAGE {pnum} TABLE {ti+1}]\n" + "\n".join(rows))
                except Exception:
                    continue
    except Exception:
        pass
    return "\n".join(text_chunks), "\n\n".join(table_chunks)

# Metric regexes
NUM = r"(?:-?\d+(?:\.\d+)?)"
PCT = r"(?:-?\d+(?:\.\d+)?\s?%)"
METRIC_PATTERNS = {
    "SD": rf"(?:\bSD\b|standard deviation)\D{{0,20}}({NUM})",
    "SE": rf"(?:\bSE\b|standard error)\D{{0,20}}({NUM})",
    "ICC": rf"(?:\bICC\b|intra[- ]?cluster.*?correlation)\D{{0,20}}({NUM})",
    "MDE": rf"(?:\bMDE\b|minimum detectable effect)\D{{0,30}}({NUM}|{PCT})",
    "Variance": rf"(?:\bvariance\b)\D{{0,20}}({NUM})",
}

def regex_extract(text: str, which_metric: str):
    pat = METRIC_PATTERNS.get(which_metric)
    if not pat:
        return None
    m = re.search(pat, text, flags=re.I)
    return m.group(1) if m else None

def openai_extract(context_text: str, metric: str) -> dict | None:
    """Structured extraction via ChatGPT API (no guessing)."""
    if not OPENAI_API_KEY:
        return None
    import requests as rq
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    system = (
        "You are a precise extraction assistant for academic PDFs. "
        "Return JSON with: value (string or null), snippet (<=40 words), page_hint (string or null). "
        "Extract ONLY if explicitly present. Do NOT guess."
    )
    user = f"Metric to extract: {metric}\n\nText:\n{context_text[:15000]}"
    body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
    }
    try:
        r = rq.post(url, headers=headers, json=body, timeout=70)
        r.raise_for_status()
        s = r.json()["choices"][0]["message"]["content"]
        return json.loads(s)
    except Exception:
        return None

def google_cse_search(query: str, num=3):
    if not (CSE_API_KEY and CSE_CX):
        return []
    params = {"key": CSE_API_KEY, "cx": CSE_CX, "q": query, "num": num}
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=25)
    if r.status_code != 200:
        return []
    return [{"title": it.get("title"), "link": it.get("link")} for it in r.json().get("items", [])]

def try_one_study(row, metric, country_kw, sector_kw, keywords, use_web, strict_filter=False):
    # column inference (handles many naming styles)
    title = str(
        row.get("Title")
        or row.get("title")
        or row.get("Study")
        or row.get("Study Title")
        or row.get("Paper")
        or row.get("Name of Study")
        or row.get("name of study")
        or ""
    )
    authors = str(row.get("Authors") or row.get("authors") or "")
    link = str(row.get("Link") or row.get("URL") or row.get("url") or "")

    # pre-filter on visible metadata if strict
    hay = " ".join([title, authors, link]).lower()
    if strict_filter:
        for kw in [country_kw, sector_kw, keywords]:
            if kw and kw.lower() not in hay:
                return None

    # try primary link
    pdf = fetch_pdf_bytes(link)
    used = link

    # web fallback (optional)
    if not pdf and use_web:
        q = f'{title} {authors} filetype:pdf'
        for hit in google_cse_search(q, num=3):
            pdf = fetch_pdf_bytes(hit["link"])
            if pdf:
                used = hit["link"]
                break
    if not pdf:
        return None

    full_text, tables_text = pdf_to_text_and_tables(pdf)
    merged = (tables_text + "\n\n" + full_text)[:18000]

    # regex first (exact)
    val = regex_extract(merged, metric)
    if val:
        return {
            "title": title or "(Untitled)",
            "authors": authors,
            "metric": metric,
            "value": val,
            "provenance": "PDF (regex)",
            "snippet": "",
            "page_hint": "",
            "link": used,
        }

    # LLM fallback
    ext = openai_extract(merged, metric) or {}
    if ext.get("value"):
        return {
            "title": title or "(Untitled)",
            "authors": authors,
            "metric": metric,
            "value": ext["value"],
            "provenance": "PDF (LLM)" if used == link else "Web (LLM)",
            "snippet": ext.get("snippet",""),
            "page_hint": ext.get("page_hint",""),
            "link": used,
        }
    return None

# ---------- UI: SEARCH BOXES ----------
st.subheader("Search")
colA, colB, colC, colD = st.columns([1.1, 1.1, 1.3, 1])
metric     = colA.selectbox("Metric", ["SD","SE","ICC","MDE","Variance"], index=0)
country_kw = colB.text_input("Country", value="", placeholder="e.g., India")
sector_kw  = colC.text_input("Sector / Domain", value="", placeholder="e.g., education")
keywords   = colD.text_input("Extra keywords", value="", placeholder="e.g., math scores")

colE, colF, colG = st.columns([1,1,2])
max_results = colE.slider("Results to return", 2, 4, 3)
scan_limit  = colF.number_input("Max rows to scan", min_value=10, max_value=1000, value=200, step=10)
use_web     = colG.checkbox("Try open web if PDF missing", value=True)
strict      = st.checkbox("Strict: require keywords in title/authors/link", value=False)

if st.button("Search"):
    df = load_sheet(SHEET_CSV_URL)
    if df.empty:
        st.stop()

    # Ensure at least a Link/URL column exists
    if not any("link" in c.lower() or "url" in c.lower() for c in df.columns):
        st.error("Your sheet must include a Link/URL column.")
        st.stop()

    hits = []
    st.info("Searching… (we’ll stop after we collect enough results)")
    prog = st.progress(0)
    total = min(len(df), int(scan_limit))

    for idx, row in df.head(total).reset_index(drop=True).iterrows():
        try:
            res = try_one_study(row, metric, country_kw, sector_kw, keywords, use_web, strict_filter=strict)
            if res:
                # simple relevance score
                score = 0
                title_text = f"{res['title']} {res['authors']}".lower()
                for kw in [country_kw, sector_kw, keywords]:
                    if kw and kw.lower() in title_text:
                        score += 2
                if res["provenance"] == "PDF (regex)":
                    score += 3
                elif "LLM" in res["provenance"]:
                    score += 1
                res["score"] = score
                hits.append(res)
                if len(hits) >= max_results:
                    break
        except Exception:
            continue
        prog.progress(int((idx+1)/max(1,total)*100))

    if not hits:
        st.warning("No results matched. Try fewer/more general keywords, or disable Strict.")
    else:
        hits = sorted(hits, key=lambda x: x.get("score",0), reverse=True)[:max_results]
        st.success(f"Found {len(hits)} result(s).")
        for h in hits:
            with st.container(border=True):
                # >>> NAME OF STUDY SHOWN HERE <<<
                st.markdown(f"### {h['title']}")
                if h['authors']:
                    st.markdown(f"**Authors:** {h['authors']}")
                st.markdown(f"**Metric:** {h['metric']}  |  **Value:** `{h['value']}`  |  **Source:** {h['provenance']}")
                if h.get("page_hint") or h.get("snippet"):
                    st.caption(f"{h.get('page_hint','')}: {h.get('snippet','')}")
                st.markdown(f"[Open study]({h['link']})", unsafe_allow_html=True)

        out = pd.DataFrame([{
            "Title": h["title"], "Authors": h["authors"], "Metric": h["metric"],
            "Value": h["value"], "Source": h["provenance"], "Page/Hint": h.get("page_hint",""),
            "Link": h["link"]
        } for h in hits])
        st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="rct_metric_search_results.csv", mime="text/csv")
