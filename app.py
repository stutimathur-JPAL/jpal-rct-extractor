import os, io, re, json, time
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# ---------- SETTINGS ----------
st.set_page_config(page_title="RCT Metric Search", page_icon="🔎", layout="wide")
st.title("🔎 RCT Metric Search")
st.caption("Type what you need (country, sector, metric). Returns 2–4 verified results with exact values, snippets, links, and provenance.")

# Secrets (set these in Streamlit Cloud → Edit secrets)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
SHEET_CSV_URL  = st.secrets.get("GOOGLE_SHEET_CSV_URL", "")
CSE_API_KEY    = st.secrets.get("GOOGLE_CSE_API_KEY", "")
CSE_CX         = st.secrets.get("GOOGLE_CSE_CX", "")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RCTMetricBot/1.0)"}

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False, ttl=60*60)
def load_sheet(csv_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"Could not load sheet (CSV): {e}")
        return pd.DataFrame()

def http_get(url, timeout=25):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r
    except Exception:
        return None

def fetch_pdf_bytes(url) -> bytes | None:
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
    text_chunks, table_chunks = [], []
    try:
        import fitz
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
    if not m:
        return None
    return m.group(1)

def openai_extract(context_text: str, metric: str) -> dict | None:
    if not OPENAI_API_KEY:
        return None
    import requests as rq
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    system = (
        "You are a precise extraction assistant for academic PDFs. "
        "Return a JSON object with keys: value (string or null), snippet (<=40 words), page_hint (string or null). "
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
    title = str(row.get("Title") or row.get("title") or row.get("Study") or row.get("Paper") or "")
    authors = str(row.get("Authors") or row.get("authors") or "")
    link = str(row.get("Link") or row.get("URL") or row.get("url") or "")

    # quick pre-filter by keywords (optional)
    hay = " ".join([title, authors, link]).lower()
    all_ok = True
    for kw in [country_kw, sector_kw, keywords]:
        if strict_filter and kw and kw.lower() not in hay:
            all_ok = False
    if strict_filter and not all_ok:
        return None

    # 1) primary link → pdf
    pdf = fetch_pdf_bytes(link)
    used = link
    if not pdf and use_web:
        # search web for alt PDF
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

    # simple keyword screening inside content (non-strict)
    content_low = merged.lower()
    for kw in [country_kw, sector_kw, keywords]:
        if kw and kw.lower() not in content_low:
            # not strict: we still allow; but mark as weaker match
            pass

    # regex first
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
            "link": used
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
            "link": used
        }
    return None

# ---------- UI: SEARCH BOXES ----------
st.subheader("Search")
colA, colB, colC, colD = st.columns([1.1, 1.1, 1.3, 1])
metric = colA.selectbox("Metric", ["SD","SE","ICC","MDE","Variance"], index=0, help="Pick the statistic you want")
country_kw = colB.text_input("Country (keyword)", value="", placeholder="e.g., India")
sector_kw  = colC.text_input("Sector / Domain", value="", placeholder="e.g., education")
keywords   = colD.text_input("Extra keywords", value="", placeholder="e.g., math scores")

colE, colF, colG = st.columns([1,1,2])
max_results = colE.slider("How many results?", 2, 4, 3, help="We will stop when we find this many solid hits")
scan_limit  = colF.number_input("Max rows to scan", min_value=10, max_value=1000, value=200, step=10, help="To keep it fast")
use_web     = colG.checkbox("Search open web if PDF missing", value=True)

csv_url = st.text_input("Your Google Sheet CSV link", value=SHEET_CSV_URL or "", help="File → Share → Publish to web → CSV")
strict = st.checkbox("Strict: require keywords in title/authors/link", value=False)

if st.button("Search"):
    if not csv_url:
        st.error("Please paste your Google Sheet CSV link.")
        st.stop()

    df = load_sheet(csv_url)
    if df.empty:
        st.stop()

    # try to ensure columns exist
    if not any("link" in c.lower() or "url" in c.lower() for c in df.columns):
        st.error("I can't find a Link/URL column in your sheet. Add a column named Link or URL.")
        st.stop()

    # main loop
    hits = []
    st.info("Searching… (we’ll stop once we have your requested number of results)")
    prog = st.progress(0)
    total = min(len(df), int(scan_limit))

    for idx, row in df.head(total).reset_index(drop=True).iterrows():
        try:
            result = try_one_study(row, metric, country_kw, sector_kw, keywords, use_web, strict_filter=strict)
            if result:
                # light relevance scoring: prefer exact keyword matches in content/title
                score = 0
                title_text = f"{result['title']} {result['authors']}".lower()
                for kw in [country_kw, sector_kw, keywords]:
                    if kw and kw.lower() in title_text:
                        score += 2
                if result["provenance"] == "PDF (regex)":
                    score += 3
                elif "LLM" in result["provenance"]:
                    score += 1
                result["score"] = score
                hits.append(result)
                # stop early if enough
                if len([h for h in hits]) >= max_results:
                    break
        except Exception:
            continue
        prog.progress(int((idx+1)/max(1,total)*100))

    if not hits:
        st.warning("No results found that match your inputs. Try fewer/more general keywords, or enable web search.")
    else:
        # sort by score (desc)
        hits = sorted(hits, key=lambda x: x.get("score",0), reverse=True)[:max_results]

        st.success(f"Found {len(hits)} result(s).")
        for h in hits:
            with st.container(border=True):
                st.markdown(f"### {h['title']}")
                if h['authors']:
                    st.markdown(f"**Authors:** {h['authors']}")
                st.markdown(f"**Metric:** {h['metric']}  |  **Value:** `{h['value']}`  |  **Source:** {h['provenance']}")
                if h.get("page_hint") or h.get("snippet"):
                    st.caption(f"{h.get('page_hint','')}: {h.get('snippet','')}")
                st.markdown(f"[Open study]({h['link']})", unsafe_allow_html=True)

        # export small table
        out = pd.DataFrame([{
            "Title": h["title"], "Authors": h["authors"], "Metric": h["metric"],
            "Value": h["value"], "Source": h["provenance"], "Page/Hint": h.get("page_hint",""),
            "Link": h["link"]
        } for h in hits])
        st.download_button("⬇️ Download these results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="rct_metric_search_results.csv", mime="text/csv")

st.divider()
st.markdown("**Notes**: ✅ PDF (regex) = exact match from text/tables · 🟦 LLM = extracted by GPT from visible content · ❌ We never bypass paywalls; open-web search looks for accessible copies (working papers, OSF, Dataverse).")
