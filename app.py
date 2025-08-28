import os, io, re, json
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# PDF + tables
import fitz  # PyMuPDF
import pdfplumber

# ---------- read secrets ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
SHEET_CSV_URL  = st.secrets.get("GOOGLE_SHEET_CSV_URL", "")
CSE_API_KEY    = st.secrets.get("GOOGLE_CSE_API_KEY", "")
CSE_CX         = st.secrets.get("GOOGLE_CSE_CX", "")

st.set_page_config(page_title="RCT Metric Extractor", page_icon="📊", layout="wide")
st.title("📊 RCT Metric Extractor")
st.caption("Pull SD, SE, ICC, MDE, Variance from PDFs. If missing, try the open web and GPT. Always label the source.")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RCTMetricBot/1.0)"}

# ---------- helpers ----------
def load_sheet(csv_url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Could not load sheet: {e}")
        return pd.DataFrame()

def http_get(url, timeout=30):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        return None

def fetch_pdf_bytes(url) -> bytes | None:
    r = http_get(url)
    if not r:
        return None
    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return r.content
    if "html" in ctype:
        soup = BeautifulSoup(r.text, "lxml")
        # look for a link ending with .pdf
        for a in soup.select("a[href]"):
            href = a["href"]
            if ".pdf" in href.lower():
                pdf_url = urljoin(url, href)
                rr = http_get(pdf_url)
                if rr and "pdf" in rr.headers.get("Content-Type","").lower():
                    return rr.content
    return None

def pdf_to_text_and_tables(pdf_bytes: bytes):
    text_chunks = []
    # text
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    text_chunks.append(f"[PAGE {i}] {text}")
    except Exception:
        pass
    # tables
    table_chunks = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for pnum, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    for ti, tb in enumerate(tables or []):
                        rows = ["\t".join([c if c else "" for c in row]) for row in tb]
                        table_text = f"[PAGE {pnum} TABLE {ti+1}]\n" + "\n".join(rows)
                        table_chunks.append(table_text)
                except Exception:
                    continue
    except Exception:
        pass
    full_text = "\n".join(text_chunks)
    tables_text = "\n\n".join(table_chunks)
    return full_text, tables_text

NUM = r"(?:-?\d+(?:\.\d+)?)"
PCT = r"(?:-?\d+(?:\.\d+)?\s?%)"

def regex_find_metrics(text: str):
    OUT = []
    for m in re.finditer(rf"(?:ICC|intra[- ]?cluster.*?correlation)\D{{0,20}}({NUM})", text, flags=re.I):
        OUT.append(("ICC", m.group(1), "regex"))
    for m in re.finditer(rf"(?:SD|standard deviation)\D{{0,20}}({NUM})", text, flags=re.I):
        OUT.append(("SD", m.group(1), "regex"))
    for m in re.finditer(rf"(?:SE|standard error)\D{{0,20}}({NUM})", text, flags=re.I):
        OUT.append(("SE", m.group(1), "regex"))
    for m in re.finditer(rf"(?:variance)\D{{0,20}}({NUM})", text, flags=re.I):
        OUT.append(("Variance", m.group(1), "regex"))
    for m in re.finditer(rf"(?:MDE|minimum detectable effect)\D{{0,20}}({NUM}|{PCT})", text, flags=re.I):
        OUT.append(("MDE", m.group(1), "regex"))
    return OUT

def openai_structured_extract(context_text: str, title: str = "", authors: str = "", link: str = "") -> dict | None:
    if not OPENAI_API_KEY:
        return None
    import requests as rq
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    system = (
        "You are a precise extraction assistant. "
        "Extract ONLY if explicitly present. Do NOT guess. "
        "Return JSON with keys sd, se, icc, mde, variance (numbers or strings or null), "
        "and 'citations' as a list of {metric, snippet, page_hint}."
    )
    user = f"""
Study: {title}
Authors: {authors}
Link: {link}

Extract SD, SE, ICC, MDE, Variance ONLY if explicitly present in this text. If not, use null.
Text begins:
{context_text[:15000]}
"""
    body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role":"system","content":system},{"role":"user","content":user}]
    }
    try:
        resp = rq.post(url, headers=headers, json=body, timeout=90)
        resp.raise_for_status()
        s = resp.json()["choices"][0]["message"]["content"]
        return json.loads(s)
    except Exception:
        return None

def google_cse_search(query: str, num=3):
    if not (CSE_API_KEY and CSE_CX):
        return []
    params = {"key": CSE_API_KEY, "cx": CSE_CX, "q": query, "num": num}
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=30)
    if r.status_code != 200:
        return []
    items = r.json().get("items", [])
    return [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]

def extract_from_pdf_url(url, title="", authors=""):
    pdf_bytes = fetch_pdf_bytes(url)
    if not pdf_bytes:
        return None, "no_pdf"
    full_text, tables_text = pdf_to_text_and_tables(pdf_bytes)
    joined = (tables_text + "\n\n" + full_text).strip()

    metrics = {"SD": None, "SE": None, "ICC": None, "MDE": None, "Variance": None}
    cites = []

    # regex first
    rx = regex_find_metrics(joined)
    for metric, value, _ in rx:
        if metrics[metric] is None:
            metrics[metric] = value

    # LLM fallback if missing
    if any(v is None for v in metrics.values()) and joined:
        llm = openai_structured_extract(joined[:15000], title, authors, url) or {}
        keymap = {"sd":"SD","se":"SE","icc":"ICC","mde":"MDE","variance":"Variance"}
        for k,v in llm.items():
            if k in keymap and v and metrics[keymap[k]] is None:
                metrics[keymap[k]] = v
        if "citations" in llm:
            cites = llm["citations"]

    rows = []
    for k in ["SD","SE","ICC","MDE","Variance"]:
        prov = "Not found"
        if metrics[k] is not None:
            prov = "PDF (regex)" if any(m==k for (m,_,_) in rx) else "PDF (LLM)"
        rows.append({"metric":k,"value":metrics[k],"provenance":prov,"details":cites})
    return rows, "ok"

def extract_with_web_fallback(title, authors, primary_link):
    rows, status = extract_from_pdf_url(primary_link, title, authors)
    if status == "ok":
        return rows, primary_link
    # try web search for an accessible PDF
    q = f'{title} {authors} filetype:pdf'
    hits = google_cse_search(q, num=4)
    for h in hits:
        rows2, status2 = extract_from_pdf_url(h["link"], title, authors)
        if status2 == "ok":
            for r in rows2:
                if r["provenance"].startswith("PDF"):
                    r["provenance"] = "Web (PDF)"
                elif r["value"]:
                    r["provenance"] = "Web (LLM)"
            return rows2, h["link"]
    # nothing
    empty = [{"metric":k,"value":None,"provenance":"Not found","details":[]} for k in ["SD","SE","ICC","MDE","Variance"]]
    return empty, None

# ---------- UI ----------
st.subheader("Step 1: Connect your Google Sheet")
csv_url_input = st.text_input("Paste your Google Sheet CSV link (from 'Publish to web')", value=SHEET_CSV_URL or "")
st.caption("Tip: File → Share → Publish to web → format: CSV → Publish, then copy the link here or set in Secrets.")

cols = st.columns([2,1])
with cols[0]:
    search_term = st.text_input("Optional: Filter by study/author/keyword", "")
with cols[1]:
    max_rows = st.number_input("How many rows to process now?", min_value=1, max_value=1000, value=20, step=5)

if st.button("Load sheet"):
    df = load_sheet(csv_url_input)
    if df.empty:
        st.stop()

    # try to guess columns for title/authors/link
    link_cols = [c for c in df.columns if "link" in c.lower() or "url" in c.lower()]
    title_cols = [c for c in df.columns if c.lower() in ("title","study","study title","paper","name of study")]
    auth_cols  = [c for c in df.columns if "author" in c.lower()]

    if not link_cols:
        st.error("I can't find a Link/URL column in your sheet. Please add one.")
        st.stop()

    link_col = link_cols[0]
    title_col = title_cols[0] if title_cols else df.columns[0]
    auth_col  = auth_cols[0]  if auth_cols else (df.columns[1] if len(df.columns)>1 else df.columns[0])

    if search_term.strip():
        mask = df.astype(str).apply(lambda r: r.str.contains(search_term, case=False, na=False)).any(axis=1)
        df = df[mask]

    st.success(f"Loaded {len(df)} rows from your sheet.")
    df = df.head(int(max_rows))
    st.dataframe(df[[title_col, auth_col, link_col]].rename(columns={title_col:"Title", auth_col:"Authors", link_col:"Link"}), use_container_width=True)

    results = []
    prog = st.progress(0)
    for i, row in df.reset_index(drop=True).iterrows():
        title  = str(row.get(title_col,"")).strip()
        auths  = str(row.get(auth_col,"")).strip()
        link   = str(row.get(link_col,"")).strip()

        with st.expander(f"{i+1}. {title or '(Untitled)'}", expanded=False):
            with st.spinner("Extracting…"):
                rows, used_link = extract_with_web_fallback(title, auths, link)
                for r in rows:
                    results.append({
                        "Title": title, "Authors": auths,
                        "Study Link": used_link or link,
                        "Metric": r["metric"], "Value": r["value"],
                        "Provenance": r["provenance"]
                    })
                st.table(pd.DataFrame([x for x in results if x["Title"]==title]))

        prog.progress(int((i+1)/len(df)*100))

    if results:
        st.subheader("All results")
        out_df = pd.DataFrame(results)
        st.dataframe(out_df, use_container_width=True)
        st.download_button("⬇️ Download CSV", out_df.to_csv(index=False).encode("utf-8"),
                           file_name="rct_metrics.csv", mime="text/csv")

st.divider()
st.markdown("**Legend:** ✅ PDF (regex) = exact text/table match · 🟦 PDF/Web (LLM) = extracted by GPT from content · ❌ Not found = nowhere available I could access")
