import io
import json
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from ollama import Client
from pypdf import PdfReader

DB_PATH = "localdocu_vault.db"
LLM_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

client = Client(host="http://localhost:11434")


def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS case_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT,
                category TEXT,
                summary TEXT,
                action_items TEXT,
                risk_flag TEXT,
                source_text TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def extract_text(uploaded) -> str:
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded.read()))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return uploaded.read().decode("utf-8", errors="ignore")


def analyze(raw_text: str, category: str) -> dict:
    prompt = f"""You are a strict, privacy-preserving local document assistant.
Analyze the following {category} and return ONLY valid JSON with this schema:
{{
  "client_name_or_id": "string",
  "executive_summary": "3 sentences max",
  "action_items": ["string"],
  "risk_flag": "Low" | "Medium" | "High"
}}

Document:
\"\"\"{raw_text[:12000]}\"\"\""""

    resp = client.generate(model=LLM_MODEL, prompt=prompt, format="json")
    try:
        data = json.loads(resp["response"])
    except json.JSONDecodeError:
        data = {
            "client_name_or_id": "Unknown",
            "executive_summary": resp["response"],
            "action_items": [],
            "risk_flag": "Medium",
        }

    if data.get("risk_flag") not in ("Low", "Medium", "High"):
        data["risk_flag"] = "Medium"

    return data


def embed(text: str) -> list[float]:
    return client.embeddings(model=EMBED_MODEL, prompt=text[:6000])["embedding"]


def save_case(data: dict, category: str, raw_text: str):
    emb = embed(data.get("executive_summary", "") + "\n" + raw_text)
    with db() as c:
        c.execute(
            """
            INSERT INTO case_summaries
            (client_id, category, summary, action_items, risk_flag, source_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("client_name_or_id"),
                category,
                data.get("executive_summary"),
                json.dumps(data.get("action_items", [])),
                data["risk_flag"],
                raw_text,
                json.dumps(emb),
            ),
        )


def semantic_search(query: str, k: int = 5) -> list[dict]:
    q = np.array(embed(query))
    with db() as c:
        rows = c.execute(
            """
            SELECT client_id, category, summary, risk_flag, created_at, embedding
            FROM case_summaries
            """
        ).fetchall()

    scored = []
    for cid, cat, summ, risk, ts, e in rows:
        v = np.array(json.loads(e))
        sim = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
        scored.append(
            {
                "score": round(sim, 3),
                "client": cid,
                "category": cat,
                "risk": risk,
                "date": ts,
                "summary": summ,
            }
        )

    return sorted(scored, key=lambda x: -x["score"])[:k]


def tabpfn_scan(df: pd.DataFrame, target: str, threshold: float):
    from sklearn.model_selection import StratifiedKFold
    from tabpfn import TabPFNClassifier

    X = pd.get_dummies(df.drop(columns=[target]), dummy_na=True).fillna(0).astype(float)
    y_raw = df[target].astype(str)
    classes = sorted(y_raw.unique())
    y = y_raw.map({c: i for i, c in enumerate(classes)}).values

    class_counts = np.bincount(y)
    n_splits = max(2, min(5, int(class_counts.min())))

    p_true = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    for tr, te in skf.split(X, y):
        model = TabPFNClassifier()
        model.fit(X.iloc[tr].values, y[tr])
        proba = model.predict_proba(X.iloc[te].values)

        for row, idx in enumerate(te):
            cls_pos = list(model.classes_).index(y[idx]) if y[idx] in model.classes_ else None
            p_true[idx] = proba[row, cls_pos] if cls_pos is not None else 0.0

    out = df.copy()
    out["label_likelihood"] = p_true.round(3)
    out["anomaly"] = p_true < threshold
    return out


st.set_page_config(page_title="LocalDocuShield", page_icon="🛡️", layout="wide")
st.title("🛡️ LocalDocuShield")
st.caption("100% local AI document intake. No cloud calls, no data leaves this machine.")

init_db()


tab1, tab2, tab3, tab4 = st.tabs(["📋 Intake", "🔎 Case Search", "📊 TabPFN Ledger Scan", "🗄️ Vault"])

with tab1:
    doc_type = st.selectbox(
        "Document context",
        ["Medical Intake Note", "Legal Brief / Contract", "Financial / Audit Ledger"],
    )
    up = st.file_uploader("Upload PDF / TXT (optional)", type=["pdf", "txt"])
    pasted = st.text_area("...or paste document / transcript", height=200)

    if st.button("⚡ Process Locally"):
        text = extract_text(up) if up else pasted
        if not text.strip():
            st.warning("Provide a document first.")
        else:
            try:
                with st.spinner(f"Running {LLM_MODEL} on this device..."):
                    result = analyze(text, doc_type)
                    save_case(result, doc_type, text)

                st.success("Done. Processed and stored on-device.")

                c1, c2 = st.columns(2)
                c1.write(f"**Client / Case ID:** {result.get('client_name_or_id')}")
                c1.write(f"**Risk Flag:** {result['risk_flag']}")

                c2.write("**Action Items:**")
                for a in result.get("action_items", []):
                    c2.write(f"- {a}")

                st.info(f"**Executive Summary:** {result.get('executive_summary')}")
            except Exception as e:
                st.error(f"Is Ollama running and are the models pulled? ({e})")

with tab2:
    q = st.text_input("Search past cases (semantic)")
    if q:
        for r in semantic_search(q):
            st.markdown(
                f"**{r['client']}** · {r['category']} · risk **{r['risk']}** · score {r['score']}  \n{r['summary']}"
            )
            st.divider()

with tab3:
    csv = st.file_uploader("Billing / vitals ledger (CSV)", type=["csv"], key="csv")
    if csv:
        df = pd.read_csv(csv)
        st.dataframe(df.head(10))

        target = st.selectbox(
            "Categorical column to validate (e.g. billing code, status)",
            df.columns,
        )
        thr = st.slider("Anomaly threshold (label likelihood below)", 0.01, 0.5, 0.1)

        if st.button("Run TabPFN Scan"):
            if len(df) < 20:
                st.warning("Need at least ~20 rows.")
            else:
                with st.spinner("Running TabPFN locally..."):
                    res = tabpfn_scan(df, target, thr)

                flagged = res[res["anomaly"]]
                st.success(f"{len(flagged)} suspicious row(s) of {len(res)}")
                st.dataframe(flagged)

with tab4:
    with db() as c:
        vault = pd.read_sql_query(
            """
            SELECT id, client_id, category, risk_flag, summary, created_at
            FROM case_summaries
            ORDER BY id DESC
            """,
            c,
        )
    st.dataframe(vault, use_container_width=True)
