"""
台灣年報下載 + 多策略 RAG 問答系統 + Telegram 推播（Streamlit 版）
─────────────────────────────────────────────────────────
與原 Gradio 版功能相同：
1. 年報下載 / PDF 載入 / 8 種 RAG 檢索策略
2. Groq API Key、Telegram Bot Token、Telegram Chat ID 皆由使用者自行輸入
   （不寫死在程式碼中，系統不會儲存或外流，只在本次連線中使用）
3. AI 回答為「條列式重點 + 每點加上情境 emoji」
4. 「傳送到 Telegram」功能：手動按鈕傳送，或勾選「回答後自動傳送」

執行方式：
    streamlit run app_streamlit.py

安裝套件：
    pip install streamlit groq pypdf sentence-transformers numpy faiss-cpu scikit-learn requests beautifulsoup4

備註：
    - Groq / Telegram 的驗證狀態、已載入的 PDF 索引，都存放在以
      st.cache_resource 快取的單一 RAG 物件中，行為與原本 Gradio 版的
      全域變數 `rag` 相同：同一個部署下所有使用者共用同一份狀態。
      若要每位使用者各自獨立，需改用 st.session_state 儲存整個 RAG 物件
      （會失去模型快取帶來的效能優勢）。
"""

import os
import re
import tempfile
import zipfile

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# 可愛顏文字庫，隨機點綴用 (｡•̀ᴗ•́)و✧
KAOMOJI_OK = ["(｡•̀ᴗ•́)و", "٩(◕‿◕｡)۶", "(≧▽≦)", "(๑>ᴗ<๑)", "ヽ(・∀・)ﾉ"]
KAOMOJI_ERR = ["(╥﹏╥)", "(´；ω；`)", "(>_<)", "(TωT)"]
KAOMOJI_THINK = ["(・ω・)？", "(｡•́︿•̀｡)", "(¬‿¬)"]

# 判斷一段文字裡是否已含有 emoji（粗略範圍即可）
_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\u2B00-\u2BFF]"
)


def _ok(msg: str) -> str:
    return f"{msg} {np.random.choice(KAOMOJI_OK)}"


def _err(msg: str) -> str:
    return f"{msg} {np.random.choice(KAOMOJI_ERR)}"


# ─────────────────────────────────────────────
#  財報下載（邏輯與原版相同）
# ─────────────────────────────────────────────

def fetch_annual_report(stock_id: str, year: str):
    url = "https://doc.twse.com.tw/server-java/t57sb01"

    data1 = {
        "id": "", "key": "", "step": "1",
        "co_id": stock_id, "year": year,
        "seamon": "", "mtype": "F", "dtype": "F04",
    }
    try:
        resp1 = requests.post(url, data=data1, timeout=15)
        soup1 = BeautifulSoup(resp1.text, "html.parser")
        link1 = soup1.find("a").text
    except Exception as e:
        return _err(f"❌ [{stock_id}] 取得檔名失敗：{e}"), None

    data2 = {"step": "9", "kind": "F", "co_id": stock_id, "filename": link1}
    try:
        resp2 = requests.post(url, data=data2, timeout=15)
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        link2 = soup2.find("a").get("href")
    except Exception as e:
        return _err(f"❌ [{stock_id}] 取得 PDF 連結失敗：{e}"), None

    try:
        resp3 = requests.get("https://doc.twse.com.tw" + link2, timeout=30)
        filename = f"{year}_{stock_id}.pdf"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        with open(filepath, "wb") as f:
            f.write(resp3.content)
        return _ok(f"✅ [{stock_id}] {year} 年報下載成功"), filepath
    except Exception as e:
        return _err(f"❌ [{stock_id}] 下載 PDF 失敗：{e}"), None


def download_reports(stock_ids_input: str, year: str):
    raw = stock_ids_input.replace(",", "\n").replace(" ", "\n")
    stock_ids = [s.strip() for s in raw.splitlines() if s.strip()]
    if not stock_ids:
        return "⚠️ 請輸入至少一個股號 " + np.random.choice(KAOMOJI_THINK), None
    if not year.strip().isdigit():
        return "⚠️ 年份格式錯誤，請輸入民國年（如：112） " + np.random.choice(KAOMOJI_THINK), None

    logs, pdf_paths = [], []
    for sid in stock_ids:
        msg, path = fetch_annual_report(sid, year.strip())
        logs.append(msg)
        if path:
            pdf_paths.append(path)

    summary = "\n".join(logs)
    if len(pdf_paths) == 1:
        return summary, pdf_paths[0]
    if pdf_paths:
        zip_path = os.path.join(tempfile.gettempdir(), f"annual_reports_{year.strip()}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in pdf_paths:
                zf.write(p, os.path.basename(p))
        return summary, zip_path
    return summary, None


# ─────────────────────────────────────────────
#  Multi-Strategy RAG ＋ Telegram 推播（邏輯與原版相同）
# ─────────────────────────────────────────────

class MultiStrategyRAG:
    def __init__(self):
        self.client: Groq | None = None
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.chunks: list[str] = []
        self.embeddings = None
        self.index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.llm_model = "llama-3.1-8b-instant"

        # Telegram 設定（由使用者於介面輸入後儲存）
        self.telegram_bot_token: str | None = None
        self.telegram_chat_id: str | None = None

    # ── Groq Token 設定 ─────────────────────────

    def set_api_key(self, api_key: str) -> str:
        api_key = (api_key or "").strip()
        if not api_key:
            return _err("⚠️ 請輸入你的 Groq API Key 才能開始使用")
        try:
            self.client = Groq(api_key=api_key)
            self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return _ok("✅ Groq Token 驗證成功，已就緒")
        except Exception as e:
            self.client = None
            return _err(f"❌ Groq Token 無效或連線失敗：{e}")

    def _ensure_client(self):
        if self.client is None:
            raise RuntimeError("尚未設定 Groq API Key，請先在上方輸入並儲存 Token")

    # ── Telegram 設定 ────────────────────────────

    def set_telegram_config(self, bot_token: str, chat_id: str) -> str:
        bot_token = (bot_token or "").strip()
        chat_id = (chat_id or "").strip()
        if not bot_token or not chat_id:
            return "⚠️ 請同時輸入 Telegram Bot Token 與 Chat ID " + np.random.choice(KAOMOJI_THINK)
        try:
            resp = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10)
            data = resp.json()
            if not data.get("ok"):
                raise Exception(data.get("description", "未知錯誤"))
            bot_username = data["result"].get("username", "unknown_bot")
            self.telegram_bot_token = bot_token
            self.telegram_chat_id = chat_id
            return _ok(f"✅ Telegram Bot 驗證成功：@{bot_username}，之後回答可推播到 Chat ID {chat_id}")
        except Exception as e:
            self.telegram_bot_token = None
            self.telegram_chat_id = None
            return _err(f"❌ Telegram 設定失敗：{e}")

    def send_to_telegram(self, text: str) -> str:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return _err("❌ 請先在上方設定並驗證 Telegram Bot Token 與 Chat ID")
        if not text or not text.strip():
            return "⚠️ 沒有內容可傳送 " + np.random.choice(KAOMOJI_THINK)

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        max_len = 3800  # 留一些餘裕給 Telegram 4096 字元上限
        chunks = [text[i: i + max_len] for i in range(0, len(text), max_len)] or [text]

        try:
            for i, chunk in enumerate(chunks, start=1):
                prefix = f"({i}/{len(chunks)})\n" if len(chunks) > 1 else ""
                resp = requests.post(
                    url,
                    data={"chat_id": self.telegram_chat_id, "text": prefix + chunk},
                    timeout=15,
                )
                data = resp.json()
                if not data.get("ok"):
                    raise Exception(data.get("description", "傳送失敗"))
            return _ok(f"✅ 已成功傳送到 Telegram（共 {len(chunks)} 則訊息）")
        except Exception as e:
            return _err(f"❌ 傳送到 Telegram 失敗：{e}")

    # ── 載入 ──────────────────────────────────

    def load_pdf_file(self, filepath: str) -> str:
        try:
            reader = PdfReader(filepath)
            full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
            self.chunks = self._split_text(full_text, chunk_size=800, overlap=150)
            self._build_indices()
            return _ok(
                f"✅ 成功載入 PDF（{os.path.basename(filepath)}）！"
                f"共 {len(reader.pages)} 頁，分割為 {len(self.chunks)} 個片段"
            )
        except Exception as e:
            return _err(f"❌ 載入失敗：{e}")

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = re.sub(r"\s+", " ", text[start: start + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _build_indices(self):
        self.embeddings = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings.astype("float32"))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

    # ── 8 種策略（邏輯不變）───────────────────

    def strategy_1_basic_similarity(self, query, top_k=3):
        qv = self.embedding_model.encode([query])
        _, idx = self.index.search(qv.astype("float32"), top_k)
        return [self.chunks[i] for i in idx[0]]

    def strategy_2_tfidf(self, query, top_k=3):
        qv = self.tfidf_vectorizer.transform([query])
        scores = (self.tfidf_matrix * qv.T).toarray().flatten()
        return [self.chunks[i] for i in scores.argsort()[-top_k:][::-1]]

    def strategy_3_hybrid(self, query, top_k=3):
        qv = self.embedding_model.encode([query])
        _, sem_idx = self.index.search(qv.astype("float32"), top_k * 2)
        qv_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * qv_tfidf.T).toarray().flatten()
        tfidf_idx = tfidf_scores.argsort()[-top_k * 2:][::-1]
        combined = list(set(sem_idx[0].tolist() + tfidf_idx.tolist()))
        return [self.chunks[i] for i in combined[:top_k]]

    def strategy_4_reranking(self, query, top_k=3):
        self._ensure_client()
        candidates = self.strategy_1_basic_similarity(query, top_k=top_k * 2)
        reranked = []
        for chunk in candidates:
            prompt = f"問題：{query}\n\n文本：{chunk[:200]}...\n\n相關度(0-10)："
            try:
                r = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10, temperature=0,
                )
                raw = r.choices[0].message.content.strip()
                score = float(re.findall(r"\d+", raw)[0]) if re.findall(r"\d+", raw) else 0
            except Exception:
                score = 0
            reranked.append((chunk, score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in reranked[:top_k]]

    def strategy_5_multi_query(self, query, top_k=3):
        self._ensure_client()
        prompt = f"將以下問題改寫成3個相關但不同角度的問題，用換行分隔：\n{query}"
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.7,
            )
            queries = [query] + r.choices[0].message.content.strip().split("\n")[:3]
        except Exception:
            queries = [query]
        all_chunks = []
        for q in queries:
            all_chunks.extend(self.strategy_1_basic_similarity(q, top_k=2))
        return list(dict.fromkeys(all_chunks))[:top_k]

    def strategy_6_contextual_compression(self, query, top_k=3):
        self._ensure_client()
        chunks = self.strategy_1_basic_similarity(query, top_k=top_k)
        compressed = []
        for chunk in chunks:
            prompt = f"從以下文本中，濃縮成1-2句與問題「{query}」最相關的重點：\n\n{chunk}"
            try:
                r = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150, temperature=0,
                )
                compressed.append(r.choices[0].message.content.strip())
            except Exception:
                compressed.append(chunk[:300])
        return compressed

    def strategy_7_parent_child(self, query, top_k=3):
        small_chunks = self._split_text(" ".join(self.chunks), chunk_size=300, overlap=50)
        small_emb = self.embedding_model.encode(small_chunks, convert_to_numpy=True)
        small_idx = faiss.IndexFlatL2(small_emb.shape[1])
        small_idx.add(small_emb.astype("float32"))
        qv = self.embedding_model.encode([query])
        _, indices = small_idx.search(qv.astype("float32"), top_k)
        results = []
        for i in indices[0]:
            for big in self.chunks:
                if small_chunks[i] in big:
                    results.append(big)
                    break
        return list(dict.fromkeys(results))[:top_k]

    def strategy_8_hypothetical_answer(self, query, top_k=3):
        self._ensure_client()
        prompt = f"請對以下問題給出一個假設性的簡短答案：\n{query}"
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.7,
            )
            hypo = r.choices[0].message.content
        except Exception:
            hypo = query
        qv = self.embedding_model.encode([hypo])
        _, idx = self.index.search(qv.astype("float32"), top_k)
        return [self.chunks[i] for i in idx[0]]

    STRATEGY_MAP = {
        "1. 基礎語意搜尋": "strategy_1_basic_similarity",
        "2. TF-IDF 關鍵詞": "strategy_2_tfidf",
        "3. 混合搜尋": "strategy_3_hybrid",
        "4. 重新排序": "strategy_4_reranking",
        "5. 多查詢擴展": "strategy_5_multi_query",
        "6. 上下文壓縮": "strategy_6_contextual_compression",
        "7. 父子文檔": "strategy_7_parent_child",
        "8. 假設性答案 (HyDE)": "strategy_8_hypothetical_answer",
    }

    # ── 生成答案：濃縮 + 條列式重點 + emoji ─────────────

    @staticmethod
    def _ensure_emoji_bullets(text: str) -> str:
        """確保每個條列項目前面都有 emoji，若 LLM 忘記加，補上預設的 📌。"""
        lines = text.split("\n")
        out = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("・", "-", "•", "*")):
                marker, rest = stripped[0], stripped[1:].strip()
                if not _EMOJI_PATTERN.search(rest[:4]):
                    line = line.replace(stripped, f"{marker} 📌 {rest}", 1)
            out.append(line)
        return "\n".join(out)

    def generate_answer(self, query: str, strategy: str, top_k: int = 3):
        if self.client is None:
            return _err("❌ 請先在上方貼上 Groq API Key 並完成驗證"), ""
        if not self.chunks:
            return _err("❌ 請先載入 PDF 檔案"), ""

        fn = getattr(self, self.STRATEGY_MAP.get(strategy, "strategy_1_basic_similarity"))
        relevant_chunks = fn(query, top_k)
        context = "\n\n---\n\n".join(relevant_chunks)

        prompt = (
            "你是專業財報分析助手，請根據下方「上下文」回答問題。\n"
            "規則：\n"
            "・只保留重點，去除贅詞與客套話\n"
            "・用條列式（每點一行，前面加「・」，「・」後面緊接一個符合該點內容意涵的 emoji，"
            "例如 📈 營收成長、📉 獲利下滑、💰 現金部位、⚠️ 風險因素，不要每點都用同一個 emoji）\n"
            "・最多 5 點\n"
            "・若有數字（營收、淨利、比例等），務必列出具體數值\n"
            "・若上下文沒有相關資訊，直接說「上下文中查無相關資訊」，不要臆測\n"
            "・全程使用繁體中文\n\n"
            f"上下文：\n{context}\n\n"
            f"問題：{query}\n\n"
            "請用條列式重點（每點附 emoji）回答："
        )

        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是精簡扼要、只講重點的財務報告分析助手，禁止長篇廢話，且每個條列重點都要附上情境 emoji。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600, temperature=0.3,
            )
            answer = r.choices[0].message.content.strip()
            answer = self._ensure_emoji_bullets(answer)
            answer = f"{answer}\n\n{np.random.choice(KAOMOJI_OK)} 回答完畢！"
            source_info = (
                f"📚 使用策略：{strategy}\n📄 檢索片段數：{len(relevant_chunks)}\n\n"
                + "=" * 50 + "\n相關文本片段：\n" + "=" * 50 + "\n\n" + context
            )
            return answer, source_info
        except Exception as e:
            return _err(f"❌ 生成答案失敗：{e}"), ""


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────

STRATEGY_CHOICES = [
    "1. 基礎語意搜尋",
    "2. TF-IDF 關鍵詞",
    "3. 混合搜尋",
    "4. 重新排序",
    "5. 多查詢擴展",
    "6. 上下文壓縮",
    "7. 父子文檔",
    "8. 假設性答案 (HyDE)",
]

st.set_page_config(page_title="台灣年報 RAG 問答系統 × Telegram 推播", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="首次啟動中，載入嵌入模型...")
def get_rag() -> MultiStrategyRAG:
    return MultiStrategyRAG()


rag = get_rag()

# session_state 預設值（僅用來記住畫面上顯示的文字/檔案，實際的
# 索引、Groq client、Telegram 設定都放在上面快取的 rag 物件裡）
_defaults = {
    "dl_summary": "", "dl_file_path": None,
    "b_status": "",
    "answer": "", "source": "", "telegram_status": "",
    "key_status": "", "tg_status": "",
    "upload_status": "",
    "q_input": "",
}
for _k, _v in _defaults.items():
    st.session_state.setdefault(_k, _v)

st.title("📊 台灣上市公司年報 × RAG 智慧問答 × Telegram 推播 (｡•ᴗ•｡)")
st.markdown(
    """
> 資料來源：[證交所 TWSE](https://doc.twse.com.tw)　｜　年份請填**民國年**（例：112 = 2023年）
>
> ⚠️ 使用前請先在下方貼上你自己的 **Groq API Key**（[免費申請](https://console.groq.com/keys)）
> 以及 **Telegram Bot Token / Chat ID**，系統不會儲存或外流你的金鑰，只在本次連線中使用喔 ٩(ˊᗜˋ*)و
>
> 🤖 Telegram 設定教學：用 [@BotFather](https://t.me/BotFather) 建立機器人取得 Token → 先傳一句話給你的機器人
> → 造訪 `https://api.telegram.org/bot<你的TOKEN>/getUpdates` 從回傳 JSON 找到 `chat.id`
"""
)

# ── Groq Token 設定區 ─────────────────────────
with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        api_key_input = st.text_input(
            "🔑 Groq API Key", type="password", placeholder="貼上你的 gsk_ 開頭金鑰...", key="api_key_input"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("💾 儲存並驗證 Groq Token", type="primary", use_container_width=True):
            st.session_state.key_status = rag.set_api_key(api_key_input)
    if st.session_state.key_status:
        st.caption(st.session_state.key_status)

# ── Telegram 設定區 ───────────────────────────
with st.container(border=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tg_token_input = st.text_input(
            "🤖 Telegram Bot Token", type="password",
            placeholder="從 @BotFather 取得，例：123456:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            key="tg_token_input",
        )
    with col2:
        tg_chatid_input = st.text_input("💬 Telegram Chat ID", placeholder="例：8722940849", key="tg_chatid_input")
    with col3:
        st.write("")
        st.write("")
        if st.button("💾 儲存並驗證 Telegram", type="primary", use_container_width=True):
            st.session_state.tg_status = rag.set_telegram_config(tg_token_input, tg_chatid_input)
    if st.session_state.tg_status:
        st.caption(st.session_state.tg_status)

tab1, tab2, tab3 = st.tabs(["📥 Step 1｜下載年報", "📂 Step 2｜載入 PDF", "💬 Step 3｜RAG 問答 + Telegram 推播"])

# ── Tab 1: 年報下載 ──────────────────────
with tab1:
    st.markdown("### 輸入股號與年份，自動從證交所抓取 PDF (｀・ω・´)ﾉ")
    col1, col2 = st.columns(2)
    with col1:
        stock_input = st.text_area(
            "股號（可多筆）",
            placeholder="每行一個，或用逗號/空格分隔\n例：\n2330\n2317",
            height=120,
            key="stock_input",
        )
        year_input = st.text_input("年份（民國年）", value="112", placeholder="例：112", key="dl_year_input")
        if st.button("🔍 下載年報", type="primary"):
            summary, path = download_reports(stock_input, year_input)
            st.session_state.dl_summary = summary
            st.session_state.dl_file_path = path
        with st.expander("範例"):
            st.code("股號：2330\\n2317　｜　年份：112\n\n股號：2330　｜　年份：111")
    with col2:
        st.text_area("下載狀態", value=st.session_state.dl_summary, height=150, disabled=True)
        if st.session_state.dl_file_path and os.path.exists(st.session_state.dl_file_path):
            with open(st.session_state.dl_file_path, "rb") as f:
                st.download_button(
                    "⬇️ 下載檔案（PDF 或 ZIP）",
                    data=f.read(),
                    file_name=os.path.basename(st.session_state.dl_file_path),
                )

# ── Tab 2: 載入 PDF ──────────────────────
with tab2:
    st.markdown(
        "### 選擇 PDF 來源 (＾▽＾)\n"
        "- **方式 A**：直接上傳本機 PDF\n"
        "- **方式 B**：輸入已下載年報的股號＋年份，自動抓取並載入"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 方式 A｜上傳 PDF")
        upload_input = st.file_uploader("選擇 PDF", type=["pdf"], key="upload_input")
        if st.button("📤 上傳並載入", type="primary"):
            if upload_input is None:
                st.session_state.upload_status = "⚠️ 請選擇 PDF 檔案 " + np.random.choice(KAOMOJI_THINK)
            else:
                tmp_path = os.path.join(tempfile.gettempdir(), upload_input.name)
                with open(tmp_path, "wb") as f:
                    f.write(upload_input.getbuffer())
                st.session_state.upload_status = rag.load_pdf_file(tmp_path)
        if st.session_state.upload_status:
            st.caption(st.session_state.upload_status)

    with col2:
        st.markdown("#### 方式 B｜指定股號年份自動載入")
        b_stock = st.text_input("單一股號", placeholder="例：2330", key="b_stock")
        b_year = st.text_input("年份（民國年）", value="112", placeholder="例：112", key="b_year")
        if st.button("🚀 下載並載入", type="primary"):
            msg, path = fetch_annual_report(b_stock.strip(), b_year.strip())
            if path is None:
                st.session_state.b_status = msg
            else:
                load_msg = rag.load_pdf_file(path)
                st.session_state.b_status = msg + "\n" + load_msg
        if st.session_state.b_status:
            st.caption(st.session_state.b_status)

# ── Tab 3: RAG 問答 + Telegram 推播 ──────────
with tab3:
    st.markdown("### 針對已載入的年報進行智慧問答，回答會**濃縮成條列重點並附上 emoji** ✍(◔◡◔)")
    col1, col2 = st.columns([1, 2])

    with col1:
        strategy_dd = st.selectbox("RAG 策略", STRATEGY_CHOICES, index=0)
        top_k_slider = st.slider("Top-K 片段數", 1, 10, 3, 1)
        st.markdown(
            """
**策略說明**
・1. 基礎語意搜尋：向量相似度
・2. TF-IDF：詞頻統計
・3. 混合搜尋：語意 + 關鍵詞
・4. 重新排序：LLM 重新評分
・5. 多查詢擴展：生成多問題
・6. 上下文壓縮：提取精華
・7. 父子文檔：小→大上下文
・8. HyDE：假設答案再搜尋
"""
        )
        auto_send_checkbox = st.checkbox("✅ 回答後自動傳送到 Telegram", value=False)

    with col2:
        example_questions = [
            "公司的主要業務為何？",
            "去年的營業收入與淨利各是多少？",
            "公司面臨哪些主要風險？",
            "研發費用佔營收的比例是多少？",
        ]

        def _apply_example():
            if st.session_state.example_pick != "－":
                st.session_state.q_input = st.session_state.example_pick

        st.selectbox(
            "範例問題（選擇後自動帶入下方問題框）",
            ["－"] + example_questions,
            key="example_pick",
            on_change=_apply_example,
        )
        q_input = st.text_area(
            "問題", placeholder="例：這份年報的營收狀況如何？", height=100, key="q_input"
        )

        if st.button("🔍 提問", type="primary"):
            with st.spinner("思考中..."):
                answer, source = rag.generate_answer(q_input, strategy_dd, top_k_slider)
            st.session_state.answer = answer
            st.session_state.source = source
            if auto_send_checkbox:
                if answer.startswith("❌") or answer.startswith("⚠️"):
                    st.session_state.telegram_status = "⚠️ 回答生成失敗，未傳送到 Telegram"
                else:
                    payload = f"❓ 問題：{q_input}\n\n{answer}"
                    st.session_state.telegram_status = rag.send_to_telegram(payload)
            else:
                st.session_state.telegram_status = ""

        st.text_area(
            "AI 回答（條列式重點 + emoji）", value=st.session_state.answer, height=260, disabled=True
        )

        if st.button("📤 傳送到 Telegram"):
            answer = st.session_state.answer
            if not answer or answer.startswith("❌") or answer.startswith("⚠️"):
                st.session_state.telegram_status = "⚠️ 沒有可傳送的回答，請先提問 " + np.random.choice(KAOMOJI_THINK)
            else:
                payload = f"❓ 問題：{q_input}\n\n{answer}"
                st.session_state.telegram_status = rag.send_to_telegram(payload)

        if st.session_state.telegram_status:
            st.caption(st.session_state.telegram_status)

        with st.expander("📚 查看檢索片段"):
            st.text_area(
                "相關來源", value=st.session_state.source, height=300, disabled=True,
                label_visibility="collapsed",
            )
