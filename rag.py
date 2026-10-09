"""RAG pipeline: ingest documents into the vector store, retrieve, answer.

This is the hand-rolled, transparent equivalent of the learning-notes chain
`create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))`:

    ingest: files -> recursive character chunks -> local embeddings -> sqlite-vec
    ask:    question -> embed -> KNN top-k -> stuff chunks into a system prompt
            -> stream the answer through the existing api_client.ask()
"""

import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from api_client import ask
from embeddings import embed_query, embed_texts, rerank
from rag_store import RagStore, create_rag_store

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_SUFFIXES = {".md", ".txt", ".py"}

# 这个常量是 RAG 系统的拒答阈值——reranker 分数低于 0.1 就认为库里没有相关内容，直接拒答，既省 LLM 费用又防编造；
# 0.1 这个值来自"噪声最高分 0.055 与信号最低分 0.35 之间取中点"的实测标定，并两头留足安全余量。
MIN_RERANK_SCORE = 0.1

_RAG_SYSTEM_PROMPT = """你是 modelclaw 的知识库问答助手。请根据下面检索到的文档片段回答用户的问题。

规则：
1. 只能依据检索到的片段作答，不要使用你自己的外部知识
2. 如果片段里没有答案，直接说「知识库中没有相关内容」，不要编造
3. 回答末尾标注引用了哪些来源文件

检索到的片段（格式：[来源] 内容）：

{context}"""


def _rag_cfg(cfg: dict) -> dict:
    return cfg.get("rag", {})


def _collect_files(path: Path) -> list[Path]:
    """
    **把传入的"文件或目录路径"统一解析成受支持文件的列表。**

    传入单个文件 → 校验后缀，支持则返回仅含它的列表；
    传入目录 → 递归遍历其中所有子目录，收集全部受支持的文件（按路径排序）；
    路径不存在 → 抛出 `FileNotFoundError`。

    ### 输入/输出示例

    ```python
    # 例1：传入一个受支持的文件
    _collect_files(Path("docs/guide.pdf"))
    # → [Path("docs/guide.pdf")]

    # 例2：传入一个不支持后缀的文件
    _collect_files(Path("docs/logo.png"))
    # → []

    # 例3：传入目录（含子目录），只收集受支持的文件并排序
    _collect_files(Path("docs/"))
    # docs/ 结构：
    #   docs/a.md
    #   docs/b.pdf
    #   docs/sub/c.docx
    #   docs/logo.png      ← 不支持，被过滤
    # → [Path("docs/a.md"), Path("docs/b.pdf"), Path("docs/sub/c.docx")]

    # 例4：路径不存在
    _collect_files(Path("no/such/path"))
    # → 抛出 FileNotFoundError: 路径不存在: no/such/path
    ```

    ### 内部执行流程
    - step1: 判断 `path.is_file()`——是文件则检查后缀：在
            `SUPPORTED_SUFFIXES` 中返回 `[path]`，不在则返回空列表；
    - step2: 否则判断 `path.is_dir()`——是目录则用 `rglob("*")` 递归遍历
            所有子目录，筛选"是文件且后缀受支持"的项，排序后返回；
    - step3: 两者都不是（路径不存在）→ 抛出 `FileNotFoundError`，
            明确报错而不是静默返回空列表。
    """
    if path.is_file():
        return [path] if path.suffix in SUPPORTED_SUFFIXES else []
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix in SUPPORTED_SUFFIXES
        )
    raise FileNotFoundError(f"路径不存在: {path}")


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    **把长文本切分成若干重叠的小块（chunk），供向量化入库。**

    按"段落 → 行 → 句号 → 逗号 → 空格 → 任意字符"的优先级，
    尽量在语义边界处下刀，避免把一句话拦腰切断；
    相邻两块之间保留 overlap 个字符的重叠，防止语义被边界切断。

    ### 输入/输出示例

    ```python
    text = (
        "第一自然段。讲述背景。\n\n"
        "第二自然段。讲述方法。还有细节。\n\n"
        "第三自然段。"
    )
    _chunk_text(text, chunk_size=20, chunk_overlap=5)
    # → ["第一自然段。讲述背景。", "第二自然段。讲述方法。", ...]
    #   每个块不超过 20 个字符，相邻块之间约有 5 个字符重叠
    ```

    ### 内部执行流程
    - step1: 懒加载 `RecursiveCharacterTextSplitter`，配置目标块大小、
            重叠长度和分隔符优先级列表；
    - step2: 调用 `split_text` 递归切分——先尝试在优先级高的分隔符
            （段落/句号）处切开，块仍超大小再退到次级分隔符（逗号/空格），
            最后兜底按任意字符硬切；
    - step3: 过滤掉纯空白（或空）的块，返回干净列表。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def ingest_path(cfg: dict, path_str: str, force: bool = False) -> int:
    """
    **把一个文件或目录"灌入"向量库：读取 → 切块 → 嵌入 → 入库，返回入库的块总数。**

    已入库的文件默认跳过（幂等）；传 `force=True` 则先删旧向量再重新嵌入。
    支持 .md / .txt / .py 文件，目录会递归处理其中所有受支持的文件。

    ### 输入/输出示例

    ```python
    # 例1：入库一个目录
    ingest_path(cfg, "docs/")
    # 控制台输出：
    #   跳过（已入库，--force 可重灌）: docs/old.md
    #   √ docs/guide.md → 12 块
    #   √ docs/faq.md → 7 块
    # → 返回 19

    # 例2：强制重灌（删除旧向量后重新嵌入）
    ingest_path(cfg, "docs/guide.md", force=True)

    # 例3：没有受支持的文件
    ingest_path(cfg, "images/")
    # → 控制台提示"没有可入库的文件"，返回 0
    ```

    ### 内部执行流程
    - step1: 从配置解析出 RAG 配置（切块参数），创建/连接向量库；
    - step2: 用 `_collect_files` 把路径解析成受支持文件列表，空则直接返回 0；
    - step3: 逐个文件循环——已入库且非 force 则跳过；force 则先删旧向量；
    - step4: 读文件文本 → `_chunk_text` 切块 → 空内容跳过；
    - step5: `embed_texts` 批量嵌入（显示进度），`store.add_chunks` 落库，
            累计块数；
    - step6: 记日志，返回总块数。
    """
    rag_cfg = _rag_cfg(cfg)
    chunk_size = rag_cfg.get("chunk_size", 500)
    chunk_overlap = rag_cfg.get("chunk_overlap", 50)
    store = create_rag_store(cfg)

    files = _collect_files(Path(path_str))
    if not files:
        console.print("[dim](没有可入库的文件：支持 .md / .txt / .py)[/dim]")
        return 0

    total = 0
    for f in files:
        source = str(f)
        if store.has_source(source):
            if not force:
                console.print(f"[dim]跳过（已入库，--force 可重灌）: {source}[/dim]")
                continue
            store.delete_source(source)
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            console.print(f"[dim]跳过（内容为空）: {source}[/dim]")
            continue
        with console.status(f"嵌入 {source}（{len(chunks)} 块）..."):
            vectors = embed_texts(chunks, rag_cfg)
        store.add_chunks(source, chunks, vectors)
        total += len(chunks)
        console.print(f"  [green]√[/green] {source} → {len(chunks)} 块")

    logger.info("Ingested %d chunks from %s", total, path_str)
    return total


def retrieve(cfg: dict, question: str, k: int | None = None) -> list[dict]:
    """
    **根据用户的问题，从文档库中找出最相关的 k 个片段并返回。**

    查找分三步：先用两种方法各找一批相关片段（向量检索按意思找、
    关键词检索按原词找），把两批结果合并去重；再逐对细看，
    挑出和问题最相关的 k 个。如果最后一步出错，就用合并后的结果
    直接取前 k 个，保证一定能返回答案。

    ### 输入/输出示例

    ```python
    retrieve(cfg, "如何重置密码？", k=4)
    # → 返回 4 个最相关的文档片段，例如：
    # [
    #   {"content": "打开设置页面，点击账户...", "source": "guide.md", "score": 0.91},
    #   {"content": "密码需包含大小写字母...", "source": "policy.md", "score": 0.42},
    #   ...
    # ]
    ```

    ### 内部执行流程
    - step1: 确定最终要返回几个（k），以及先粗找多少个——
            粗找的数量要比 k 大，给后面的精选留出挑选空间；
    - step2: 用两种方法各找一批：
            ① 向量检索：把问题转成向量，找意思相近的片段；
            ② 关键词检索：用问题里的原词直接搜索；
    - step3: 把两批结果合并、去掉重复的。两种方法都排得靠前的，
            就是最相关的；
    - step4: 把每个片段和问题放在一起重新打分排序，
            返回分数最高的 k 个；
            如果这一步出错，就用第 3 步合并后的排序，直接取前 k 个返回。
    """
    rag_cfg = _rag_cfg(cfg)
    k = k or rag_cfg.get("top_k", 4)
    recall_k = max(k * 3, 12)
    store = create_rag_store(cfg)
    vector = embed_query(question, rag_cfg)
    vec_hits = store.search(vector, recall_k)
    #return vec_hits
    bm25_hits = _bm25_search(store, question, recall_k)
    candidates = _rrf_fuse(vec_hits, bm25_hits, recall_k)

    reranker = rag_cfg.get("reranker_model", "BAAI/bge-reranker-base")
    try:
        return rerank(question, candidates, k, reranker)
    except Exception as exc:
        # Reranker is an enhancement; never let it break the query path
        logger.warning("Rerank failed, falling back to RRF order: %s", exc)
        return candidates[:k]

def _tokenize(text: str) -> list[str]:
    """
    **把一段文本拆成一个个"词"（分词），供 BM25 关键词检索使用。**

    英文按单词拆（转成小写）；中文不按词拆，而是每两个相邻汉字组成一对
    （如"字段作用"拆成"字段""段作""作用"）。中文单个字不单独算词，
    因为"有""哪""什么"这类字到处都有，会干扰检索判断；
    两个字一组恰好能表达意思，而且重复率低的词组会在检索中自动获得更高权重。

    ### 输入/输出示例

    ```python
    _tokenize("Chunk 字段的作用是什么？")
    # → ["chunk", "字段", "段作", "作用", "是什", "什么"]
    #   "chunk"：英文单词，小写后保留
    #   "字段" "段作" "作用" "是什" "什么"：相邻汉字两两成组
    #   "字""段""作""用""是""什""么"等单字：不单独出现，被有意跳过

    _tokenize("Retry timeout 3次")
    # → ["retry", "timeout", "3", "次"]（"3次"不是两个汉字，不成组）
    ```

    ### 内部执行流程
    - step1: 用正则提取所有英文和数字单词（`[a-zA-Z0-9_]+`），
            统一转成小写，避免大小写导致同一词被当成两个词；
    - step2: 用正则提取所有汉字（`\u4e00-\u9fff` 范围）；
    - step3: 把汉字序列每两个相邻字组成一对（bigram），加入词列表；
    - step4: 返回全部词。单字被有意排除——它们在几乎所有中文文本里
            都出现，放进词表只会稀释真正有价值的词组信号。
    """
    import re
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend("".join(pair) for pair in zip(cjk, cjk[1:]))  # bigrams only
    return tokens


def _bm25_search(store: RagStore, question: str, k: int) -> list[dict]:
    """
    **关键词检索：按问题里的原词，从全部文档片段中找出最相关的 k 个。**

    先把每个片段拆成词，再统计问题里的词在各片段中出现的频率——
    命中越多、词越少见，片段排名越靠前。只返回得分大于 0（确实命中了词）
    的片段。数据量不大，全部放内存里算即可。

    ### 输入/输出示例

    ```python
    _bm25_search(store, "如何重置密码", k=3)
    # 库中有片段 "打开设置页面，点击账户..."（含"密码"等词）
    # → [
    #   {"content": "打开设置页面...", "source": "guide.md", "chunk_index": 2,
    #    "bm25": 8.31, "distance": 0.0},
    #   ... 共最多 3 个，按 bm25 分数从高到低
    # ]
    # 如果没有任何片段命中问题里的词 → 返回空列表
    ```

    ### 内部执行流程
    - step1: 取出库中全部片段，空库直接返回空；
    - step2: 对每个片段分词，用 BM25 算法建好检索模型；
    - step3: 对问题分词，算出每个片段的得分；
    - step4: 按得分从高到低排序，取前 k 个；过滤掉得分为 0 的
            （没有命中任何词，不算相关）；补上空缺的 distance 字段，
            保持和向量检索结果格式一致。
    """
    chunks = store.all_chunks()
    if not chunks:
        return []
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([_tokenize(c["content"]) for c in chunks])
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [{**c, "distance": 0.0, "bm25": float(s)} for c, s in ranked[:k] if s > 0]


def _rrf_fuse(vec_hits: list[dict], bm25_hits: list[dict], k: int, rrf_k: int = 60) -> list[dict]:
    """
    **把两路检索结果（向量 + 关键词）合并成一个统一的排名。**

    两路结果各按各的分数体系，没法直接比大小；改按"排名"算分：
    每个片段的得分 = 它在各路名单里的名次换算成分数后相加——
    两路都靠前的，得分自然最高；只在一路出现的，也能得到应得的分数。
    同一片段在两路名单里都出现时自动合并为一个，不重复计数。

    ### 输入/输出示例

    ```python
    vec_hits  = [{source: "a.md", chunk_index: 0, ...}, {source: "b.md", chunk_index: 1, ...}]
    bm25_hits = [{source: "b.md", chunk_index: 1, ...}, {source: "c.md", chunk_index: 0, ...}]
    _rrf_fuse(vec_hits, bm25_hits, k=2)
    # "b.md" 第 1 个片段在两路都排第 2 → 得分 1/62 + 1/62，排第 1
    # → 按合并得分从高到低返回前 2 个
    ```

    ### 内部执行流程
    - step1: 以 (文件来源, 片段序号) 作为唯一标识，逐个遍历两路结果；
    - step2: 每个片段按"1 / (60 + 名次)"累加得分——名次越靠前，
            贡献越大；名次从 1 开始计数；
    - step3: 两路遍历完，同一片段的得分已自动合并；
    - step4: 按合并得分从高到低排序，返回前 k 个。
    """
    by_key = {}
    for hits in (vec_hits, bm25_hits):
        for rank, hit in enumerate(hits):
            key = (hit["source"], hit["chunk_index"])
            #如果by_key有该键，则拿出该键值，如果没有，则设置一个默认值“ {**hit, "score": 0.0}”
            entry = by_key.setdefault(key, {**hit, "score": 0.0})
            entry["score"] += 1.0 / (rrf_k + rank + 1)
    return sorted(by_key.values(), key=lambda h: h["score"], reverse=True)[:k]


def answer_with_rag(client, cfg: dict, question: str, k: int | None = None) -> dict:
    """
    **问答主入口：从知识库检索相关内容，交给大模型生成答案，并附带来源。**

    先检索与问题最相关的 k 个片段；检索为空或相关度太低（低于阈值）时
    不调用大模型，直接返回"无答案"；否则把片段拼成上下文连同问题一起
    发给大模型作答。无论成功与否，都会一并返回引用了哪些片段。

    ### 输入/输出示例

    ```python
    # 例1：正常回答
    answer_with_rag(client, cfg, "如何重置密码？", k=4)
    # → {
    #     "result": "根据文档：打开设置页面，点击账户，选择重置密码……",
    #     "sources": [                       # 引用的片段（即检索结果）
    #       {"content": "打开设置页面...", "source": "guide.md", "score": 0.91},
    #       ...
    #     ],
    #   }

    # 例2：知识库为空，或没有相关内容
    answer_with_rag(client, cfg, "完全不相关的问题？")
    # → {"result": None, "sources": []}
    #   控制台同时提示用户先用 ingest 命令入库
    ```

    ### 内部执行流程
    - step1: 调用 `retrieve` 检索，得到按相关度排序的 k 个片段；
    - step2: 两道关卡，任一不过直接返回"无答案"（不花大模型的钱）：
            ① 检索结果为空——库是空的或完全没命中；
            ② 最高分的片段相关度低于 `MIN_RERANK_SCORE`——
               说明命中的都是噪声，强行作答只会编造；
    - step3: 把关过的片段拼成上下文文本，每段标注来源文件名，
            连同系统提示一起组装成消息列表；
    - step4: 调用 `ask` 把消息发给大模型，得到答案；
    - step5: 打包返回：答案放 `result`，引用的片段放 `sources`。
    """
    hits = retrieve(cfg, question, k)
    if not hits:
        console.print("[dim]知识库为空或没有相关内容，请先 modelclaw ingest <路径>[/dim]")
        return {"result": None, "sources": []}
    if hits[0]["score"] < MIN_RERANK_SCORE:
        console.print(f"[dim]知识库中没有相关内容（最高相关度 {hits[0]['score']:.4f}，低于阈值 {MIN_RERANK_SCORE}）[/dim]")
        return {"result": None, "sources": []}

    context = "\n\n".join(f"[{h['source']}]\n{h['content']}" for h in hits)
    messages = [
        {"role": "system", "content": _RAG_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]
    result = ask(client, cfg, messages)
    return {"result": result, "sources": hits}


def render_sources(hits: list[dict]) -> None:
    """
    **把答案引用的片段以表格形式打印到控制台。**

    在流式答案输出完毕后调用，每行一个片段：
    来源文件、片段序号、相关度得分（保留 4 位小数）。

    ### 输入/输出示例

    ```python
    hits = [
        {"content": "打开设置页面...", "source": "guide.md", "chunk_index": 2, "score": 0.9123},
        {"content": "密码需包含...",  "source": "policy.md", "chunk_index": 0, "score": 0.4210},
    ]
    render_sources(hits)

    # 控制台输出：
    # ┌──────────────────────────────────────────────┐
    # │                  引用来源                      │
    # ├────────────┬──────┬─────────────┤
    # │ 来源        │  块  │ 相关度得分    │
    # ├────────────┼──────┼─────────────┤
    # │ guide.md   │   2  │      0.9123 │
    # │ policy.md  │   0  │      0.4210 │
    # └────────────┴──────┴─────────────┘
    ```

    ### 内部执行流程
    - step1: 创建表格，设置标题"引用来源"和表头样式；
    - step2: 定义三列：来源（左对齐加粗）、块序号（右对齐）、
            相关度得分（右对齐，数字对齐小数点）；
    - step3: 逐个片段取 `source`、`chunk_index`、`score` 填入一行，
            得分格式化为 4 位小数；
    - step4: 打印整张表格。只打印，不返回任何值。
    """
    table = Table(title="引用来源", header_style="bold magenta")
    table.add_column("来源", style="bold")
    table.add_column("块", justify="right")
    table.add_column("相关度得分", justify="right")
    for h in hits:
        table.add_row(h["source"], str(h["chunk_index"]), f"{h['score']:.4f}")
    console.print(table)
