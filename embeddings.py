"""Embedding + rerank wrappers: pluggable model sources, dynamic dimensions.

Sources (rag.embedding_source):
  - "modelscope":   local model via the modelscope SDK cache
                    (e.g. Qwen/Qwen3-Embedding-0.6B, 1024-dim)
  - "huggingface":  local model via the hf-mirror endpoint
                    (e.g. BAAI/bge-small-zh-v1.5, 512-dim)
  - "dashscope":    cloud API (Aliyun Maas gateway) — no local compute,
                    key from DASHSCOPE_API_KEY in .env

The embedding dimension is NEVER hardcoded: it is derived from the loaded
model, and the vector stores learn it from the first batch of vectors they
receive (see rag_store / pgvector_store).

The reranker (cross-encoder) stays local: bge-reranker-base.

Models load lazily on first use, so CLI commands that don't need embeddings
never pay the torch startup cost.
"""

import logging
import os
import time

import requests

# Must be set before huggingface_hub is imported anywhere — HF mirror for China.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_DASHSCOPE_URL = (
    "https://llm-gd0zjpwlkp8emiz4.cn-beijing.maas.aliyuncs.com"
    "/api/v1/services/embeddings/text-embedding/text-embedding"
)
_DASHSCOPE_BATCH = 10  # texts per API call

_model = None
_model_key = None


def _resolve_model_path(model_name: str, source: str) -> str:
    """把一个"模型名称"解析成"本地文件路径"，供 SentenceTransformer 加载模型时使用。"""
    if source == "modelscope":
        # Uses the modelscope cache; an already-downloaded snapshot returns
        # instantly (no network).
        from modelscope.hub.snapshot_download import snapshot_download
        return snapshot_download(model_name)
    return model_name  # huggingface hub id (downloaded via HF_ENDPOINT mirror)


def _get_model(model_name: str, source: str):
    """加载SentenceTransformer,确保整个进程中 SentenceTransformer 模型只被加载一次"""
    global _model, _model_key
    key = (model_name, source)
    if _model is None or _model_key != key:
        from sentence_transformers import SentenceTransformer
        path = _resolve_model_path(model_name, source)
        logger.info("Loading embedding model %s (source=%s)...", model_name, source)
        _model = SentenceTransformer(path)
        _model_key = key
    return _model


def get_embedding_dim(rag_cfg: dict) -> int:
    """
    **获取调用向量模型的维度**
    ## 向量的维度
    [0.012, -0.339, 0.881, ..., 0.054]\n
    这就是一个向量,维度就是里面浮点数的数量\n

    """
    if rag_cfg.get("embedding_source") == "dashscope":
        return len(_embed_via_dashscope(["dimension probe"], rag_cfg)[0])
    model = _get_model(rag_cfg.get("embedding_model", DEFAULT_MODEL),
                       rag_cfg.get("embedding_source", "huggingface"))
    return model.get_sentence_embedding_dimension()


def _l2_normalize(vector: list[float]) -> list[float]:
    """归一化工具函数"""
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else vector


def _embed_via_dashscope(texts: list[str], rag_cfg: dict) -> list[list[float]]:
    """
    **通过兼容 DashScope 协议的 HTTP 接口，把一批文本转成 embedding 向量**\n
    ## 输入示例：
     ```python
    texts = ["苹果是水果", "猫是动物"]
    rag_cfg = {
        "embedding_source": "dashscope",        # 来源标识
        "embedding_model": "text-embedding-v3", # 向量模型名
        "dashscope_url": "https://...",         # 兼容 DashScope 协议的网关地址
    }
    ```
    ## 输出示例：
     ```python
    [
        [0.012, -0.339, 0.881, ..., 0.054],   # "苹果是水果" 的归一化向量（1024 维）
        [-0.221, 0.447, 0.103, ..., -0.668],  # "猫是动物" 的归一化向量（1024 维）
    ]
    ```
    
    ## 执行流程:
    - step1:读取`rag_cfg`配置和api密钥,得到目标url和目标向量模型\n
    - step2:构造http请求,调用向量模型,分批次发送得到返回的向量,同时将每个批次得到的向量进行`归一化处理`\n
    - step3:返回`向量list[list[float]]`\n

    ## 异常处理:
    - 未配置 `DASHSCOPE_API_KEY` 时抛出 `RuntimeError`
    - 某一批重试 3 次后仍失败时, 抛出最后一次的异常

    ## 归一化：
    每个向量"缩放"成同样长度（长度为 1），使满足:\n
    `√(0.012² + (-0.339)² + 0.881² + ...) = 1`

    """
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY，请在 .env 文件中配置")
    url = rag_cfg.get("dashscope_url", DEFAULT_DASHSCOPE_URL)
    model = rag_cfg.get("embedding_model", DEFAULT_MODEL)

    vectors: list[list[float]] = []
    for i in range(0, len(texts), _DASHSCOPE_BATCH):
        batch = texts[i:i + _DASHSCOPE_BATCH]
        payload = {"model": model, "input": {"texts": batch}}
        last_exc: Exception | None = None
        for attempt in range(3):  # 指数退避 1s→2s，对齐项目重试哲学
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=30,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"embedding API {resp.status_code}: {resp.text[:200]}")
                embeddings = sorted(
                    resp.json()["output"]["embeddings"], key=lambda e: e["text_index"]
                )
                vectors.extend(_l2_normalize(e["embedding"]) for e in embeddings)
                last_exc = None
                break
            except (requests.RequestException, RuntimeError) as exc:
                last_exc = exc
                logger.warning("embedding API attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        if last_exc is not None:
            raise last_exc
    logger.info("Embedded %d texts via dashscope (%s)", len(texts), model)
    return vectors


def embed_texts(texts: list[str], rag_cfg: dict) -> list[list[float]]:
    """
    **把一批文本转成 embedding 向量**\n
    ## 输入示例：
     ```python
    texts = ["苹果是水果", "猫是动物"]
    rag_cfg = {
        "embedding_source": "dashscope",        # 来源标识
        "embedding_model": "text-embedding-v3", # 向量模型名
        "dashscope_url": "https://...",         # 兼容 DashScope 协议的网关地址
    }
    ```
    ## 输出示例：
     ```python
    [
        [0.012, -0.339, 0.881, ..., 0.054],   # "苹果是水果" 的归一化向量（1024 维）
        [-0.221, 0.447, 0.103, ..., -0.668],  # "猫是动物" 的归一化向量（1024 维）
    ]
    ```

    """
    if not texts:
        return []
    if rag_cfg.get("embedding_source") == "dashscope":
        return _embed_via_dashscope(texts, rag_cfg)
    model = _get_model(rag_cfg.get("embedding_model", DEFAULT_MODEL),
                       rag_cfg.get("embedding_source", "huggingface"))
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str, rag_cfg: dict) -> list[float]:
    """
    **将单个文本转为向量**\n
    ## 输入示例：
     ```python
    texts = "苹果是水果"
    rag_cfg = {
        "embedding_source": "dashscope",        # 来源标识
        "embedding_model": "text-embedding-v3", # 向量模型名
        "dashscope_url": "https://...",         # 兼容 DashScope 协议的网关地址
    }
    ```
    ## 输出示例：
     ```python
    [
        [0.012, -0.339, 0.881, ..., 0.054],   # "苹果是水果" 的归一化向量
    ]
    ```

    """
    instruction = rag_cfg.get("query_instruction", "")
    if instruction:
        text = instruction + text
    return embed_texts([text], rag_cfg)[0]


# ---------------------------------------------------------------------------
# reranker (cross-encoder): the second stage of retrieve-then-rerank
# ---------------------------------------------------------------------------

_reranker = None
_reranker_name = None


def _get_reranker(name: str):
    """懒加载 CrossEncoder."""
    global _reranker, _reranker_name
    if _reranker is None or _reranker_name != name:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model %s ...", name)
        _reranker = CrossEncoder(name)
        _reranker_name = name
    return _reranker


def rerank(question: str, candidates: list[dict], top_k: int, model_name: str = DEFAULT_RERANKER) -> list[dict]:
    """
    **用交叉编码器（cross-encoder）重新给「问题-候选片段」打分，返回分数最高的 top_k 个候选。**

    双编码器（bi-encoder）检索速度快但较粗糙；交叉编码器把问题和候选片段
    放在一起逐对精读打分，对小语料、关键词密集的问题明显更准确。
    本函数常作为 RAG 检索的第二步：先粗排召回 candidates，再精排筛 top_k。

    ### 输入示例

    ```python
    question = "如何重置密码？"
    candidates = [
        {"content": "打开设置页面，点击账户...", "source": "user_guide.pdf", "page": 3},
        {"content": "密码需包含大小写字母...", "source": "security_policy.pdf", "page": 1},
        {"content": "如需帮助请联系客服...", "source": "faq.pdf", "page": 7},
    ]
    top_k = 2
    # model_name 不传则用默认重排序模型
    ```

    ### 输出示例

    ```python
    [
        {"content": "打开设置页面，点击账户...", "source": "user_guide.pdf", "page": 3, "score": 0.91},
        {"content": "密码需包含大小写字母...", "source": "security_policy.pdf", "page": 1, "score": 0.42},
    ]
    ```

    返回的列表按 `score` 从高到低排序，最多 `top_k` 个元素；
    每个元素是原候选字典加上一个 `score` 字段，原有字段全部保留。

    ### 执行流程
    - step1: 空候选保护——`candidates` 为空时直接返回空列表，不加载模型；
    - step2: 通过 `_get_reranker(model_name)` 懒加载/复用交叉编码器模型，
             把 `question` 与每个候选的 `content` 组成 `(问题, 片段)` 对，
             调用 `predict` 一次性得到每个对的相关性分数；
    - step3: 把每个候选字典与对应分数合并（新增 `score` 字段），
             按分数从高到低排序；
    - step4: 切片取前 `top_k` 个返回。
    """
    if not candidates:
        return []
    scores = _get_reranker(model_name).predict(
        [(question, c["content"]) for c in candidates]
    )
    ranked = sorted(
        ({**c, "score": float(s)} for c, s in zip(candidates, scores)),
        key=lambda h: h["score"],
        reverse=True,
    )
    return ranked[:top_k]
