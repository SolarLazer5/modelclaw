"""Context engineering: token counting, window trimming, rolling summarization.

Pipeline applied before every model call (see build_context):

    raw history (SQLite)
      -> rolling summarization when over `summary_trigger_tokens`
         (old turns are compressed by the LLM into sessions.summary and
          deleted from the messages table — ConversationSummaryBufferMemory style)
      -> trim_messages hard cap at `max_context_tokens`
      -> assembled as [system] + [summary SystemMessage] + trimmed messages

All thresholds come from the `memory` config group, so behavior is tunable
without touching code.
"""

import logging

import tiktoken
from langchain_core.messages import trim_messages

from api_client import complete

logger = logging.getLogger(__name__)

# ModelScope model ids are unknown to tiktoken's model registry, so count with
# the fixed cl100k_base encoding (same trick as the learning notes) — close
# enough for context budgeting.
_encoding = tiktoken.get_encoding("cl100k_base")

_SUMMARY_PROMPT = (
    "请把以下对话浓缩成一段简短摘要，保留关键信息（人名、约定、事实、待办）。"
    "如果提供了已有摘要，请把新内容合并进去，输出一段完整的新摘要。只输出摘要本身。"
)

# LangChain message type -> OpenAI role
_ROLE_BY_TYPE = {"human": "user", "ai": "assistant", "system": "system"}


def count_tokens(messages) -> int:
    """近似计算 token 数目，适合三种调用场合：
    1. 直接传 str 字符串计算
    2. 计算标准 OpenAI API 风格消息格式，即::

           [
               {"id": 1, "role": "user",      "content": "你好"},
               {"id": 2, "role": "assistant", "content": "你好！有什么可以帮你？"},
               {"id": 3, "role": "user",      "content": "帮我写个脚本"},
           ]

    3. 支持传入 BaseMessage 对象计算 token 数目（用于 trim_messages 的回调）
    """
    #如果messages是一个str，直接计算返回
    if isinstance(messages, str):
        return len(_encoding.encode(messages))
    total = 3  # every reply is primed with <|start|>assistant<|message|>
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content") or ""
        elif isinstance(m, str):
            content = m
        else:  # LangChain BaseMessage
            content = m.content or ""
            if not isinstance(content, str):
                content = str(content)
        total += 3 + len(_encoding.encode(content))  # ~3 tokens of role overhead
    return total


def _split_head_tail(messages: list[dict], keep_recent_turns: int) -> tuple[list[dict], list[dict]]:
    """
    # 切割OpenAI API风格的`messages`为head message列表和tail message列表\n
    ## 参数:\n
    1. `messages`: OpenAI API风格的messages列表(内部每一个元素都是一个dict)
    2. `keep_recent_turns`: 切割用户轮数

    ## 返回值:
    1. `tuple[list[dict], list[dict]]`:\n
    前者为head message,后者为tail message\n
    如果当前messages的用户轮数小于需要满足的用户轮数(keep_recent_turns),则返回的head message为空
    """
    #生成一个列表，列表按顺序保存用户消息的下标值
    #(方便后续按照user下标切，保证切割后的head message和tail message是user问题开头)
    user_positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
    #比较当前用户消息轮数和目标轮数
    if len(user_positions) <= keep_recent_turns:
        return [], messages#直接返回，不需要切割
    #取第倒数“目标轮数”个的用户消息下标值，并从该位置切割message
    cut = user_positions[-keep_recent_turns]
    return messages[:cut], messages[cut:]
    ##知识点：
    # 1，user_positions[-keep_recent_turns]
    # 代表取倒数第几个元素(某一个，而不是重新生成列表)


def _summarize(client, cfg: dict, old_summary: str, head: list[dict]) -> str:
    """压缩上下文，返回当前上下文的摘要"""
    #生成要压缩的上下文的完整字符串
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in head)
    #准备压缩上下文的模型提示词
    prompt = _SUMMARY_PROMPT
    #如果已有压缩后的摘要，则将已有摘要追加到提示词，让模型融合之后压缩的上下文摘要和已有摘要
    if old_summary:
        prompt += f"\n\n已有摘要:\n{old_summary}"
    #调用模型，传入上下文，返回提炼的摘要内容
    return complete(client, cfg, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript},
    ])
    ##知识点：
    ## 1，(f"{m['role']}: {m['content']}" for m in head)
    ## (f"{m['role']}: {m['content']}" for m in head)将会返回一个生成器，生成器是一个可迭代对象，
    ## 每次迭代时才会实际执行计算一次。以下是其他家族
    ## [x for x in head]     # 列表推导式 → list
    ## (x for x in head)     # 生成器表达式 → generator
    ## {x for x in head}     # 集合推导式 → set（自动去重）
    ## {k: v for k, v in ...}  # 字典推导式 → dict（注意有冒号）
    ## 2.join方法
    ## 把一堆字符串用分隔符首尾相接成一个大字符串。
    ## ", ".join(["a", "b", "c"])   # → "a, b, c"
    ## "\n".join(["a", "b", "c"])   # → "a\nb\nc"（换行连接）  


def build_context(client, cfg: dict, store, session_id: str, system: str | None = None) -> list[dict]:
    """
    # 根据上下文管理策略，返回某个会话处理后的上下文\n
    ## 参数:\n
    1. `store`: 后端存储对象(可以是sqlite,也可以postgresql)
    2. `cfg`: 上下文管理策略
    3. `session_id`: 要进行处理的上下文所属会话

    ## 返回值:\n
    1. `上下文(list[dict])`: 系统提示词 + 摘要（可能为空） + 对话历史
    
    ## 执行过程:\n
    **step1**. 先从 `store` 中取出当前 `上下文` \n
    **step2**. 然后将取出的 `上下文` 根据 `cfg` 上下文管理策略，进行压缩（提炼摘要）和裁剪\n
    **step3**. 最后返回处理后的 `上下文` (list[dict])\n
    """

    #读取上下文窗口管理策略配置信息
    mem_cfg = cfg.get("memory", {})
    max_context = mem_cfg.get("max_context_tokens", 4000)
    trigger = mem_cfg.get("summary_trigger_tokens", 6000)
    keep_turns = mem_cfg.get("keep_recent_turns", 4)

    #从数据库中获取会话的messages和summary
    messages = store.get_messages(session_id)
    summary = store.get_summary(session_id)

    # 第一步：先计算当前会话的messages是否需要进行上下文压缩，得到压缩后的摘要
    if messages and count_tokens(messages) > trigger:
        #将当前messages分隔为head messages和tail messages
        head, tail = _split_head_tail(messages, keep_turns)
        #如果head messages不为空，则代表需要进行上下文压缩
        if head:
            logger.info("Summarizing %d old messages for session %s", len(head), session_id)
            #对head messages进行提炼摘要（获得summary）
            summary = _summarize(client, cfg, summary, head)
            #更新数据库当前会话的summary字段
            store.set_summary(session_id, summary)
            #删除数据库当前会话被压缩的head messages，
            #head[-1]["id"]代表被压缩的上下文的最后一条消息id，删除所有小于等于这个id的message行
            store.delete_messages_before(session_id, head[-1]["id"])
            messages = tail

    # 第二步：获取裁剪后的上下文
    trimmed = trim_messages(
        [{"role": m["role"], "content": m["content"]} for m in messages],   #OpenAI API风格对话历史
        max_tokens=max_context,     #允许的最大上下文token数目
        strategy="last",            #尾部裁剪策略
        token_counter=count_tokens, #token计算策略（回调函数）
        allow_partial=False,        #是否允许截断消息
        start_on="human",           #裁剪后的上下文窗口的第一条消息类型 
    )

    # 第三步：将裁剪后的上下文对话历史，转成OpenAI API风格对话历史
    trimmed_dicts = [
        {"role": _ROLE_BY_TYPE.get(m.type, m.type), "content": m.content} for m in trimmed
    ]
    ## 知识点：
    ## 1,为什么进行_ROLE_BY_TYPE.get(m.type, m.type)转换？
    ## trim_messages返回的是一个BaseMessage列表，消息类型为human，ai，system
    ## 而OpenAI API风格的消息类型为user，assistant，system，所以要进行转换

    #如果裁剪前的messages的最后一条消息被trim_messages裁剪，则复原追加
    if messages and (not trimmed_dicts or trimmed_dicts[-1]["content"] != messages[-1]["content"]):
        trimmed_dicts.append({"role": messages[-1]["role"], "content": messages[-1]["content"]})
    ## 知识点：
    ## 1，max_tokens算的是谁？
    ## max_context_tokens 只约束历史部分,系统提示词和摘要(摘要一般加在系统提示词中)不占这 4000 的额度，
    ## 它们的长度另外叠加在上面
    ## 2，为什么需要写上面这个代码？
    ## 因为在第一步中,采用的是轮数压缩,末尾的消息是原封不动保留的,只有前面消息被总结为摘要
    ## 但是末尾的最后一条消息是有可能因为trim_messages(其中参数allow_partial=False)导致被删掉
    ## 即如果最后一条消息token数目就大于了max_tokens，那么末尾的消息就会被删掉(通常整个对话历史会变成空列表)，
    ## 那么这是无法接受的错误，所以要写这段代码防止发生这种错误

    #将系统提示词，压缩后的上下文摘要，以及对话历史放到context列表中，并返回这个上下文对象
    context = []
    if system:
        context.append({"role": "system", "content": system})
    if summary:
        context.append({"role": "system", "content": f"[此前对话摘要] {summary}"})
    context.extend(trimmed_dicts)
    return context
