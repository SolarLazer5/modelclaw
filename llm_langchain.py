"""LangChain engine for the chat path (M4a0): ChatOpenAI + LCEL, feature-parity
with the native engine in api_client.py.

Two engines, one contract:
    api_client.ask()      -> native engine (raw OpenAI SDK)
    llm_langchain.ask()   -> LangChain engine (this module), selected via
                             config llm.engine = "langchain"

The subclass patch: langchain-openai 1.7 drops the provider-specific
`reasoning_content` delta field (it only handles first-party formats), so
DeepSeek's thinking would be silently swallowed. ModelScopeChatOpenAI rescues
it into additional_kwargs by overriding the chunk conversion hook.
"""

import logging
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_ROLE_TO_CLASS = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}


class ModelScopeChatOpenAI(ChatOpenAI):
    """ChatOpenAI patched to pass through DeepSeek's reasoning_content."""

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        """
        **流式聊天中，把 API 返回的原始数据块（chunk）转换成 LangChain
        标准格式的生成块——并顺手救回被官方实现丢掉的"思维链"内容。**

        背景：DeepSeek 这类模型的回复分两段——先输出思维链（reasoning_content
        字段），再输出正式答案（content 字段）。但 langchain-openai 1.7 版的
        官方转换代码只搬运 content 等标准字段，reasoning_content 这种
        厂商私有字段被直接丢弃——用户就看不到模型"思考过程"了。
        本函数在官方转换结果之上，把思维链片段塞进附加字段区
        （additional_kwargs），下游的 stream_events 就能原样取到并显示。

        ### 输入/输出示例

        ```python
        # API 流里来了一个"思维链"数据块（内容简化示意）
        chunk = {
            "choices": [{
                "delta": {
                    "reasoning_content": "让我先分析 retry 配置…",   # 思维链片段
                    # 注意：没有 "content" 字段
                }
            }]
        }
        self._convert_chunk_to_generation_chunk(chunk, default_class, info)
        # → 返回官方转换好的生成块，且其附加字段区多了：
        #   gen_chunk.message.additional_kwargs["reasoning_content"] == "让我先分析 retry 配置…"

        # API 流里来了一个"正式答案"数据块
        chunk = {"choices": [{"delta": {"content": "retry 组有 4 个字段…"}}]}
        self._convert_chunk_to_generation_chunk(chunk, default_class, info)
        # → 正常转换（由官方代码处理 content），附加字段区不受影响

        # 数据块结构异常（没有 choices / 不是字典等）
        self._convert_chunk_to_generation_chunk(chunk, default_class, info)
        # → 依然返回官方转换结果，思维链救取失败但绝不因此抛异常
        ```

        ### 内部执行流程
        - step1: 先调用官方转换（super()），把 content 等标准字段的转换
             工作原样交给 langchain-openai 处理，自己不重复实现；
        - step2: 从原始数据块里尝试取出思维链片段——数据块可能结构异常
                （缺字段、类型不对），用 try/except 兜住，取不到就当作
                没有思维链，继续走；
        - step3: 只有"确实有思维链内容"且"官方转换成功"时，才把它写入
                生成块的附加字段区（additional_kwargs——LangChain 专门
                留给厂商私有字段的位置），否则什么都不做；
        - step4: 返回生成块，交给流式管线的下一环。
        """
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        # Rescue the provider-specific reasoning delta that stock
        # langchain-openai drops during chunk conversion.
        try:
            reasoning = chunk.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
        except (AttributeError, IndexError, TypeError):
            reasoning = None
        if reasoning and gen_chunk is not None:
            gen_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return gen_chunk


def create_chat_model(cfg: dict) -> ModelScopeChatOpenAI:
    """    
    **根据 `api` 配置组装一个 LangChain 聊天模型（带思维链补丁的版本）。**

    模型名、服务地址、采样参数全部来自配置的 `api` 段（都有默认值）；
    密钥只认环境变量 MODELSCOPE_API_KEY，缺失时直接报错提示去 .env 配置，
    不发出必然失败的请求。固定开启流式输出（本项目的交互形态就是流式）。

    ### 输入/输出示例

    ```python
    # 配置：{"api": {"model": "deepseek-r1", "base_url": "https://...",
    #               "temperature": 0.7, "max_tokens": 2048, "timeout": 30}}
    # 环境变量 MODELSCOPE_API_KEY 已配置
    create_chat_model(cfg)
    # → ModelScopeChatOpenAI 实例（流式开启、参数已注入）

    # 环境变量没配密钥
    create_chat_model(cfg)
    # → 抛出 RuntimeError: 未找到 MODELSCOPE_API_KEY，请在 .env 文件中配置 ModelScope Token
    ```

    ### 内部执行流程
    - step1: 取出配置的 `api` 段，从环境变量读密钥；
    - step2: 密钥为空 → 立即报错，报文指明配置位置（.env）；
    - step3: 用配置参数（未配则用默认值）创建模型实例——注意用的是
            打了思维链补丁的 ModelScopeChatOpenAI，而非原版 ChatOpenAI；
    - step4: 返回实例。
    """
    api_cfg = cfg["api"]
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("未找到 MODELSCOPE_API_KEY，请在 .env 文件中配置 ModelScope Token")
    return ModelScopeChatOpenAI(
        model=api_cfg["model"],
        base_url=api_cfg["base_url"],
        api_key=api_key,
        temperature=api_cfg.get("temperature", 0.7),
        max_tokens=api_cfg.get("max_tokens", 2048),
        timeout=api_cfg.get("timeout", 30),
        streaming=True,
    )


def create_llm(cfg: dict):
    """
    **在聊天模型外套一层"失败自动重试"，返回带重试能力的完整模型。**

    重试参数全部来自配置的 `retry` 段（重试次数、初始等待、
    退避倍数、最长等待），不写在代码里——调行为改配置即可。
    重试机制用的是 LangChain 内置的 with_retry（底层是 tenacity），
    指数退避加抖动：等待 1s → 2s → 4s …封顶 30s。

    ### 输入/输出示例

    ```python
    # 配置：{"retry": {"max_attempts": 5, "initial_wait": 1,
    #                 "backoff_multiplier": 2, "max_wait": 30}}
    create_llm(cfg)
    # → 带重试的模型实例：请求失败时自动重试，最多 5 次，
    #   等待时长 1s→2s→4s→…→30s 封顶

    # 配置里没有 retry 段 → 用默认值（5 次、1s 起步、2 倍退避、30s 封顶）
    ```

    ### 内部执行流程
    - step1: 取出配置的 `retry` 段（可整个缺省，取默认值）；
    - step2: 先创建底层聊天模型（create_chat_model）；
    - step3: 调用 with_retry 包上重试层，参数逐项映射：
            最大尝试次数、初始等待、指数基数、等待封顶；
    - step4: 返回带重试的模型。
    """
    retry_cfg = cfg.get("retry", {})
    return create_chat_model(cfg).with_retry(
        stop_after_attempt=retry_cfg.get("max_attempts", 5),
        exponential_jitter_params={
            "initial": retry_cfg.get("initial_wait", 1),
            "exp_base": retry_cfg.get("backoff_multiplier", 2),
            "max": retry_cfg.get("max_wait", 30),
        },
    )


def _to_lc_messages(messages: list[dict]) -> list[BaseMessage]:
    """
    **把 OpenAI 风格的消息字典列表，翻译成 LangChain 的消息对象列表。**

    项目内部统一用 `{role, content}` 字典表示对话（原生引擎直接吃这种），
    而 LangChain 引擎需要自己的消息对象（HumanMessage/AIMessage/
    SystemMessage）。本函数做两者之间的格式翻译。

    ### 输入/输出示例

    ```python
    _to_lc_messages([
        {"role": "system",    "content": "你是问答助手"},
        {"role": "user",      "content": "retry 组有哪些字段？"},
        {"role": "assistant", "content": "有 4 个字段…"},
    ])
    # → [SystemMessage("你是问答助手"),
    #    HumanMessage("retry 组有哪些字段？"),
    #    AIMessage("有 4 个字段…")]
    ```

    ### 内部执行流程
    - step1: 逐条取出消息字典，按 role 查转换表
            （_ROLE_TO_CLASS：user→HumanMessage、assistant→AIMessage、
            system→SystemMessage），把 content 包进对应的消息类；
    - step2: 返回翻译好的消息对象列表。
    """
    return [_ROLE_TO_CLASS[m["role"]](content=m["content"]) for m in messages]


def stream_events(cfg: dict, messages: list[dict]):
    """
    **流式调用模型，把输出逐片段吐成两类事件：(种类, 文本)。**

    种类只有两种："reasoning"（思维链片段）和 "answer"（正式答案片段）。
    本函数是流式数据的"分发中枢"——CLI 打印（ask）和服务端 SSE 推送
    （server.py）都消费这一份事件流，各自决定怎么展示。
    思维链片段来自模型补丁写入的附加字段区（additional_kwargs）。

    ### 输入/输出示例

    ```python
    for kind, text in stream_events(cfg, messages):
        ...
    # 模型先输出思维链时依次产生：
    #   ("reasoning", "让我先分析…")  ("reasoning", "retry 配置在…") …
    # 随后输出正式答案：
    #   ("answer", "retry 组有 4 个字段：")  ("answer", "max_attempts…") …
    ```

    ### 内部执行流程
    - step1: 创建带重试的模型（create_llm），把消息翻译后发起流式调用；
    - step2: 每收到一个输出片段，先查附加字段区有没有思维链内容——
            有就作为 ("reasoning", 片段) 吐出；
    - step3: 否则取标准内容字段——非空就作为 ("answer", 片段) 吐出；
            两者都为空（如只有工具调用等元信息）则跳过这一片段；
    - step4: 片段逐个产生，直到模型流结束。
    """
    llm = create_llm(cfg)
    logger.info("Sending request (langchain engine): model=%s", cfg["api"]["model"])
    for chunk in llm.stream(_to_lc_messages(messages)):
        reasoning_chunk = chunk.additional_kwargs.get("reasoning_content")
        answer_chunk = chunk.content if isinstance(chunk.content, str) else ""
        if reasoning_chunk:
            yield "reasoning", reasoning_chunk
        elif answer_chunk:
            yield "answer", answer_chunk


def ask(cfg: dict, messages: list[dict]) -> dict:
    """
    **LangChain 引擎的聊天入口：边流式输出边打印思维链和答案，
    最后把完整结果拼好返回。**

    对外契约与原生引擎的 api_client.ask 完全一致：同样的输入、
    同样的终端展示（Thinking / Final Answer 两段标题）、
    同样的返回值 {"reasoning": 完整思维链, "answer": 完整答案}——
    上层调用方切换引擎时无感知。

    ### 输入/输出示例

    ```python
    ask(cfg, [{"role": "user", "content": "retry 组有哪些字段？"}])
    # 终端边生成边打印：
    #   === Thinking ===
    #   让我先分析 retry 配置…
    #   === Final Answer ===
    #   retry 组有 4 个字段：max_attempts…
    # → 返回 {"reasoning": "让我先分析 retry 配置…",
    #         "answer": "retry 组有 4 个字段：max_attempts…"}
    ```

    ### 内部执行流程
    - step1: 消费流事件（stream_events），按种类分流；
    - step2: 思维链片段：首次出现时打印「=== Thinking ===」标题，
            之后逐片段打印并累积；
    - step3: 答案片段：首次出现时打印「=== Final Answer ===」标题，
            之后逐片段打印并累积——两类内容分段展示，互不混杂；
    - step4: 流结束后，把累积的片段各自拼接成完整字符串，
            按契约组装成字典返回。
    """
    reasoning_parts = []
    answer_parts = []
    thinking_header_printed = False
    answer_header_printed = False

    for kind, text in stream_events(cfg, messages):
        if kind == "reasoning":
            if not thinking_header_printed:
                print("\n\n === Thinking ===\n")
                thinking_header_printed = True
            print(text, end="", flush=True)
            reasoning_parts.append(text)
        else:
            if not answer_header_printed:
                print("\n\n === Final Answer ===\n")
                answer_header_printed = True
            print(text, end="", flush=True)
            answer_parts.append(text)

    print()
    return {"reasoning": "".join(reasoning_parts), "answer": "".join(answer_parts)}
