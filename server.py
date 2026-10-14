"""HTTP service for modelclaw (M4d): the CLI's capabilities over FastAPI.

Endpoints:
    GET  /healthz           liveness probe
    POST /chat              one-shot chat (JSON in, JSON out)
    POST /chat/stream       one-shot chat as SSE (reasoning/answer events)
    POST /ask               RAG knowledge-base QA (+ sources)
    POST /agent             agent task (tool trace included)

The CLI stays unchanged — this server is just another client of the same
capability modules. Sessions share the same pluggable store, so a conversation
started over HTTP can be resumed from the CLI.
"""

import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api_client import create_client
from context_engine import build_context
from evaluator import predict_one
from llm_langchain import stream_events
from logger import setup_logging
from main import load_config
from session_store import create_session_store


class ChatReq(BaseModel):
    message: str
    session: str | None = None


class AskReq(BaseModel):
    question: str
    k: int | None = None


class AgentReq(BaseModel):
    task: str
    session: str | None = None


def create_app() -> FastAPI:
    """
    **FastAPI 应用工厂：装配配置、客户端、会话库，并注册全部四个 HTTP 接口。**

    为什么需要它：M4d 要把 CLI 已有的能力（聊天、流式聊天、RAG 问答、Agent）
    以 HTTP 服务形式暴露出去。这个函数是装配中心——加载配置、创建 LLM 客户端
    和会话存储，然后把四个路由（/healthz、/chat、/chat/stream、/ask、/agent）
    挂到 FastAPI 应用上。设计成"工厂函数"（返回 app 对象而不是直接启动服务）
    有两个好处：测试时可以直接调用 create_app() 拿到 app 做接口测试；启动服务
    的 uvicorn 代码单独放在 run() 里，职责分离。

    ### 输入/输出示例
    ```python
    app = create_app()
    # → 返回一个装配完成的 FastAPI 实例：
    #   配置已加载、日志已初始化、LLM 客户端和会话库已创建，
    #   四个接口已注册，可以用测试客户端或 uvicorn 直接跑
    ```

    ### 内部执行流程
    - step1: 加载 .env 环境变量，读取项目配置，按配置初始化日志
    - step2: 创建 LLM 客户端和会话存储（与 CLI 用的是同一套工厂函数，所以会话数据完全互通）
    - step3: 创建 FastAPI 应用，在内部依次定义并注册五个接口函数：/healthz 健康检查、/chat 一次性问答、/chat/stream 流式问答、/ask 知识库问答、/agent 智能体任务
    - step4: 返回装配好的 app
    """
    load_dotenv()
    cfg = load_config()
    setup_logging(cfg)
    client = create_client(cfg)
    store = create_session_store(cfg)

    app = FastAPI(title="modelclaw API")

    @app.get("/healthz")
    def healthz():
        """
        **健康检查接口：返回固定 JSON，证明服务活着。**

        为什么需要它：部署后的服务需要一条"探活"途径——负载均衡器、容器编排
        系统、监控工具会定期访问这个接口，收到正常响应就说明进程没挂。它不碰
        任何业务逻辑和外部依赖，所以永远是服务里最可靠的一个接口。

        ### 输入/输出示例
        ```python
        # GET /healthz 的响应
        {"status": "ok"}
        ```

        ### 内部执行流程
        - step1: 直接返回固定字典，无任何业务逻辑
        """
        return {"status": "ok"}

    def _chat_messages(session: str | None, message: str) -> tuple[str, list[dict]]:
        """
        **聊天接口的公共准备步骤：建会话、存用户消息、构建上下文，返回会话 id 和消息列表。**

        为什么需要它：/chat 和 /chat/stream 两个接口的第一步完全一样——都要
        "把用户消息落库，然后按上下文引擎构建要发给模型的消息列表"。这段公共
        逻辑抽成内部函数，避免两个接口各写一遍；开头带下划线表示它是模块内部
        使用的辅助函数，不作为接口暴露。

        ### 输入/输出示例
        ```python
        _chat_messages(None, "你好")
        # → (会话id, [上下文消息列表])
        #   第一个元素是新建（或恢复）的会话 id；第二个元素是 build_context
        #   构建好的完整上下文（含历史消息、摘要等），直接喂给模型
        ```

        ### 内部执行流程
        - step1: 调用 store.create_session——传了 session 就恢复已有会话，没传就新建
        - step2: 把用户这条消息写入会话存储
        - step3: 调用上下文引擎构建上下文（历史 + 滚动摘要等），与会话 id 一起返回
        """
        sid = store.create_session(session)
        store.add_message(sid, "user", message)
        return sid, build_context(client, cfg, store, sid)

    @app.post("/chat")
    def chat(req: ChatReq):
        """
        **一次性问答接口：接收 JSON 提问，返回完整 JSON 回答。**

        为什么需要它：这是最基本的聊天接口——客户端发一个问题，等服务算完，
        一次拿回完整答案。适合不需要流式体验的场景（脚本调用、其他系统集成）。
        整条链路复用 CLI 的同一套模块：上下文引擎组装历史 → 客户端请求模型 →
        回答落库 → 返回。

        ### 输入/输出示例
        ```python
        # 请求：POST /chat，body: {"message": "你好", "session": null}
        # 响应：
        {"session": "abc123", "answer": "你好！有什么可以帮你？"}
        # session 是本次会话 id，客户端存下来下次带上即可续聊
        ```

        ### 内部执行流程
        - step1: 调用公共准备函数 _chat_messages：建会话、存用户消息、构建上下文
        - step2: 调用 complete 让模型基于上下文生成完整回答（函数内延迟导入，保持与流式路径一致）
        - step3: 把助手回答写入会话存储（下次续聊能读到）
        - step4: 返回会话 id 和答案的 JSON
        """
        from api_client import complete
        sid, messages = _chat_messages(req.session, req.message)
        answer = complete(client, cfg, messages)
        store.add_message(sid, "assistant", answer)
        return {"session": sid, "answer": answer}

    @app.post("/chat/stream")
    def chat_stream(req: ChatReq):
        """
        **流式问答接口：边生成边推送，客户端逐段收到思考过程和回答（SSE）。**

        为什么需要它：/chat 要等模型全部生成完才返回，用户盯着空白屏幕等几十
        秒体验很差。这个接口用 SSE（Server-Sent Events，服务器推送技术）把模型
        的输出拆成一段段实时推送：先到的 reasoning 段落先显示，answer 段落
        逐字到达。返回的是 StreamingResponse——FastAPI 不缓存整个响应，而是
        把生成器里 yield 的内容一块块发给客户端。

        ### 输入/输出示例
        ```python
        # 请求：POST /chat/stream，body: {"message": "你好"}
        # 响应（text/event-stream，逐段推送）：
        # data: {"type": "reasoning", "text": "用户在打招呼，"}
        # data: {"type": "reasoning", "text": "应该礼貌回应"}
        # data: {"type": "answer", "text": "你好！"}
        # data: {"type": "answer", "text": "有什么可以帮你？"}
        # data: {"type": "done", "session": "abc123"}
        ```

        ### 内部执行流程
        - step1: 调用公共准备函数 _chat_messages 拿到会话 id 和上下文
        - step2: 定义内部生成器函数 gen（流式响应的内容来源），返回 StreamingResponse 包裹它
        - step3（gen 内）: 逐段消费 stream_events 产生的事件流，把每段打包成 SSE 格式（data: 前缀 + JSON + 空行）yield 给客户端；answer 段落同时收集起来
        - step4（gen 内）: 事件流结束后，把收集齐的完整回答写入会话存储，最后 yield 一条 done 事件（带会话 id）通知客户端流结束
        """
        sid, messages = _chat_messages(req.session, req.message)

        def gen():
            """
            **SSE 内容生成器：消费模型事件流，逐段产出 SSE 格式的数据块。**

            为什么需要它：StreamingResponse 需要一个可迭代对象作为内容来源，
            这个生成器函数就是来源。它不是接口，而是接口返回值的"内容生产线"。

            ### 输入/输出示例
            ```python
            # 被 StreamingResponse 内部逐个消费，每次 yield 一个字符串：
            # 'data: {"type": "reasoning", "text": "..."}\\n\\n'
            # 'data: {"type": "done", "session": "abc123"}\\n\\n'
            ```

            ### 内部执行流程
            - step1: 遍历 stream_events 产生的事件流（kind 为 reasoning 或 answer）
            - step2: answer 类型的段落追加到收集列表；每段都格式化成 SSE 的 data 行 yield 出去
            - step3: 流结束后把收集的 answer 段落拼成完整回答写入会话存储
            - step4: yield 一条 done 事件告知客户端流结束
            """
            answer_parts = []
            for kind, text in stream_events(cfg, messages):
                if kind == "answer":
                    answer_parts.append(text)
                yield f"data: {json.dumps({'type': kind, 'text': text}, ensure_ascii=False)}\n\n"
            store.add_message(sid, "assistant", "".join(answer_parts))
            yield f"data: {json.dumps({'type': 'done', 'session': sid}, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/ask")
    def ask(req: AskReq):
        """
        **知识库问答接口：走 RAG 流程回答问题（不带会话记忆）。**

        为什么需要它：这是"查资料"专用接口——每个问题独立处理，不记历史、
        不建会话，流程是完整的 RAG 链路：检索知识库 → 重排序 → 带着出处回答。
        实现直接复用评估器里的 predict_one（它和 CLI 的 RAG 问答是同一条生产
        路径，包含检索为空、分数过低时的兜底行为）。

        ### 输入/输出示例
        ```python
        # 请求：POST /ask，body: {"question": "项目的重试策略是什么", "k": null}
        # 响应：
        {"answer": "重试采用指数退避：1s、2s、4s，共 3 次……"}
        ```

        ### 内部执行流程
        - step1: 调用 predict_one，传配置和问题（k 为可选参数），它内部完成检索、重排序和生成
        - step2: 把答案打包成 JSON 返回（无会话概念，响应里没有 session 字段）
        """
        answer = predict_one(cfg, req.question)
        return {"answer": answer}

    @app.post("/agent")
    def agent(req: AgentReq):
        """
        **智能体任务接口：跑 LangGraph 引擎的 ReAct 智能体，返回答案和工具调用记录。**

        为什么需要它：前三个接口都是"直接问答"，这个接口是"自主完成任务"——
        模型自己决定调哪些工具、调几次（比如算数学题会自己调计算器）。底层是
        M4c 的 LangGraph 引擎，会话记忆存在 Postgres 的 checkpointer 里。客户端
        可以传 session 指定线程（续跑同一个任务线程），不传则新建。

        ### 输入/输出示例
        ```python
        # 请求：POST /agent，body: {"task": "300 乘以 0.25 是多少", "session": null}
        # 响应：
        {
          "session": "lg_20261013_110026",
          "answer": "300 × 0.25 = 75",
          "tool_trace": [
            {"tool": "calculator",
             "arguments": "{'expression': '300 * 0.25'}",
             "result": "75.0"}
          ]
        }
        ```

        ### 内部执行流程
        - step1: 确定线程号：客户端传了 session 就续用，没传就调 new_thread_id 生成新的
        - step2: 延迟导入 LangGraph 引擎的两个函数（不启用该引擎时不加载框架），调用 run_agent_langgraph 跑任务——内部会打印工具调用面板并返回统一结果
        - step3: 把线程号、最终答案、工具调用记录打包成 JSON 返回
        """
        from agent_langgraph import new_thread_id, run_agent_langgraph
        thread_id = req.session or new_thread_id()
        result = run_agent_langgraph(cfg, thread_id, req.task)
        return {"session": thread_id, "answer": result["answer"],
                "tool_trace": result["tool_trace"]}

    return app


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    **启动 HTTP 服务：用 uvicorn 把 create_app() 装配的应用跑起来。**

    为什么需要它：create_app 只负责"装配"，这个函数负责"点火"——调用 uvicorn
    这个 ASGI 服务器运行应用，开始监听端口、接收请求。uvicorn 延迟导入，只有
    真正启动服务时才需要这个依赖。

    ### 输入/输出示例
    ```python
    run()                  # 监听 127.0.0.1:8000
    run(port=9000)         # 监听 127.0.0.1:9000
    # → 无返回值；进程进入服务状态，直到手动停止（Ctrl+C）
    ```

    ### 内部执行流程
    - step1: 延迟导入 uvicorn
    - step2: 调用 create_app() 装配应用，交给 uvicorn.run 按指定主机和端口启动
    """
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port)