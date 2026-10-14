"""LangGraph engine for the agent (M4c): create_react_agent + PostgresSaver.

Counterpart to agent.py (the hand-rolled native engine). The framework version
replaces our loop with LangGraph's prebuilt ReAct graph and replaces our
SessionStore with the official Postgres checkpointer (thread_id = session id).

Selected via config agent.engine = "langgraph"; native stays the default and
the reference implementation.
"""

import logging
import os
import time

from rich.console import Console
from rich.panel import Panel

from llm_langchain import create_chat_model
from tools import get_lc_tools

logger = logging.getLogger(__name__)
console = Console()


def _checkpoint_uri(cfg: dict) -> str:
    """
    **根据配置和环境变量拼出 PostgreSQL 连接字符串，供 LangGraph 存档对话状态用。**

    为什么需要它：M4c 用 LangGraph 官方的 PostgresSaver（Postgres 存档器）替代手写
    的 SessionStore，对话的每一步状态都存进 Postgres。PostgresSaver 需要一个标准
    格式的连接字符串才能工作。这个函数把配置里的主机、端口、库名和 .env 里的密码
    拼成这个字符串。密码不写在配置文件里，而是单独从环境变量 MODELCLAW_PG_PASSWORD
    读取——密码和配置分离，避免配置文件被分享时泄露密码。

    ### 输入/输出示例
    ```python
    _checkpoint_uri({"memory": {"postgres": {"host": "127.0.0.1", "port": 5432,
                                             "user": "postgres", "database": "modelclaw"}}})
    # → "postgresql://postgres:密码@127.0.0.1:5432/modelclaw"
    #   配置里缺的项用默认值补齐（host 默认 127.0.0.1，port 默认 5432 等）

    _checkpoint_uri({})
    # → 若环境变量 MODELCLAW_PG_PASSWORD 没设置，抛 RuntimeError：
    #   "未找到 MODELCLAW_PG_PASSWORD，请在 .env 文件中配置 PostgreSQL 密码"
    ```

    ### 内部执行流程
    - step1: 从配置里取出 memory.postgres 小节（缺省为空字典），准备读取连接参数
    - step2: 从环境变量 MODELCLAW_PG_PASSWORD 读密码；密码为空就直接抛异常，不拼一个连不上的字符串
    - step3: 按 postgresql://用户名:密码@主机:端口/库名 的格式拼连接字符串，每个部分取配置值、缺省用默认值
    - step4: 返回拼好的连接字符串
    """
    pg = cfg.get("memory", {}).get("postgres", {})
    password = os.environ.get("MODELCLAW_PG_PASSWORD", "")
    if not password:
        raise RuntimeError("未找到 MODELCLAW_PG_PASSWORD，请在 .env 文件中配置 PostgreSQL 密码")
    return (
        f"postgresql://{pg.get('user', 'postgres')}:{password}"
        f"@{pg.get('host', '127.0.0.1')}:{pg.get('port', 5432)}"
        f"/{pg.get('database', 'modelclaw')}"
    )


def new_thread_id() -> str:
    """
    **生成一个新的会话编号（thread_id），格式为时间戳字符串。**

    为什么需要它：LangGraph 用 thread_id 区分不同会话的存档（对应原生引擎里的
    会话 id）。每次开新对话需要一个独一无二的编号，这里直接用当前时间格式化成
    字符串——前缀 lg 标明来自 LangGraph 引擎，后面跟年月日_时分秒，肉眼就能看出
    会话是什么时候创建的。和原生引擎一样，同一秒内开两个会话可能撞号，但
    CLI 场景人操作来不及，所以接受这个取舍。

    ### 输入/输出示例
    ```python
    new_thread_id()
    # → "lg_20261012_143052"   （2026 年 10 月 12 日 14 时 30 分 52 秒创建的会话）
    ```

    ### 内部执行流程
    - step1: 取当前系统时间，按 lg_年月日_时分秒 的格式转成字符串
    - step2: 返回这个字符串作为新会话的 thread_id
    """
    return time.strftime("lg_%Y%m%d_%H%M%S")


def run_agent_langgraph(cfg: dict, thread_id: str, task: str) -> dict:
    """
    **用 LangGraph 预置的 ReAct 智能体跑一个任务，返回思考过程、最终答案和工具调用记录。**

    为什么需要它：这是 M4c 的核心函数，与手写的原生引擎（agent.py）平行。原生
    引擎自己维护"模型思考→调工具→看结果→再思考"的循环，这个版本把这个循环交给
    LangGraph 官方预置的 ReAct 图（create_react_agent），对话状态的持久化也交给
    官方 PostgresSaver 而不是手写 SessionStore。函数负责三件事：装配智能体
    （工具+模型+存档器）、运行任务、把返回的消息列表整理成项目统一的结果格式
    （含逐步的工具调用面板展示）。langgraph 相关导入放在函数内部，是延迟导入——
    没选 langgraph 引擎时，启动 CLI 不用加载这套框架。

    ### 输入/输出示例
    ```python
    run_agent_langgraph(cfg, "lg_20261012_143052", "300 乘以 0.25 是多少")
    # → {
    #     "reasoning": "用户要做乘法，用计算器",      # 模型思考过程（可能为空串）
    #     "answer": "300 * 0.25 = 75.0",            # 最终答案
    #     "tool_trace": [                            # 每次工具调用的记录
    #       {"tool": "calculator", "arguments": "{'expression': '300 * 0.25'}",
    #        "result": "75.0"},
    #     ],
    #   }
    # 运行过程中还会在终端逐步打印黄色边框的工具调用面板、思考过程和最终答案
    ```

    ### 内部执行流程
    - step1: 延迟导入 PostgresSaver 和 create_react_agent，并从配置读出智能体小节
    - step2: 按配置的 enabled_tools 列表拿到工具的 LangChain 版本，再创建聊天模型，完成智能体原料准备
    - step3: 用连接字符串打开 PostgresSaver（with 语句保证用完自动断开），调用 setup() 建好存档所需的表
    - step4: 用模型、工具、存档器创建 ReAct 智能体，以 thread_id 为会话标识传入用户任务并运行，LangGraph 接管整个思考-调工具循环
    - step5: 运行结束后从最终消息列表中提取工具调用轨迹：先收集所有 type 为 tool 的消息（按 tool_call_id 存结果），再按出现顺序给每个 tool_calls 编号、截断超长结果到 400 字符，逐个打印面板并记入 tool_trace
    - step6: 取最后一条消息的内容作为答案、additional_kwargs 里的 reasoning_content 作为思考过程；有思考过程就先打印 === Thinking ===，再打印 === Final Answer ===
    - step7: 记一条运行日志（会话号、工具调用次数），返回统一格式的结果字典
    """
    """Run one agent task via LangGraph; returns {"reasoning", "answer", "tool_trace"}."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.prebuilt import create_react_agent

    agent_cfg = cfg.get("agent", {})
    tools = get_lc_tools(agent_cfg.get("enabled_tools", []))
    model = create_chat_model(cfg)

    with PostgresSaver.from_conn_string(_checkpoint_uri(cfg)) as checkpointer:
        checkpointer.setup()
        agent = create_react_agent(model, tools, checkpointer=checkpointer)
        result = agent.invoke(
            {"messages": [("user", task)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    # Render the tool trace post-hoc from the final message list
    tool_trace = []
    messages = result["messages"]
    tool_results = {}
    for m in messages:
        if getattr(m, "type", None) == "tool":
            tool_results[m.tool_call_id] = m.content
    step = 0
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            step += 1
            res = tool_results.get(tc["id"], "")
            shown = res if len(res) <= 400 else res[:400] + "\n...（截断展示）"
            console.print(Panel(
                f"[bold cyan]{tc['name']}[/bold cyan] [dim]{tc['args']}[/dim]\n[dim]→[/dim] {shown}",
                title=f"[yellow]🔧 第 {step} 步 · 工具调用（LangGraph）[/yellow]",
                border_style="yellow",
            ))
            tool_trace.append({"tool": tc["name"], "arguments": str(tc["args"]), "result": res})
##messages结构参考：
#     [
#       {"type": "human", "content": "300 乘以 0.25 是多少"},
#       {"type": "ai", "content": "",                  # 模型这轮没直接回答
#       "tool_calls": [                                # 但发起了 1 次工具请求
#           {"id": "call_1", "name": "calculator", "args": {"expression": "300 * 0.25"}}
#       ]},
#       {"type": "tool", "tool_call_id": "call_1",     # 工具的执行结果
#       "content": "75.0"},
#       {"type": "ai", "content": "300 * 0.25 = 75.0", # 模型拿到结果后的最终回答
#       "tool_calls": []},
#     ]

    answer = messages[-1].content if messages else ""
    reasoning = messages[-1].additional_kwargs.get("reasoning_content", "") if messages else ""
    if reasoning:
        console.print("\n === Thinking ===\n")
        console.print(f"[dim]{reasoning}[/dim]")
    console.print("\n === Final Answer ===\n")
    console.print(answer)
    logger.info("LangGraph agent finished: thread=%s, %d tool call(s)", thread_id, len(tool_trace))
    return {"reasoning": reasoning, "answer": answer, "tool_trace": tool_trace}