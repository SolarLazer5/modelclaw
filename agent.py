"""Agent loop: native function calling, hand-rolled.

The loop mirrors the classic Thought -> Action -> Observation cycle
(LangChain's AgentExecutor does exactly this under the hood):

    working messages = context (from M1's context_engine) + tool protocol messages
    loop, at most agent.max_iterations times:
        model responds with tool_calls  -> execute each tool locally,
                                           append assistant(tool_calls) + tool result
                                           messages to working, continue
        model responds with content     -> this is the final answer, stop
    iteration cap exceeded              -> raise (runaway-loop safety net)

Persistence policy: only the user input and the final assistant answer go into
the session store (keeping the messages table schema unchanged); the tool
trace is rendered live and written to the log.
"""

import json
import logging

from rich.console import Console
from rich.panel import Panel

from api_client import ask_with_tools
from context_engine import build_context
from tools import get_tool_schemas, run_tool

logger = logging.getLogger(__name__)
console = Console()


def _render_tool_call(step: int, name: str, arguments_json: str, result: str) -> None:
    shown = result if len(result) <= 400 else result[:400] + "\n...（结果已截断展示，完整内容已喂给模型）"
    console.print(Panel(
        f"[bold cyan]{name}[/bold cyan] [dim]{arguments_json}[/dim]\n[dim]→[/dim] {shown}",
        title=f"[yellow]🔧 第 {step} 步 · 工具调用[/yellow]",
        border_style="yellow",
    ))


def run_agent(client, cfg: dict, store, session_id: str, user_input: str, system: str | None = None) -> dict:
    """Run one agent task: tool-calling loop until a final answer. Returns
    {"reasoning", "answer", "tool_trace"}; persists user + final answer."""
    store.add_message(session_id, "user", user_input)
    working = build_context(client, cfg, store, session_id, system)

    agent_cfg = cfg.get("agent", {})
    max_iterations = agent_cfg.get("max_iterations", 8)
    schemas = get_tool_schemas(agent_cfg.get("enabled_tools", []))
    tool_trace = []

    for step in range(1, max_iterations + 1):
        msg = ask_with_tools(client, cfg, working, schemas)
        reasoning = getattr(msg, "reasoning_content", None) or ""

        if msg.tool_calls:
            # The model wants to act: record its tool_calls, execute, feed back.
            working.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                result = run_tool(tc.function.name, tc.function.arguments)
                _render_tool_call(step, tc.function.name, tc.function.arguments, result)
                working.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                tool_trace.append({"tool": tc.function.name, "arguments": tc.function.arguments,
                                   "result": result})
            continue

        # No tool calls -> final answer.
        answer = msg.content or ""
        if reasoning:
            console.print("\n === Thinking ===\n")
            console.print(f"[dim]{reasoning}[/dim]")
        console.print("\n === Final Answer ===\n")
        console.print(answer)
        store.add_message(session_id, "assistant", answer)
        logger.info("Agent finished in %d step(s), %d tool call(s)", step, len(tool_trace))
        return {"reasoning": reasoning, "answer": answer, "tool_trace": tool_trace}

    raise RuntimeError(
        f"Agent 超过最大迭代次数（{max_iterations}）仍未给出最终答案——"
        "任务可能过于复杂或模型陷入循环，请拆小任务或调高 agent.max_iterations"
    )
