"""Evaluation pipeline (M4a): auto-generate QA pairs from the knowledge base,
batch-predict through the RAG pipeline, and grade with an LLM judge.

This module is the LangChain/LCEL home turf — offline batch work with no
streaming or session concerns, so we use the real framework here:

    gen chain:   PromptTemplate | ChatOpenAI | StrOutputParser   (test-set gen)
    predict:     rag.retrieve + api_client.complete             (non-streaming)
    judge chain: PromptTemplate | with_structured_output(Grade)  (LLM-as-a-Judge)
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from api_client import complete, create_client
from llm_langchain import create_chat_model
from rag import _RAG_SYSTEM_PROMPT, create_rag_store, retrieve

logger = logging.getLogger(__name__)

MIN_QA_PAIRS = 5

_QA_GEN_TEMPLATE = """你是一位出题老师。根据下面的文档片段，生成一个基于该片段内容的问题和对应的标准答案。
问题必须能仅根据片段内容回答，答案要准确简洁。

格式（严格遵守，不要输出其他内容）：
QUESTION: <问题>
ANSWER: <答案>

文档片段：
{doc}

请使用中文输出"""

_EVAL_TEMPLATE = """你是一个评判作业答案的评审老师。
给你一个考试题目、标准答案和学生提交的答案，请判断学生提交的答案是否与标准答案相关且一致。
学生答案可能更详细，只要核心信息一致就判 CORRECT；如果学生说「知识库中没有相关内容」而标准答案确实存在，判 INCORRECT。

QUESTION: {query}
TRUE ANSWER: {answer}
STUDENT ANSWER: {result}
"""


class Grade(BaseModel):
    """Structured judge output."""

    grade: Literal["CORRECT", "INCORRECT"] = Field(description="判定结果")
    reason: str = Field(description="一句话判定理由")


def generate_qa_pairs(cfg: dict, n: int = MIN_QA_PAIRS) -> list[dict]:
    """
    **从知识库随机抽取若干块，用大模型自动生成"问题+标准答案"的测试题。**

    免手工出题：LLM 看着块内容出一道"仅根据该块就能回答"的题。
    生成走一条 LangChain LCEL 链（提示词模板 → 模型 → 文本解析），
    链级重试 3 次——评估是离线批量任务，网络抖动不该让它整体失败。
    输出按固定格式（QUESTION:/ANSWER:）解析，解析失败的条目直接丢弃。

    ### 输入/输出示例

    ```python
    generate_qa_pairs(cfg, n=5)
    # 从库里随机抽 5 块，每块生成一道题，例如：
    # → [
    #   {"query": "retry 组有哪些字段？", "answer": "max_attempts、backoff..."},
    #   ... 最多 n 条（个别生成格式不合规的会被丢弃，可能略少于 n）
    # ]

    # 知识库是空的
    generate_qa_pairs(cfg)
    # → 抛出 RuntimeError: 知识库为空，先 modelclaw ingest <路径> 灌入文档
    ```

    ### 内部执行流程
    - step1: 连接向量库，取出全部块；空库直接报错并提示先入库；
    - step2: 随机抽取 min(n, 总块数) 块作为出题素材（随机保证每次
            评估覆盖不同文档区域）；
    - step3: 组装生成链（模板 + 模型 + 文本解析器，包 3 次重试），
            批量为每块生成"问题+答案"文本；
    - step4: 用正则从每段生成文本里提取 QUESTION/ANSWER 两部分，
            解析成功才收录，最后返回题目列表。
    """
    store = create_rag_store(cfg)
    chunks = store.all_chunks()
    if not chunks:
        raise RuntimeError("知识库为空，先 modelclaw ingest <路径> 灌入文档")
    sample = random.sample(chunks, min(n, len(chunks)))
    ##知识点
    ##1，random.sample
    ##从 chunks 中随机抽取 min(n, len(chunks)) 个元素，无放回抽样，也就是同一个位置的元素不会被重复抽到。
    ##返回一个新的列表。

    llm = create_chat_model(cfg)
    gen_chain = (
        PromptTemplate.from_template(_QA_GEN_TEMPLATE) | llm | StrOutputParser()
    ).with_retry(stop_after_attempt=3)  # chain-level retry: network blips shouldn't kill eval
    outputs = gen_chain.batch([{"doc": c["content"]} for c in sample])
    ##知识点
    ## 1,StrOutputParser 在哪里发挥作用？
    ## 如果 PromptTemplate.from_template(_QA_GEN_TEMPLATE) | llm | StrOutputParser()改为：
    ## PromptTemplate.from_template(_QA_GEN_TEMPLATE) | llm ，那么最终这条链吐出得结果会是一个AIMessage对象
    ## 讲这个AIMessage对象输入给StrOutputParser()，则最终出来得是一个字符串，后面正则表达需要得是字符串而不是
    ## AIMessage对象

    pairs = []
    for out in outputs:
        m = re.search(r"QUESTION[:：]\s*(.+?)\s*ANSWER[:：]\s*(.+)", out, re.S)
        if m:
            pairs.append({"query": m.group(1).strip(), "answer": m.group(2).strip()})
    logger.info("Generated %d QA pairs (from %d chunks)", len(pairs), len(sample))
    ##知识点:
    #1,group函数和re.S
    #re.search 返回的 Match 对象，可以理解成"把刚才那次匹配的结果打包"：
    # 模式里有几个捕获组（几对圆括号 (...)），结果就分几格存；
    # m.group(1) → 第 1 对圆括号抓到的文本（问题）；
    # m.group(2) → 第 2 对圆括号抓到的文本（答案）；
    # 如果 re.search 没匹配到，m 就是 None
    # re.search(模式, out, re.S) 在文本里找第一处"QUESTION: … ANSWER: …"结构，
    # [:：]兼容两种冒号、(.+?)非贪婪抓问题、(.) + re.S跨行吞答案
    return pairs


def predict_one(cfg: dict, question: str) -> str:
    """
    **让 RAG 系统回答一道题（非流式），返回它的答案文本。**

    复用生产路径的全部组件：同一个 retrieve（混合召回+重排）、
    同一份系统提示词、同一个拒答阈值——保证"评估的对象"和
    "用户实际用的系统"是同一个，评估结果才可信。
    检索为空或相关度低于阈值时，返回和产品一致的拒答话术。

    ### 输入/输出示例

    ```python
    predict_one(cfg, "retry 组有哪些字段？")
    # → "retry 组有 4 个字段：max_attempts..."

    # 检索不到相关内容（低于阈值）
    predict_one(cfg, "量子引力的最新进展？")
    # → "知识库中没有相关内容"
    ```

    ### 内部执行流程
    - step1: 调用 retrieve 检索相关块；
    - step2: 检索为空或最高分低于 MIN_RERANK_SCORE → 直接返回拒答话术
            （与产品的拒答行为保持一致，拒答会被裁判判为 INCORRECT，
            这正是评估想捕捉的失败）；
    - step3: 把关过的块拼成上下文，连同系统提示词一起发给大模型；
    - step4: 非流式地拿完整答案并返回。
    """
    from rag import MIN_RERANK_SCORE
    hits = retrieve(cfg, question)
    if not hits or hits[0]["score"] < MIN_RERANK_SCORE:
        return "知识库中没有相关内容"
    context = "\n\n".join(f"[{h['source']}]\n{h['content']}" for h in hits)
    client = create_client(cfg)
    return complete(client, cfg, [
        {"role": "system", "content": _RAG_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ])


def judge_pairs(cfg: dict, predictions: list[dict]) -> list[Grade]:
    """
    **让 LLM 裁判批量给预测结果打分，返回结构化的判定（对/错+理由）。**

    裁判链（提示词模板 → 带结构化输出的模型）把"题目、标准答案、系统答案"
    三件套交给大模型，按"核心信息是否一致"判 CORRECT / INCORRECT。
    结构化输出走 function calling 通道——因为当前服务商不支持
    json_schema 严格模式，但支持工具调用（M2 已验证），这是能力降级后的
    务实绕行。链级重试 3 次，容忍批量打分中的偶发网络故障。

    ### 输入/输出示例

    ```python
    judge_pairs(cfg, [
        {"query": "retry 组有哪些字段？", "answer": "max_attempts...",
         "result": "retry 组有 4 个字段：max_attempts..."},
        ...
    ])
    # → [Grade(grade="CORRECT", reason="核心字段信息一致"),
    #    Grade(grade="INCORRECT", reason="答非所问，未提及字段清单"), ...]
    ```

    ### 内部执行流程
    - step1: 创建模型，组装裁判链——模板的三个占位符对应三件套，
            模型套上结构化输出（method="function_calling"），
            输出强制符合 Grade 结构（grade + reason 两字段）；
    - step2: 整条链包 3 次重试；
    - step3: 批量对所有预测打分，返回 Grade 对象列表。
    """
    llm = create_chat_model(cfg)
    # ModelScope 不支持 response_format=json_schema（严格结构化输出），
    # 但支持 function calling（M2 已验证）——判分结构化走工具调用通道。
    judge_chain = (
        PromptTemplate.from_template(_EVAL_TEMPLATE)
        | llm.with_structured_output(Grade, method="function_calling")
    ).with_retry(stop_after_attempt=3)
    return judge_chain.batch(predictions)


def run_eval(cfg: dict, n: int = MIN_QA_PAIRS) -> dict:
    """
    **评估主流程：自动出题 → 逐题预测 → 裁判打分 → 汇总成报告，返回报告字典。**

    一条命令完成"出题、考试、阅卷、出成绩单"全过程：
    报告含时间、模型名、总题数、答对数、准确率，以及每道题的完整档案
    （题目/标准答案/系统答案/判定/判定理由）——逐题明细是排查
    "哪类题在失败"的第一手材料。

    ### 输入/输出示例

    ```python
    run_eval(cfg, n=10)
    # → {
    #     "timestamp": "2026-10-10 21:30:00",
    #     "model": "deepseek-r1",
    #     "total": 10, "correct": 7, "accuracy": 0.7,
    #     "cases": [
    #       {"query": "...", "answer": "...", "result": "...",
    #        "grade": "CORRECT", "reason": "..."},
    #       ...
    #     ],
    #   }
    ```

    ### 内部执行流程
    - step1: 自动生成 n 道测试题（generate_qa_pairs）；
    - step2: 逐题走 RAG 管线作答（predict_one），把题目、标准答案、
            系统答案组装成一条预测记录；
    - step3: 全部预测交给裁判链批量打分（judge_pairs）；
    - step4: 把判定结果合入各题记录，统计答对数与准确率，
            连同时间、模型名一起组装成报告字典返回。
    """
    pairs = generate_qa_pairs(cfg, n)
    predictions = []
    for p in pairs:
        predictions.append({"query": p["query"], "answer": p["answer"],
                            "result": predict_one(cfg, p["query"])})
    grades = judge_pairs(cfg, predictions)

    cases = [dict(**pred, grade=g.grade, reason=g.reason)
             for pred, g in zip(predictions, grades)]
    correct = sum(1 for c in cases if c["grade"] == "CORRECT")
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": cfg["api"]["model"],
        "total": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases) if cases else 0.0,
        "cases": cases,
    }


def save_report(cfg: dict, report: dict) -> Path:
    """
    **把评估报告写入 JSON 文件，返回文件路径。**

    文件名带时间戳（eval_年月日_时分秒.json），多次评估自动归档不覆盖；
    目录取自配置的 storage.output_dir（默认 output/），不存在则自动创建。
    中文内容不转义（ensure_ascii=False），保证报告可直接阅读。

    ### 输入/输出示例

    ```python
    save_report(cfg, report)
    # → PosixPath("output/eval_20261010_213000.json")
    # 文件内容为格式化缩进的 JSON，含全部逐题明细
    ```

    ### 内部执行流程
    - step1: 解析输出目录并确保存在；
    - step2: 生成带时间戳的文件路径；
    - step3: 把报告序列化为 JSON 写入文件（保留中文原文、缩进排版）；
    - step4: 返回文件路径。
    """
    output_dir = Path(cfg.get("storage", {}).get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path