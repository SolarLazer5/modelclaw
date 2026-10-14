"""LLM regression test: accuracy guard for the RAG pipeline.

Skipped by default (costs real API quota). Run explicitly with:

    MODELCLAW_RUN_LLM_TESTS=1 python -m pytest tests/test_eval_regression.py

Fails when accuracy drops below MODELCLAW_EVAL_MIN_ACCURACY (default 0.6).
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MODELCLAW_RUN_LLM_TESTS") != "1",
    reason="LLM 回归测试默认跳过（消耗 API 额度），用 MODELCLAW_RUN_LLM_TESTS=1 启用",
)


def test_rag_eval_accuracy_guard():
    """
    **LLM 回归测试：真实调用大模型跑一轮 RAG 评估，准确率跌破阈值就报警失败。**

    为什么需要它：普通的单元测试只验证单个函数的逻辑，但无法发现"整条 RAG 链路
    的准确率悄悄下降了"这类问题——比如换了 embedding 模型、改了检索参数之后，
    每个函数单独看都没错，整体回答质量却变差了。这个测试调用 evaluator 里的
    run_eval，用真实 LLM 跑 n=5 个问答并自动评分，把准确率当作守护指标：低于
    阈值（默认 0.6）就断言失败，并在报错信息里列出所有答错的用例，方便定位问题。

    由于要消耗真实 API 额度，整个文件默认跳过，只有设置环境变量
    MODELCLAW_RUN_LLM_TESTS=1 时才会运行（由文件顶部的 pytestmark 控制）。

    ### 输入/输出示例
    ```python
    # 由 pytest 自动执行（需先设置 MODELCLAW_RUN_LLM_TESTS=1，否则跳过）
    test_rag_eval_accuracy_guard()
    # → 通过：report["accuracy"] >= 0.6（或环境变量 MODELCLAW_EVAL_MIN_ACCURACY 指定的值）
    # → 失败：抛 AssertionError，报错里带具体百分比和答错的 query 列表，例如：
    #   "RAG 准确率 40% 跌破阈值 60%，失败用例: ['query1', 'query2']"
    ```

    ### 内部执行流程
    - step1: 加载 .env 环境变量，然后导入 evaluator.run_eval 和 main.load_config（放在函数内部导入，避免文件被收集时就触发依赖加载）
    - step2: 读取项目配置 cfg
    - step3: 调用 run_eval(cfg, n=5) 真实跑一轮评估：生成问答对 → 用生产路径预测 → LLM 打分，返回含 accuracy 和逐用例结果的报告
    - step4: 从环境变量 MODELCLAW_EVAL_MIN_ACCURACY 读准确率下限，缺省为 0.6
    - step5: 断言报告准确率不低于阈值；低于时断言失败，报错信息中列出所有评分不是 CORRECT 的用例 query，便于排查是哪类问题变难了
    """
    from dotenv import load_dotenv
    load_dotenv()
    from evaluator import run_eval
    from main import load_config

    cfg = load_config()
    report = run_eval(cfg, n=5)
    min_accuracy = float(os.environ.get("MODELCLAW_EVAL_MIN_ACCURACY", "0.6"))
    assert report["accuracy"] >= min_accuracy, (
        f"RAG 准确率 {report['accuracy']:.0%} 跌破阈值 {min_accuracy:.0%}，"
        f"失败用例: {[c['query'] for c in report['cases'] if c['grade'] != 'CORRECT']}"
    )