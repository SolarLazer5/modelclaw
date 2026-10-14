"""Storage tests: the three output formats of save_result."""

import json

from storage import save_result


def _cfg(tmp_path, fmt):
    """
    **测试辅助函数：生成一份指向临时目录、指定输出格式的测试配置。**

    为什么需要它：save_result 需要一份完整配置才能工作（输出目录、格式、
    文件名前缀等）。测试不能把结果写到项目的真实输出目录，所以用 pytest
    的 tmp_path 临时目录当输出位置，再把格式参数 fmt 嵌进配置里，一行
    代码就能为每种格式测试造出干净的配置。

    ### 输入/输出示例
    ```python
    _cfg(tmp_path, "json")
    # → {
    #     "api": {"model": "test-model"},
    #     "storage": {"output_dir": "/tmp/...（临时目录）", "default_format": "json",
    #                 "filename_prefix": "test", "save_metadata": True},
    #   }
    ```

    ### 内部执行流程
    - step1: 把临时目录路径转成字符串，作为输出目录
    - step2: 组装配置字典：模型名固定为 test-model，格式取参数 fmt，文件名前缀固定为 test，开启元数据保存
    - step3: 返回这份配置
    """
    return {
        "api": {"model": "test-model"},
        "storage": {"output_dir": str(tmp_path), "default_format": fmt,
                    "filename_prefix": "test", "save_metadata": True},
    }


def _result():
    """
    **测试辅助函数：返回一份固定的"模型回答"数据。**

    为什么需要它：save_result 的第二个参数是模型的回答结果。测试只关心
    "保存的文件格式对不对"，不关心回答内容本身，所以造一份内容固定、
    字段完整的假结果，每个测试共用它，断言时也知道该期待什么值。

    ### 输入/输出示例
    ```python
    _result()
    # → {"reasoning": "思考过程", "answer": "最终答案"}
    ```

    ### 内部执行流程
    - step1: 返回写死的字典：包含思考过程和最终答案两个字段
    """
    return {"reasoning": "思考过程", "answer": "最终答案"}


def test_save_json_with_metadata(tmp_path):
    """
    **测试 JSON 格式保存：文件内容必须是合法 JSON，且元数据字段齐全正确。**

    为什么需要它：JSON 格式带完整元数据（问题、答案、思考过程、模型名等），
    是程序化消费的结果文件。这个测试验证两点：一是写出的文件能被 json.loads
    正常解析（格式合法），二是关键字段的值和写入时一致（内容完整，没有丢
    字段或写错值）。

    ### 输入/输出示例
    ```python
    # 由 pytest 自动执行，tmp_path 是 pytest 提供的临时目录
    test_save_json_with_metadata(tmp_path)
    # → 无返回值；四条 assert（prompt/answer/reasoning/model 的值）全部成立则通过
    ```

    ### 内部执行流程
    - step1: 用 JSON 格式配置调用 save_result，拿到写出的文件路径
    - step2: 读取文件文本并用 json.loads 解析成字典
    - step3: 依次断言 prompt、answer、reasoning 三个字段与输入一致，且 model 字段记录的是配置里的模型名
    """
    path = save_result(_cfg(tmp_path, "json"), "测试问题", _result())
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["prompt"] == "测试问题"
    assert data["answer"] == "最终答案"
    assert data["reasoning"] == "思考过程"
    assert data["model"] == "test-model"


def test_save_markdown(tmp_path):
    """
    **测试 Markdown 格式保存：文件内容必须包含答案文本和格式化的元数据行。**

    为什么需要它：Markdown 格式主要给人阅读，所以验证标准不是 JSON 那种
    "字段值精确相等"，而是"该有的内容都在"——答案文本在文件里能找到，
    元数据以约定的 Markdown 列表格式（"- Prompt: 问题"）出现。这个测试
    用包含关系断言，匹配 Markdown 模板渲染的正确性。

    ### 输入/输出示例
    ```python
    # 由 pytest 自动执行，tmp_path 是 pytest 提供的临时目录
    test_save_markdown(tmp_path)
    # → 无返回值；assert 同时找到 "最终答案" 和 "- Prompt: 问题" 则通过
    ```

    ### 内部执行流程
    - step1: 用 md 格式配置调用 save_result，拿到写出的文件路径
    - step2: 读取文件全文
    - step3: 断言答案文本和 "- Prompt: 问题" 元数据行都在内容中
    """
    path = save_result(_cfg(tmp_path, "md"), "问题", _result())
    text = path.read_text(encoding="utf-8")
    assert "最终答案" in text and "- Prompt: 问题" in text


def test_save_txt(tmp_path):
    """
    **测试纯文本格式保存：文件内容必须恰好只有最终答案本身。**

    为什么需要它：txt 格式是"只要答案、不要多余内容"的极简输出，给直接
    复制答案的场景用。所以这个测试的标准最严格——不是"包含"而是"完全相等"：
    整个文件内容必须等于答案字符串，一个多余字符（包括思考过程、元数据、
    甚至结尾换行）都不允许出现。

    ### 输入/输出示例
    ```python
    # 由 pytest 自动执行，tmp_path 是 pytest 提供的临时目录
    test_save_txt(tmp_path)
    # → 无返回值；assert 文件内容 == "最终答案" 成立则通过
    ```

    ### 内部执行流程
    - step1: 用 txt 格式配置调用 save_result，拿到写出的文件路径
    - step2: 读取文件全文
    - step3: 断言文件内容与答案字符串完全相等，验证纯文本输出不含任何额外内容
    """
    path = save_result(_cfg(tmp_path, "txt"), "问题", _result())
    assert path.read_text(encoding="utf-8") == "最终答案"