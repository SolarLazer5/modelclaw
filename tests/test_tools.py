"""Tool safety tests: the three security boundaries of M2."""

from tools import run_tool


class TestCalculator:
    """计算器工具的安全边界测试。"""

    def test_normal_math(self):
        """
        **测试计算器工具的正常功能：常规四则运算和函数调用要算出正确结果。**

        为什么需要它：安全测试的前提是工具本身功能正常——如果连 300 * 0.25 都
        算不对，后面那些"拦截攻击"的测试就算全过也没有意义。这个测试先验证
        计算器的正常使用路径（乘法、sqrt 函数）返回正确数值，作为安全测试的
        基线。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_normal_math()
        # → 无返回值；两条 assert 全过则测试通过：
        #   300 * 0.25 的结果是 "75.0"，sqrt(16) 的结果是 "4.0"
        ```

        ### 内部执行流程
        - step1: 调用 run_tool 计算 300 * 0.25，断言返回字符串 "75.0"
        - step2: 调用 run_tool 计算 sqrt(16)，断言返回字符串 "4.0"
        """
        assert run_tool("calculator", '{"expression": "300 * 0.25"}') == "75.0"
        assert run_tool("calculator", '{"expression": "sqrt(16)"}') == "4.0"

    def test_injection_blocked(self):
        """
        **测试注入拦截：在计算表达式里调用 __import__ 导入系统模块，必须被拒绝。**

        为什么需要它：计算器工具的实现原理是把表达式字符串交给 Python 的 eval
        计算。eval 本身能执行任意 Python 代码，如果表达式里能调用 __import__，
        攻击者就能导入 os 模块执行系统命令（比如删文件）。安全防线是：运行
        表达式时把 __builtins__（内置函数命名空间）清空，让 __import__ 解析
        不到。这个测试验证注入这种恶意表达式时，工具不会执行它，而是返回
        包含"出错"字样的错误信息。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_injection_blocked()
        # → 无返回值；assert 通过则证明：返回结果里同时含 "出错" 和 "__import__"，
        #   即工具报错了，且报错信息里带出了出问题的函数名，方便排查
        ```

        ### 内部执行流程
        - step1: 构造恶意表达式 __import__('os')，通过 run_tool 传给计算器
        - step2: 保存返回结果（预期是错误提示字符串，而不是任何计算结果或系统行为）
        - step3: 断言结果包含"出错"（确实被拦截了）且包含"__import__"（报错信息指出了罪魁）
        """
        # __builtins__ is empty: __import__ must not resolve
        result = run_tool("calculator", '{"expression": "__import__(\'os\')"}')
        assert "出错" in result and "__import__" in result

    def test_bad_expression(self):
        """
        **测试语法错误处理：不完整的表达式（1 +）要返回报错，而不是崩溃。**

        为什么需要它：用户或模型可能给出语法不完整的表达式。工具的 eval 遇到
        语法错误会抛异常，如果实现没有捕获，整个程序就会崩掉。这个测试验证
        工具把异常转换成用户可读的"出错"提示。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_bad_expression()
        # → 无返回值；assert "出错" in 结果 成立则测试通过
        ```

        ### 内部执行流程
        - step1: 构造语法不完整的表达式 "1 +" 传给计算器
        - step2: 断言返回结果包含"出错"，验证语法异常被正常捕获并转为错误提示
        """
        assert "出错" in run_tool("calculator", '{"expression": "1 +"}')


class TestReadLocalFile:
    """文件读取工具的沙箱边界测试。"""

    def test_sandbox_traversal_blocked(self):
        """
        **测试目录穿越拦截：用 ../ 跳出沙箱目录读取 .env，必须被拒绝。**

        为什么需要它：文件读取工具只允许访问项目目录内的文件（沙箱），否则
        模型一旦失控就能读到任意敏感文件。最常见的逃逸手段是路径里带 ../
        逐级向上跳到上级目录。这个测试尝试用 ../.env 读取项目外的环境变量
        文件（里面通常有数据库密码、API key），验证工具识别出逃逸行为并拒绝。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_sandbox_traversal_blocked()
        # → 无返回值；assert "拒绝读取" in 结果 成立则测试通过
        ```

        ### 内部执行流程
        - step1: 构造路径 "../.env"（上级目录的环境变量文件）传给文件读取工具
        - step2: 保存返回结果
        - step3: 断言结果包含"拒绝读取"，验证目录穿越被沙箱检查拦截
        """
        result = run_tool("read_local_file", '{"path": "../.env"}')
        assert "拒绝读取" in result

    def test_sandbox_absolute_path_blocked(self):
        """
        **测试绝对路径拦截：直接给系统路径（C:/Windows/system.ini），必须被拒绝。**

        为什么需要它：拦截了 ../ 还不够——攻击者可以直接写绝对路径绕过相对
        路径检查，读取操作系统关键文件。这个测试用 Windows 系统配置文件做
        目标，验证沙箱同样拒绝项目目录之外的一切绝对路径，不管里面有没有 ../。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_sandbox_absolute_path_blocked()
        # → 无返回值；assert "拒绝读取" in 结果 成立则测试通过
        ```

        ### 内部执行流程
        - step1: 构造系统绝对路径 "C:/Windows/system.ini" 传给文件读取工具
        - step2: 保存返回结果
        - step3: 断言结果包含"拒绝读取"，验证绝对路径逃逸被拦截
        """
        result = run_tool("read_local_file", '{"path": "C:/Windows/system.ini"}')
        assert "拒绝读取" in result

    def test_legit_file(self):
        """
        **测试合法读取：沙箱内的正常文件要能正常读到内容。**

        为什么需要它：和计算器测试同理，安全拦截不能以牺牲正常功能为代价。
        如果沙箱检查写得过严，把合法路径也拦了，工具就没法用了。这个测试
        读取项目内的 config.json，验证正常路径不受安全策略影响。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_legit_file()
        # → 无返回值；assert 文件内容里含 '"api"' 则测试通过
        ```

        ### 内部执行流程
        - step1: 传入项目内合法路径 config.json
        - step2: 断言返回的文件内容包含 '"api"'（配置文件里应有的字段名），验证合法读取正常
        """
        result = run_tool("read_local_file", '{"path": "config.json"}')
        assert '"api"' in result

    def test_missing_file(self):
        """
        **测试文件不存在的情况：读取一个不存在的文件，要返回提示而不是崩溃。**

        为什么需要它：模型可能编造一个不存在的路径（幻觉）。文件系统对不存在的
        文件会抛异常，工具需要捕获它并转成"文件不存在"的友好提示，让模型读到
        后能自己纠正，而不是让异常直接炸掉整个对话流程。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_missing_file()
        # → 无返回值；assert "不存在" in 结果 成立则测试通过
        ```

        ### 内部执行流程
        - step1: 传入一个不存在的文件路径 nope.md
        - step2: 断言返回结果包含"不存在"，验证文件缺失异常被正常捕获并转为提示
        """
        assert "不存在" in run_tool("read_local_file", '{"path": "nope.md"}')


class TestRobustness:
    """工具调度层的健壮性测试。"""

    def test_unknown_tool(self):
        """
        **测试未知工具处理：调用一个不存在的工具名，要返回提示而不是崩溃。**

        为什么需要它：模型完全可能幻觉出一个不存在的工具名（比如把 "calculator"
        拼成别的）。run_tool 是工具调度的总入口，如果连这种输入都没兜住，一次
        幻觉就能让整个程序崩掉。这个测试验证未知工具名被识别并返回"未知工具"
        提示，让模型有机会改口换正确的工具。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_unknown_tool()
        # → 无返回值；assert "未知工具" in 结果 成立则测试通过
        ```

        ### 内部执行流程
        - step1: 传入不存在的工具名 "hack" 和空参数 {}
        - step2: 断言返回结果包含"未知工具"，验证调度层对未知工具名的兜底处理
        """
        assert "未知工具" in run_tool("hack", "{}")

    def test_bad_json_arguments(self):
        """
        **测试参数解析失败处理：传入非法 JSON 作为参数，要返回提示而不是崩溃。**

        为什么需要它：工具参数以 JSON 字符串传递。模型可能输出格式不完整的
        JSON（比如 "not-json"），或者压根不是 JSON。解析失败会抛异常，run_tool
        需要在最外层兜住，返回"解析失败"提示，把异常挡在工具系统内部。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，无参数
        test_bad_json_arguments()
        # → 无返回值；assert "解析失败" in 结果 成立则测试通过
        ```

        ### 内部执行流程
        - step1: 给计算器传入非法参数字符串 "not-json"（不是 JSON）
        - step2: 断言返回结果包含"解析失败"，验证参数解析异常被最外层兜底
        """
        assert "解析失败" in run_tool("calculator", "not-json")