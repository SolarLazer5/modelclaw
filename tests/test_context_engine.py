"""Context engineering tests: token counting, turn-aware splitting, rolling
summarization (LLM call mocked), and the latest-message safety net.
"""

import context_engine
from context_engine import _split_head_tail, build_context, count_tokens


def _msgs(pairs):
    """
    **测试辅助函数：快速生成指定数量的"一问一答"假对话消息列表。**

    测试上下文引擎（context_engine）需要很多份对话记录当输入。手写每一条消息
    太麻烦，这个函数按固定的编号规则批量造出假消息：每个"问答对"生成一条用户
    消息和一条助手消息，id 自动递增。这样每个测试用一行代码就能拿到结构标准
    的对话数据。

    ### 输入/输出示例
    ```python
    _msgs(2)
    # → [
    #     {"id": 1, "role": "user",      "content": "问题0"},
    #     {"id": 2, "role": "assistant", "content": "回答0"},
    #     {"id": 3, "role": "user",      "content": "问题1"},
    #     {"id": 4, "role": "assistant", "content": "回答1"},
    #   ]
    # 传入 pairs=2 表示造 2 个问答对，共 4 条消息
    ```

    ### 内部执行流程
    - step1: 准备一个空列表 out，用于存放生成的消息
    - step2: 循环 pairs 次，每次处理一个问答对（下标 i 从 0 开始）
    - step3: 向 out 追加一条用户消息：id 为 i*2+1，内容为 "问题{i}"
    - step4: 向 out 追加一条助手消息：id 为 i*2+2，内容为 "回答{i}"
    - step5: 循环结束后返回整个消息列表
    """
    out = []
    for i in range(pairs):
        out.append({"id": i * 2 + 1, "role": "user", "content": f"问题{i}"})
        out.append({"id": i * 2 + 2, "role": "assistant", "content": f"回答{i}"})
    return out


class FakeStore:
    """In-memory SessionStore stand-in for context_engine tests."""

    def __init__(self, messages, summary=""):
        """
        **FakeStore 的初始化方法：把传入的对话数据和摘要存进内存属性。**

        为什么需要 FakeStore：build_context 需要一个"会话数据库"来读取对话、
        保存摘要、删除旧消息。测试不能动真实数据库，于是写一个内存版的假数据库，
        方法和真实数据库一样，但数据只存在内存变量里，测试随便折腾也不受影响。

        ### 输入/输出示例
        ```python
        store = FakeStore(_msgs(4), summary="旧摘要")
        # → store._messages 里是 4 条假消息，store.summary 是 "旧摘要"，
        #   store.deleted_before 是 None（还没有删除过任何消息）
        ```

        ### 内部执行流程
        - step1: 把参数 messages 存到 self._messages，作为这个假数据库里的全部对话
        - step2: 把参数 summary 存到 self.summary，作为当前已有的对话摘要（默认空字符串）
        - step3: 把 self.deleted_before 初始化为 None，记录"尚未执行过删除旧消息操作"
        """
        self._messages = messages
        self.summary = summary
        self.deleted_before = None

    def get_messages(self, session_id):
        """
        **模仿真实数据库的取消息方法：返回内存里的全部对话。**

        真实数据库的 get_messages 要根据 session_id 去 SQLite 里查消息；
        假数据库不需要真的查，直接把初始化时存好的消息列表整个返回就行。
        session_id 参数留着不接，是为了让方法签名和真实数据库保持一致。

        ### 输入/输出示例
        ```python
        store.get_messages("任意会话id")
        # → 返回初始化时传入的那个消息列表（self._messages）
        ```

        ### 内部执行流程
        - step1: 直接返回 self._messages，不做任何过滤或查询
        """
        return self._messages

    def get_summary(self, session_id):
        """
        **模仿真实数据库的取摘要方法：返回内存里存的对话摘要。**

        和 get_messages 同理：真实方法要查库，假方法直接返回内存里的
        self.summary，签名保持一致即可。

        ### 输入/输出示例
        ```python
        store.get_summary("任意会话id")
        # → 返回初始化时传入的 summary（没传则返回空字符串 ""）
        ```

        ### 内部执行流程
        - step1: 直接返回 self.summary
        """
        return self.summary

    def set_summary(self, session_id, summary):
        """
        **模仿真实数据库的存摘要方法：把新摘要写进内存。**

        build_context 在做完滚动摘要后，会调用 set_summary 把摘要存起来。
        假数据库就简单地用新值覆盖 self.summary，同时这个值会被测试用
        assert 检查，验证"摘要确实被保存了"。

        ### 输入/输出示例
        ```python
        store.set_summary("t", "滚动摘要内容")
        # → 无返回值；副作用是 store.summary 从旧值变成 "滚动摘要内容"
        ```

        ### 内部执行流程
        - step1: 把参数 summary 赋值给 self.summary，完成"保存"
        """
        self.summary = summary

    def delete_messages_before(self, session_id, max_id):
        """
        **模仿真实数据库的删旧消息方法：从内存列表里删掉 id 不超过 max_id 的消息。**

        build_context 在做完摘要后，会调用这个方法把已经被摘要覆盖掉的旧对话
        从库里删掉。假数据库用列表推导式过滤掉 id <= max_id 的消息，同时把
        max_id 记录到 self.deleted_before，方便测试断言"删除确实发生了，且删的
        分界线正确"。

        ### 输入/输出示例
        ```python
        store.delete_messages_before("t", 5)
        # → 无返回值；副作用有两个：
        #   store.deleted_before == 5
        #   store._messages 里只剩 id > 5 的消息（id 1~5 的被删掉）
        ```

        ### 内部执行流程
        - step1: 把 max_id 记录到 self.deleted_before，留作测试检查
        - step2: 重建 self._messages：只保留 id 大于 max_id 的消息，其余删除
        """
        self.deleted_before = max_id
        self._messages = [m for m in self._messages if m["id"] > max_id]


def test_count_tokens_basic():
    """
    **测试 count_tokens 的基础行为：能算字符串，也能算消息列表，且列表计数多于纯文本。**

    为什么测这个：上下文窗口的全部策略（截断、摘要触发）都建立在"token 数算得准"
    的前提上。这个测试做两条最基本的断言：一是函数对普通字符串能返回正数，
    二是传入消息列表时，结果必须比只传内容文本更大——因为消息格式本身
    （role 字段、消息包装结构）也要占 token，这部分"结构性开销"必须被算进去。

    ### 输入/输出示例
    ```python
    # 本函数由 pytest 自动发现并执行，无参数无返回值
    test_count_tokens_basic()
    # → 两条 assert 全过则测试通过：
    #   count_tokens("hello") > 0
    #   count_tokens([{"role": "user", "content": "你好"}]) > count_tokens("你好")
    ```

    ### 内部执行流程
    - step1: 断言 count_tokens("hello") 的结果大于 0，验证函数对字符串能正常计数
    - step2: 构造一条用户消息（角色 + 内容 "你好"），分别对整条消息和纯文本 "你好" 计数
    - step3: 断言消息列表的 token 数严格大于纯文本的 token 数，验证 role 等包装开销被计入
    """
    assert count_tokens("hello") > 0
    msgs = [{"role": "user", "content": "你好"}]
    assert count_tokens(msgs) > count_tokens("你好")  # role overhead included


def test_split_head_tail_keeps_whole_turns():
    """
    **测试 _split_head_tail：切分对话时必须保留完整的"问答对"，不能从半中间断开。**

    为什么测这个：滚动摘要的前提是"旧消息可以整轮整轮地被摘要覆盖"。如果切分时
    把某个问答对劈成两半（比如只留问题、丢掉回答），摘要就会丢信息、上下文也会
    变得不完整。这个测试构造 5 个问答对（共 10 条消息），要求保留最近 2 轮，
    然后逐条验证切分结果的结构完整性。

    ### 输入/输出示例
    ```python
    # 输入：5 个问答对共 10 条消息，要求保留最近 2 轮
    head, tail = _split_head_tail(_msgs(5), keep_recent_turns=2)
    # → tail 有 4 条（最近 2 轮完整保留），且 tail[0] 是 user 开头
    #   head 有 6 条（剩下的前 3 轮，留给摘要处理）
    ```

    ### 内部执行流程
    - step1: 用 _msgs(5) 生成 10 条消息，调用 _split_head_tail 要求保留最近 2 轮
    - step2: 断言 tail 恰好 4 条消息（2 轮 × 每轮 2 条），说明尾部拿全了整个问答对
    - step3: 断言 tail 第一条的角色是 user，验证切分点落在轮次边界上，没有从半轮开始
    - step4: 断言 head 恰好 6 条消息（前 3 轮），验证切分后 head + tail 正好覆盖全部 10 条
    """
    head, tail = _split_head_tail(_msgs(5), keep_recent_turns=2)
    assert len(tail) == 4                      # last 2 complete turns
    assert tail[0]["role"] == "user"           # never starts mid-turn
    assert len(head) == 6


def test_split_head_tail_not_enough_turns():
    """
    **测试 _split_head_tail 的边界情况：对话轮数不够保留数时，全部保留、不做切分。**

    为什么测这个：用户可能只聊了一轮就触发检查。此时如果还硬切，head 会是空的、
    tail 也凑不满，逻辑容易出 bug（比如把仅有的消息也归进 head，导致 tail 为空、
    上下文丢失）。这个测试传 1 个问答对却要求保留 4 轮，验证函数在这种"不够切"
    的情况下优雅处理：head 为空，所有消息都留在 tail。

    ### 输入/输出示例
    ```python
    # 输入：只有 1 个问答对（2 条消息），却要求保留最近 4 轮
    head, tail = _split_head_tail(_msgs(1), keep_recent_turns=4)
    # → head == []（没有旧消息可分），tail 保留全部 2 条消息
    ```

    ### 内部执行流程
    - step1: 用 _msgs(1) 生成 2 条消息，调用 _split_head_tail 要求保留最近 4 轮
    - step2: 断言 head 是空列表（轮数不够，没有旧消息可切出来）
    - step3: 断言 tail 有 2 条消息（全部对话都保留在尾部，一条不丢）
    """
    head, tail = _split_head_tail(_msgs(1), keep_recent_turns=4)
    assert head == [] and len(tail) == 2


def test_build_context_triggers_rolling_summary(monkeypatch):
    """
    **测试 build_context 的完整滚动摘要流程：触发摘要、保存摘要、删除旧消息、把摘要放进上下文。**

    为什么测这个：这是上下文引擎最核心的链路——token 超过阈值后，要调 LLM 生成
    摘要、把摘要写回存储、删掉被摘要覆盖的旧消息、并在最终上下文里放上摘要。
    测试里 LLM 调用被 monkeypatch 替换成了返回固定字符串的假实现，这样不依赖
    真实模型也能验证整条链路。配置里把 summary_trigger_tokens 压到 5，是人为
    制造"必定触发摘要"的条件。

    ### 输入/输出示例
    ```python
    # 由 pytest 自动执行；monkeypatch 是 pytest 内置 fixture
    test_build_context_triggers_rolling_summary(monkeypatch)
    # → 通过则证明四件事同时成立：
    #   store.summary == "滚动摘要内容"          （摘要已保存）
    #   store.deleted_before is not None        （旧消息已被删除）
    #   上下文里存在含"对话摘要"字样的消息       （摘要已放进上下文）
    ```

    ### 内部执行流程
    - step1: 用 monkeypatch 把 context_engine 模块里的 complete 函数替换为返回 "滚动摘要内容" 的假实现，截断对真实 LLM 的调用
    - step2: 构造一份把触发阈值压到 5 token 的配置，确保 4 轮对话必然触发摘要
    - step3: 用 FakeStore 装载 4 个问答对的假消息，调用 build_context 构建上下文
    - step4: 断言 store.summary 等于假 LLM 返回的摘要内容，验证摘要被保存
    - step5: 断言 store.deleted_before 不是 None，验证旧消息删除操作确实发生
    - step6: 断言返回的上下文里存在内容含"对话摘要"字样的消息，验证摘要出现在最终上下文中
    """
    monkeypatch.setattr(context_engine, "complete",
                        lambda client, cfg, messages: "滚动摘要内容")
    cfg = {"memory": {"summary_trigger_tokens": 5,   # force summarization
                      "max_context_tokens": 4000, "keep_recent_turns": 1}}
    store = FakeStore(_msgs(4))
    ctx = build_context(client=None, cfg=cfg, store=store, session_id="t")
    # summary persisted, old messages deleted from store, summary in context
    assert store.summary == "滚动摘要内容"
    assert store.deleted_before is not None
    assert any("对话摘要" in m["content"] for m in ctx)


def test_build_context_safety_net_keeps_latest_message():
    """
    **测试 build_context 的"最后一条消息兜底"机制：无论怎么裁剪，最新的助手回复绝不被丢弃。**

    为什么测这个：极端配置下（比如 max_context_tokens 压到 1），任何消息都"超限"，
    裁剪逻辑可能把消息删光。但用户最新的提问和助手的最新回复是回答下一轮问题的
    直接依据，绝不能丢。这个测试用 10 亿的不可能触发的摘要阈值 + 1 token 的
    极限裁剪上限，构造最恶劣场景，验证最后一条消息仍然幸存。

    ### 输入/输出示例
    ```python
    # 输入：2 个问答对，max_context_tokens=1（一切都被视为超限）
    ctx = build_context(client=None, cfg=cfg, store=store, session_id="t")
    # → 通过则 ctx[-1]["content"] == "回答1"，即裁剪后最后一条仍是最新回复
    ```

    ### 内部执行流程
    - step1: 构造一份"永不触发摘要"（阈值 10**9）且"裁剪极狠"（上限 1 token）的配置
    - step2: 用 FakeStore 装载 2 个问答对的假消息，调用 build_context
    - step3: 断言返回上下文的最后一条消息内容是 "回答1"，验证兜底机制在极限场景下保住了最新消息
    """
    cfg = {"memory": {"summary_trigger_tokens": 10**9,   # no summarization
                      "max_context_tokens": 1,           # trims everything
                      "keep_recent_turns": 1}}
    store = FakeStore(_msgs(2))
    ctx = build_context(client=None, cfg=cfg, store=store, session_id="t")
    assert ctx[-1]["content"] == "回答1"  # latest message survives the trim