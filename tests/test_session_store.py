"""Contract tests for SessionStore: the same suite runs against every backend,
proving drop-in compatibility (the Repository pattern made executable).
"""

import pytest

from tests.conftest import pg_required


def _exercise_store(store):
    """
    **契约测试核心：用同一套操作序列检查任意 SessionStore 后端的行为是否达标。**

    为什么需要它：项目里有两套会话存储实现（SqliteSessionStore 和
    PostgresSessionStore），它们对外承诺的行为必须完全一致——不管底层换成
    哪种数据库，上层代码都能直接换着用（这就是文件头提到的 Repository 模式，
    "可替换性"不能只靠口头保证）。这个函数把"达标所需的操作和断言"写成一份
    公共清单：任何后端实例传进来跑一遍，全部断言通过才算符合契约。同一套
    检查不用写两遍，也保证两套后端接受的是完全相同的检验标准。

    ### 输入/输出示例
    ```python
    _exercise_store(sqlite_session_store)   # 或传入 PostgresSessionStore 实例
    # → 无返回值；全部 assert 通过则该后端符合契约，任何一条失败说明实现有 bug
    ```

    ### 内部执行流程
    - step1: 创建会话并断言拿到有效会话 id；写入一条用户消息和一条助手消息，断言取回的消息角色顺序正确
    - step2: 断言会话列表里自动标题以"你好"开头（首条用户消息生成标题）、轮数为 1，验证自动命名和轮数统计
    - step3: 用会话 id 的最后 6 位做模糊查找，断言能找回完整 id；对不存在的片段断言抛 FileNotFoundError
    - step4: 写入摘要再读回，断言内容一致，验证摘要存取往返正确
    - step5: 删除单条消息，断言剩余消息数为 1，验证单条回滚
    - step6: 清空会话，断言消息为空且摘要被重置为空串，验证 clear 同时清两类数据
    - step7: 删除整个会话，断言再查不到该会话，验证删除彻底
    """
    """The contract: every backend must satisfy this exact behavior."""
    # create + message lifecycle
    sid = store.create_session()
    assert sid
    mid = store.add_message(sid, "user", "你好，我叫测试")
    store.add_message(sid, "assistant", "你好测试")
    msgs = store.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]

    # auto-title from first user message
    sessions = store.list_sessions()
    assert sessions[0]["title"].startswith("你好") and sessions[0]["turns"] == 1

    # fuzzy find
    assert store.find_session(sid[-6:]) == sid
    with pytest.raises(FileNotFoundError):
        store.find_session("不存在的片段xyz")

    # summary round-trip
    store.set_summary(sid, "摘要内容")
    assert store.get_summary(sid) == "摘要内容"

    # rollback single message
    store.delete_message(mid)
    assert len(store.get_messages(sid)) == 1

    # clear wipes messages + summary
    store.clear_session(sid)
    assert store.get_messages(sid) == [] and store.get_summary(sid) == ""

    # delete removes everything
    store.delete_session(sid)
    assert store.get_session(sid) is None


class TestSqliteSessionStore:
    """SqliteSessionStore 后端的契约测试与会话 id 唯一性测试。"""

    def test_contract(self, sqlite_session_store):
        """
        **对 SqliteSessionStore 跑公共契约清单，验证其全部对外行为达标。**

        本测试本身不含具体断言，而是调用 _exercise_store 把 sqlite 后端
        传入公共清单。断言集中在 _exercise_store 里，这里只负责提供
        conftest.py 准备好的临时数据库实例。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，sqlite_session_store 是 conftest.py 提供的临时库 fixture
        test_contract(sqlite_session_store)
        # → 无返回值；_exercise_store 内全部 assert 通过则测试通过
        ```

        ### 内部执行流程
        - step1: pytest 注入临时 sqlite 会话库实例
        - step2: 把它传给 _exercise_store，由后者执行会话生命周期全套断言
        """
        _exercise_store(sqlite_session_store)

    def test_same_second_id_collision(self, sqlite_session_store):
        """
        **测试会话 id 的唯一性：同一秒内连续创建两个会话，id 不允许相同。**

        为什么需要它：会话 id 很可能和时间戳有关（比如精确到秒），而用户完全
        可能在一秒内连续开两个新会话。如果 id 生成逻辑只用了秒级时间戳而没有
        额外的区分手段，两个会话就会撞 id，后一个会覆盖前一个的数据。这个测试
        在不做任何延时的情况下连建两个会话，专测这个最容易撞车的场景。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，sqlite_session_store 是 conftest.py 提供的临时库 fixture
        test_same_second_id_collision(sqlite_session_store)
        # → 无返回值；assert a != b 成立（两秒内连建的 id 不同）则测试通过
        ```

        ### 内部执行流程
        - step1: 连续调用两次 create_session()，中间不做任何等待，让两次创建几乎肯定落在同一秒内
        - step2: 断言两个返回的会话 id 不相等，证明 id 生成逻辑带了额外的唯一性保障
        """
        a = sqlite_session_store.create_session()
        b = sqlite_session_store.create_session()
        assert a != b


@pg_required
class TestPostgresSessionStore:
    """PostgresSessionStore 后端的契约测试，整个类在本地 PostgreSQL 不可用时跳过。"""

    def test_contract(self, pg_cfg):
        """
        **对 PostgresSessionStore 跑公共契约清单，验证其与 sqlite 后端行为一致。**

        作用和 TestSqliteSessionStore.test_contract 相同，只是测试对象换成
        PostgreSQL 实现，证明上层代码可以把 sqlite 无缝替换成 PG 而不改任何
        调用方式。类上的 @pg_required 装饰器保证本地 PostgreSQL 连不上时
        整个测试自动跳过（见 conftest.py）。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行（PG 可用时），pg_cfg 是 conftest.py 提供的连接配置
        test_contract(pg_cfg)
        # → 无返回值；_exercise_store 内全部 assert 通过则测试通过
        ```

        ### 内部执行流程
        - step1: 从 postgres_store 模块导入 PostgresSessionStore（延迟导入，避免 PG 不可用时导入失败）
        - step2: 用 pg_cfg 创建 PG 会话库实例
        - step3: 把实例传给 _exercise_store，执行与 sqlite 完全相同的全套契约断言
        """
        from postgres_store import PostgresSessionStore
        store = PostgresSessionStore(pg_cfg)
        _exercise_store(store)