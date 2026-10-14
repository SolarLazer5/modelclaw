"""Shared pytest fixtures: offline config + temp sqlite stores.

Tests that need a live PostgreSQL skip themselves when it is unreachable.
No test in the offline suite calls a real LLM or embedding API.
"""

import os

import pytest

from session_store import SqliteSessionStore
from rag_store import SqliteVecStore


@pytest.fixture
def sqlite_session_store(tmp_path):
    """
    **pytest fixture：给每个测试提供一个全新的、隔离的 SQLite 会话数据库。**

    测试不能直接操作项目里真实的 sessions.db，否则会污染用户数据，而且每次测试
    的结果会互相影响。pytest 自带一个名叫 tmp_path 的 fixture，它会给每个测试
    单独创建并提供一个临时的空目录（测试结束后自动清理）。本 fixture 就利用
    tmp_path 创建一个干净的测试数据库，供需要验证会话存储功能的测试使用。

    ### 输入/输出示例
    ```python
    # pytest 自动调用本 fixture，把返回值注入测试函数的参数里
    def test_save_session(sqlite_session_store):
        sqlite_session_store.save(...)   # 实际写入的是 tmp 目录下的测试库
        # 测试结束后，临时目录连同测试数据库一起被清理，不留下任何痕迹
    # → 返回一个 SqliteSessionStore 实例（底层是临时目录下的 sessions_test.db）
    ```

    ### 内部执行流程
    - step1: 调用方（pytest）准备好 tmp_path（本次测试专属的临时目录），把它作为参数传入
    - step2: 用 SqliteSessionStore 打开 tmp_path 下的 sessions_test.db 文件（文件不存在会自动新建）
    - step3: 返回这个数据库连接对象，pytest 把它交给声明了 sqlite_session_store 参数的测试函数
    """
    return SqliteSessionStore(tmp_path / "sessions_test.db")


@pytest.fixture
def sqlite_rag_store(tmp_path):
    """
    **pytest fixture：给每个测试提供一个全新的、隔离的 SQLite 向量数据库。**

    作用和 sqlite_session_store 相同，只是对象换成了向量知识库。测试向量存储
    （建表、写入、删除、清空等）时，需要一个不会影响真实知识库的干净数据库。
    同样利用 pytest 的 tmp_path，让每个测试都在自己的临时数据库上操作，彼此隔离。

    ### 输入/输出示例
    ```python
    # pytest 自动调用本 fixture，把返回值注入测试函数的参数里
    def test_add_chunks(sqlite_rag_store):
        sqlite_rag_store.add_chunks([...])  # 写入的是 tmp 目录下的测试库
        # 测试结束后临时目录被清理，写入的测试数据随之消失
    # → 返回一个 SqliteVecStore 实例（底层是临时目录下的 rag_test.db）
    ```

    ### 内部执行流程
    - step1: 调用方（pytest）准备好 tmp_path（本次测试专属的临时目录），把它作为参数传入
    - step2: 用 SqliteVecStore 打开 tmp_path 下的 rag_test.db 文件
    - step3: 返回这个向量库实例，pytest 把它交给声明了 sqlite_rag_store 参数的测试函数
    """
    return SqliteVecStore(tmp_path / "rag_test.db")


def _pg_available() -> bool:
    """
    **判断本机的 PostgreSQL 数据库当前是否可连接（返回 True/False）。**

    为什么需要它：pgvector（向量库）相关测试只有连上真实的 PostgreSQL 才能跑。
    但不是每台开发机器都装了数据库（比如 CI 环境或别人的电脑），所以要在测试
    开始前先探一下路。连得上就正常跑测试，连不上就把这些测试标记为"跳过"，
    不让整个测试套件报错失败。这个检查不影响测试结果，只是决定跑还是不跑。

    ### 输入/输出示例
    ```python
    _pg_available()
    # → True  能连上 127.0.0.1:5432 且密码正确（integration 测试正常运行）
    _pg_available()
    # → False 没设 MODELCLAW_PG_PASSWORD 环境变量，或连接失败/超时（相关测试被跳过）
    ```

    ### 内部执行流程
    - step1: 调用 load_dotenv() 把项目 .env 文件里的环境变量加载进来，读取数据库密码 MODELCLAW_PG_PASSWORD
    - step2: 如果密码是空的，直接返回 False（没有密码一定连不上，不用继续试）
    - step3: 尝试用 psycopg 连接 127.0.0.1:5432 的 postgres 库，连接超时时间限制为 3 秒
    - step4: 连接成功后立刻断开并返回 True
    - step5: 连接过程中出现任何异常（超时、密码错误、服务没启动）都捕获并返回 False
    """
    from dotenv import load_dotenv
    load_dotenv()
    password = os.environ.get("MODELCLAW_PG_PASSWORD", "")
    if not password:
        return False
    try:
        import psycopg
        conn = psycopg.connect(
            host="127.0.0.1", port=5432, user="postgres",
            password=password, dbname="postgres", connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


pg_required = pytest.mark.skipif(not _pg_available(), reason="本地 PostgreSQL 不可用")


@pytest.fixture
def pg_cfg():
    """
    **pytest fixture：提供连接测试用 PostgreSQL 所需的配置参数。**

    pgvector 相关测试不仅需要数据库连得通，还需要知道连哪个库、用什么账号。
    这些连接参数在测试代码里会反复用到，抽成 fixture 之后，测试函数直接声明
    pg_cfg 参数就能拿到，不用在每个测试里重复写一遍。

    ### 输入/输出示例
    ```python
    # pytest 自动调用本 fixture，把返回值注入测试函数的参数里
    def test_pg_add(pg_cfg):
        store = PgVectorStore(pg_cfg)   # pg_cfg 就是下面这个字典
    # → 返回字典 {"host": "127.0.0.1", "port": 5432,
    #             "user": "postgres", "database": "modelclaw_test"}
    ```

    ### 内部执行流程
    - step1: pytest 发现某个测试函数声明了 pg_cfg 参数，自动调用本 fixture
    - step2: 返回写死的数据库连接信息（本机地址、端口、账号、专门的测试库名 modelclaw_test）
    - step3: pytest 把返回值注入测试函数，测试用这些参数创建向量库实例
    """
    return {"host": "127.0.0.1", "port": 5432, "user": "postgres",
            "database": "modelclaw_test"}