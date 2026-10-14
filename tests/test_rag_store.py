"""RagStore contract tests with fake 3-dim vectors (no embedding model needed)
plus dimension-mismatch guard tests.
"""

import pytest

from rag_store import DimensionMismatchError
from tests.conftest import pg_required


def _vec(x, y, z):
    """
    **测试辅助函数：把三个数包装成一个 3 维浮点向量（列表）。**

    为什么需要它：向量库的写入和检索都需要向量，但测试里并不想真的加载 embedding
    模型（太慢、依赖外部环境）。embedding 模型本质上只是"把文本变成向量"，
    对向量库来说向量就是几个浮点数，所以测试直接手写简单的 3 维向量当输入。
    每个分量都转成 float，是为了保证和真实 embedding 产出的数据类型一致。

    ### 输入/输出示例
    ```python
    _vec(1, 0, 0)
    # → [1.0, 0.0, 0.0]   （三个 int 被转成三个 float 组成的列表）
    ```

    ### 内部执行流程
    - step1: 把参数 x、y、z 分别转成 float
    - step2: 组装成列表并返回
    """
    return [float(x), float(y), float(z)]


def _exercise_store(store):
    """
    **契约测试核心：用同一套操作序列检查任意 RagStore 实现的行为是否达标。**

    为什么需要它：项目里有两套向量库实现（SqliteVecStore 和 PgVectorStore），
    它们对外承诺的行为应该完全一致——写入能查到、最近邻检索正确、删除和清空
    生效。这个函数把"达标所需的操作和断言"写成一份公共清单：任何实现传进来
    跑一遍，全部断言通过才算符合契约（contract）。这样同一套检查不用写两遍，
    也保证两套实现接受的是完全相同的检验标准。

    ### 输入/输出示例
    ```python
    _exercise_store(sqlite_rag_store)   # 或传入 PgVectorStore 实例
    # → 无返回值；全部 assert 通过则该实现符合契约，任何一条失败说明实现有 bug
    ```

    ### 内部执行流程
    - step1: 写入两条 doc_a 的块，向量维度（3 维）由首次写入自动确定；断言块总数为 2，且 doc_a 在库中、doc_b 不在
    - step2: 用向量 [1,0,0] 做最近邻检索，断言排第一的命中来自 doc_a 且内容含"重试"，验证 KNN 排序正确
    - step3: 断言来源列表里 doc_a 的块数为 2；调用 delete_source("doc_a") 断言返回 2（删掉的块数）；再断言库已清空
    - step4: 写入一条 doc_b 的块，调用 clear() 断言块数为 0 且 all_chunks() 返回空列表，验证清空生效
    """
    # dimension is learned from the first write
    store.add_chunks("doc_a", ["重试策略内容", "无关内容"], [_vec(1, 0, 0), _vec(0, 1, 0)])
    assert store.count_chunks() == 2
    assert store.has_source("doc_a") and not store.has_source("doc_b")

    # KNN: nearest to [1,0,0] is the first chunk
    hits = store.search(_vec(1, 0, 0), 1)
    assert hits[0]["source"] == "doc_a" and "重试" in hits[0]["content"]

    # listing / delete / clear
    assert store.list_sources()[0]["chunks"] == 2
    assert store.delete_source("doc_a") == 2
    assert store.count_chunks() == 0

    store.add_chunks("doc_b", ["再来一条"], [_vec(0, 0, 1)])
    store.clear()
    assert store.count_chunks() == 0 and store.all_chunks() == []


class TestSqliteVecStore:
    """SqliteVecStore 实现的契约测试与维度防护测试。"""

    def test_contract(self, sqlite_rag_store):
        """
        **对 SqliteVecStore 跑公共契约清单，验证其全部对外行为达标。**

        本测试本身不含具体断言，而是调用 _exercise_store 把 sqlite 实现
        传入公共清单。断言集中在 _exercise_store 里，这里只负责提供
        conftest.py 准备好的临时数据库实例。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行，sqlite_rag_store 是 conftest.py 提供的临时库 fixture
        test_contract(sqlite_rag_store)
        # → 无返回值；_exercise_store 内全部 assert 通过则测试通过
        ```

        ### 内部执行流程
        - step1: pytest 注入临时 sqlite 向量库实例
        - step2: 把它传给 _exercise_store，由后者执行写入/检索/删除/清空的全套断言
        """
        _exercise_store(sqlite_rag_store)

    def test_dimension_mismatch_guard(self, sqlite_rag_store):
        """
        **测试维度防护：用 2 维向量查询 3 维的库，必须抛出 DimensionMismatchError。**

        为什么需要它：向量库的维度在首次写入时确定（这里是 3 维）。如果之后传入
        维度不一致的向量（比如换了个 embedding 模型），底层数据库不会报错，只会
        算出错误结果或直接在序列化时崩掉。防护机制要求在维度不匹配时立刻抛
        DimensionMismatchError，而不是让错误悄悄发生。本测试验证这个异常确实
        会被抛出。

        ### 输入/输出示例
        ```python
        # 先写入 3 维向量建库，再用 2 维向量查询
        test_dimension_mismatch_guard(sqlite_rag_store)
        # → 无返回值；查询时若抛出 DimensionMismatchError 则测试通过，
        #   没抛或抛了别的异常则测试失败
        ```

        ### 内部执行流程
        - step1: 写入一条 3 维向量，确定库的维度为 3
        - step2: 用 pytest.raises 上下文管理器包裹 search 调用：预期它抛出 DimensionMismatchError；函数体没抛或抛了别的异常，pytest 都会判失败
        - step3: 传入一个 2 维查询向量执行检索，触发维度校验并抛出异常
        """
        sqlite_rag_store.add_chunks("a", ["x"], [_vec(1, 0, 0)])
        with pytest.raises(DimensionMismatchError):
            sqlite_rag_store.search([1.0, 0.0], 1)  # 2-dim query into 3-dim store

    def test_empty_store_search(self, sqlite_rag_store):
        """
        **测试空库检索：库里什么都没有时，检索应返回空列表而不是报错。**

        为什么需要它：检索是 RAG 流程里的常规操作，用户完全可能在还没录入任何
        文档时就提问。空库检索属于边界情况，实现如果没处理好（比如直接访问
        空表结构、或对空结果做多余假设），可能抛异常。本测试验证空库也能安全
        返回空结果。

        ### 输入/输出示例
        ```python
        # 刚创建的临时库里没有任何数据，直接检索
        test_empty_store_search(sqlite_rag_store)
        # → 无返回值；assert 检索结果 == [] 成立则测试通过
        ```

        ### 内部执行流程
        - step1: pytest 注入一个刚创建、内容为空（或仅含元信息）的临时向量库
        - step2: 发起一次 3 维向量的 top-3 检索
        - step3: 断言返回结果是空列表
        """
        assert sqlite_rag_store.search(_vec(1, 0, 0), 3) == []


@pg_required
class TestPgVectorStore:
    """PgVectorStore 实现的契约测试，整个类在本地 PostgreSQL 不可用时跳过。"""

    def test_contract(self, pg_cfg):
        """
        **对 PgVectorStore 跑公共契约清单，验证其与 sqlite 实现行为一致。**

        作用和 TestSqliteVecStore.test_contract 相同，只是测试对象换成
        PostgreSQL 实现。测试前调用 clear() 清一次库，保证测试在干净的
        状态上运行，不受之前测试残留数据的影响。类上的 @pg_required 装饰器
        保证本地 PostgreSQL 连不上时整个测试自动跳过（见 conftest.py）。

        ### 输入/输出示例
        ```python
        # 由 pytest 自动执行（PG 可用时），pg_cfg 是 conftest.py 提供的连接配置
        test_contract(pg_cfg)
        # → 无返回值；_exercise_store 内全部 assert 通过则测试通过
        ```

        ### 内部执行流程
        - step1: 从 pgvector_store 模块导入 PgVectorStore（延迟导入，避免 PG 不可用时导入失败）
        - step2: 用 pg_cfg 创建 PG 向量库实例，并先调用 clear() 清空，确保测试起点干净
        - step3: 把实例传给 _exercise_store，执行与 sqlite 完全相同的全套契约断言
        """
        from pgvector_store import PgVectorStore
        store = PgVectorStore(pg_cfg)
        store.clear()  # start clean
        _exercise_store(store)