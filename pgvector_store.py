"""PostgreSQL + pgvector backend for the RAG vector store.

Dialect contrasts with SqliteVecStore (the learning point of this backend):

    sqlite-vec                          pgvector
    two tables joined by rowid          one table, embedding is a column
    vec0 virtual table                  vector(N) column type (extension)
    WHERE embedding MATCH ? AND k=?     ORDER BY embedding <=> %s LIMIT %s
    serialize_float32 blob              '[v1,v2,...]'::vector text cast

Connection settings come from config `memory.postgres`; the password comes
ONLY from MODELCLAW_PG_PASSWORD (.env) — never from config.json.
"""

import os

import psycopg
from psycopg.rows import dict_row

from rag_store import DimensionMismatchError, RagStore

_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS rag_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _vec_literal(vector: list[float]) -> str:
    """
    **把一个 Python 浮点列表转成 pgvector 认识的文本字面量形式。**

    pgvector 扩展接受 `'[1,2,3]'` 这样格式的字符串表示向量，
    配合 `::vector` 类型转换即可写入或查询。本函数负责把
    Python 侧的 `[0.012, -0.339, ...]` 拼成同格式的字符串。

    ### 输入/输出示例

    ```python
    _vec_literal([0.012, -0.339, 0.881])
    # → '[0.012,-0.339,0.881]'   ← 方括号包裹，逗号分隔，无空格

    _vec_literal([1.0, 2.0, 3.0])
    # → '[1.0,2.0,3.0]'          ← 统一转成浮点形式，即使传入的是整数
    ```

    ### 内部执行流程
    - step1: 逐个取出向量中的数，先转成 float 再用 `repr` 取规范
            的字符串形式（保证总是"数字"的形态，不会因整数输入
            或其他类型混入破坏格式）；
    - step2: 用逗号把所有数的字符串拼接起来；
    - step3: 两端补上 `[` 和 `]`，返回最终字面量字符串。
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgVectorStore(RagStore):
    """Server-side vector backend: shares the deployment with session storage."""

    def __init__(self, pg_cfg: dict):
        """
        **连接 PostgreSQL 数据库并初始化 pgvector 扩展和元数据表。**

        连接参数（主机、端口、用户、库名）取自传入的配置；
        密码只认环境变量 `MODELCLAW_PG_PASSWORD`（放在 .env 里），
        不允许写进配置文件——配置文件可能被分享或入库，密码不能跟着泄漏。
        初始化完成后，pgvector 扩展就绪、rag_meta 表就绪；
        存向量的 rag_chunks 表不在这里建——它推迟到首次写入向量时
        按维度创建（见 `_check_or_init_dim`）。

        ### 输入/输出示例

        ```python
        # 配置齐全且环境变量已配置密码
        PgVectorStore({"host": "127.0.0.1", "port": 5432,
                       "user": "postgres", "database": "modelclaw"})
        # → 对象就绪，vector 扩展已创建，rag_meta 表已建好

        # 环境变量没配密码
        PgVectorStore({...})
        # → 抛出 RuntimeError: 未找到 MODELCLAW_PG_PASSWORD，请在 .env 文件中配置
        ```

        ### 内部执行流程
        - step1: 从配置读连接参数（都有默认值），从环境变量读密码；
        - step2: 密码为空 → 直接报错，提示去 .env 配置，不发出必然失败的连接；
        - step3: 建立连接，创建 pgvector 扩展（`IF NOT EXISTS`，已装则跳过）；
        - step4: 创建 rag_meta 元数据表（已存在则跳过），连接关闭，对象就绪。
        """
        self.host = pg_cfg.get("host", "127.0.0.1")
        self.port = int(pg_cfg.get("port", 5432))
        self.user = pg_cfg.get("user", "postgres")
        self.dbname = pg_cfg.get("database", "modelclaw")
        self.password = os.environ.get("MODELCLAW_PG_PASSWORD", "")
        if not self.password:
            raise RuntimeError(
                "未找到 MODELCLAW_PG_PASSWORD，请在 .env 文件中配置 PostgreSQL 密码"
            )
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(_SCHEMA_META)

    def _connect(self) -> psycopg.Connection:
        """
        **创建一个新的 PostgreSQL 连接。**

        查询结果以字典形式返回（列名 → 值），与 SqliteVecStore 的
        行为保持一致，上层代码不用关心用的是哪个后端。
        返回的连接适用于 with 语句：块正常结束自动提交，抛出异常
        自动回滚，且退出时自动关闭——调用方不用手动管理这三件事。

        ### 输入/输出示例

        ```python
        with self._connect() as conn:
            conn.execute("INSERT ...")   # 块正常结束 → 自动提交并关闭
        # 连接已关闭，不能再用了

        with self._connect() as conn:
            conn.execute("INSERT ...")
            raise ValueError("出错了")    # 块抛出异常 → 自动回滚并关闭
        ```

        ### 内部执行流程
        - step1: 用初始化时存的连接参数（含密码）发起连接；
        - step2: 指定行工厂为 dict_row——之后每行查询结果都是
                `{列名: 值}` 的字典，而不是默认的元组；
        - step3: 返回连接对象，交给调用方的 with 块使用，
                提交/回滚/关闭由 with 协议托管。
        """
        # psycopg3: `with conn:` commits on success AND closes on exit.
        return psycopg.connect(
            host=self.host, port=self.port, user=self.user,
            password=self.password, dbname=self.dbname, row_factory=dict_row,
        )

    # ------------------------------------------------------------------
    # dimension bookkeeping
    # ------------------------------------------------------------------

    def _recorded_dim(self, conn) -> int | None:
        """
        **读出知识库当前记录的向量维度。**

        维度在首次写入向量时记入 `rag_meta` 表，之后每次读写都以此为准。
        库是全新的（还没灌过任何向量）时返回 None。

        ### 输入/输出示例

        ```python
        # rag_meta 表中已记录 embedding_dim = 512
        self._recorded_dim(conn)
        # → 512

        # 全新知识库，rag_meta 里没有这条记录
        self._recorded_dim(conn)
        # → None
        ```

        ### 内部执行流程
        - step1: 查询 `rag_meta` 表中 key 为 'embedding_dim' 的记录；
        - step2: 有记录则把值转成整数返回；没有记录（None）原样返回。
        """
        row = conn.execute(
            "SELECT value FROM rag_meta WHERE key = 'embedding_dim'").fetchone()
        return int(row["value"]) if row else None

    def _check_or_init_dim(self, conn, dim: int) -> None:
        """
        **写入向量前的维度把关：首次写入时建好向量表并记下维度；
        之后每次写入校验维度一致，不一致则报错拒绝。**

        pgvector 的向量维度声明在列类型里（`vector(512)`），建表后
        无法更改——所以这个函数保证一个知识库永远只用一种维度的向量：
        首次写入按实际维度建表，换模型导致维度变化时给出明确的报错
        和补救指引，而不是让数据库抛一个看不懂的类型错误。

        ### 输入/输出示例

        ```python
        # 例1：首次写入，库是空的 → 建 rag_chunks 表（512 维向量列），
        #      记下维度，正常返回
        self._check_or_init_dim(conn, 512)

        # 例2：再次写入，维度仍是 512 → 校验通过，什么都不发生
        self._check_or_init_dim(conn, 512)

        # 例3：换模型后写入，维度变成 1024 → 抛出 DimensionMismatchError：
        #   向量维度不匹配：知识库是 512 维，当前模型输出 1024 维。
        #   请 modelclaw docs --clear 清空后用新模型重灌
        ```

        ### 内部执行流程
        - step1: 读出库里已记录的维度（`_recorded_dim`）；
        - step2: 没有记录（首次写入）→ 创建 rag_chunks 表：
                文字列（来源、序号、内容）和向量列在同一张表里，
                向量列类型为 `vector(当前维度)`，同时给来源列建索引，
                最后把维度写进 `rag_meta`；
        - step3: 有记录且与当前维度一致 → 什么都不做，直接通过；
        - step4: 有记录但不一致 → 抛出 `DimensionMismatchError`，
                报文中说明两边各是多少维、以及如何重建（先清空再重灌）。
        """
        recorded = self._recorded_dim(conn)
        if recorded is None:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    source      TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector({dim}) NOT NULL
                )
                """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_source ON rag_chunks(source)")
            conn.execute(
                "INSERT INTO rag_meta(key, value) VALUES ('embedding_dim', %s)", (str(dim),))
        elif recorded != dim:
            raise DimensionMismatchError(
                f"向量维度不匹配：知识库是 {recorded} 维，当前模型输出 {dim} 维。"
                "请 modelclaw docs --clear 清空后用新模型重灌"
            )

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add_chunks(self, source: str, contents: list[str], vectors: list[list[float]]) -> int:
        """
        **把同一个文件的若干文本块连同它们的向量一起写入知识库，返回写入的块数。**

        与 SQLite 后端"两张表分开存"不同，pgvector 后端把文字和向量
        放在同一张表的同一行里（embedding 就是一个普通列），一次插入搞定。
        写入前先做维度把关——首次写入会按向量维度建好表，维度不一致则报错拒绝。

        ### 输入/输出示例

        ```python
        add_chunks("guide.md",
                   ["打开设置页面，点击账户", "密码需包含大小写字母"],
                   [[0.012, -0.339, ...], [-0.221, 0.447, ...]])
        # rag_chunks 表新增 2 行，每行同时含文字和向量：
        #   id=1  source="guide.md"  chunk_index=0  content="打开设置页面..."  embedding=[向量A]
        #   id=2  source="guide.md"  chunk_index=1  content="密码需包含..."    embedding=[向量B]
        # → 返回 2

        # 维度与知识库不符时（如库里是 512 维，传入 1024 维）
        # → 抛出 DimensionMismatchError，一行都不会写入
        ```

        ### 内部执行流程
        - step1: 打开连接（退出时自动提交并关闭）；
        - step2: 若这批向量非空，先做维度检查：首次写入建表并记录维度，
                维度不符直接抛错，中止整个写入；
        - step3: 逐个块配对（第 i 块内容 + 第 i 个向量），一次 INSERT
                写入同一行——向量的文本字面量经 `%s::vector` 转换后
                存入 embedding 列；
        - step4: 返回写入的块数。
        """
        with self._connect() as conn:
            if vectors:
                self._check_or_init_dim(conn, len(vectors[0]))
            for i, (content, vector) in enumerate(zip(contents, vectors)):
                conn.execute(
                    "INSERT INTO rag_chunks(source, chunk_index, content, embedding)"
                    " VALUES (%s, %s, %s, %s::vector)",
                    (source, i, content, _vec_literal(vector)),
                )
        return len(contents)

    def delete_source(self, source: str) -> int:
        """
        **把一个文件的全部块从知识库中删除，返回删掉的块数。**

        利用 PostgreSQL 的 RETURNING 语法：删除的同时把被删行的 id
        带回来，删了几行直接知道，不用再查一遍。
        向量表可能还没建过（懒加载建表）——表不存在时当作"没有可删的"，
        返回 0 而不是报错。

        ### 输入/输出示例

        ```python
        # guide.md 在库中有 3 个块
        delete_source("guide.md")
        # rag_chunks 表中 source="guide.md" 的 3 行被删除
        # → 返回 3

        # 库里没有 this.md
        delete_source("this.md")
        # → 返回 0

        # 向量表压根没建过（库是全新的）
        delete_source("any.md")
        # → 返回 0（不报"表不存在"的错误）
        ```

        ### 内部执行流程
        - step1: 执行按 source 删除，并要求返回被删行的 id 列表；
        - step2: 表不存在（还没灌过向量）→ 捕获该异常，返回 0；
        - step3: 返回被删行数，即该文件的块数。
        """
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "DELETE FROM rag_chunks WHERE source = %s RETURNING id", (source,)
                ).fetchall()
            except psycopg.errors.UndefinedTable:
                return 0
        return len(rows)

    def has_source(self, source: str) -> bool:
        """
        **检查一个文件是否已经入库，返回 True（已入库）或 False（未入库）。**

        只关心"有没有"，不关心有多少块——查到一个块就说明该文件入库过。
        向量表可能还没建过，表不存在时按"未入库"处理，返回 False。

        ### 输入/输出示例

        ```python
        has_source("guide.md")   # 已入库 → True
        has_source("this.md")    # 库里没有 → False
        has_source("any.md")     # 向量表没建过（全新库）→ False
        ```

        ### 内部执行流程
        - step1: 在 rag_chunks 表里查该 source 是否存在（最多查一条）；
        - step2: 表不存在 → 捕获异常，返回 False；
        - step3: 查到了返回 True，没查到返回 False。
        """
        with self._connect() as conn:
            try:
                return conn.execute(
                    "SELECT 1 FROM rag_chunks WHERE source = %s LIMIT 1", (source,)
                ).fetchone() is not None
            except psycopg.errors.UndefinedTable:
                return False

    def clear(self) -> None:
        """
        **清空整个知识库：删掉全部块和向量（同一张表），连维度记录一起抹掉。**

        用于重建场景——比如换了 embedding 模型（维度变化），旧向量全部作废，
        清空后才能用新模型重新灌入。删完后库回到"全新"状态：
        下次写入会按新模型的维度重新建表。

        ### 输入/输出示例

        ```python
        # 库中有 100 个块，维度记录为 512
        clear()
        # rag_chunks 表整体删除，rag_meta 里的维度记录删除
        # count_chunks() → 0
        # _recorded_dim() → None（恢复全新状态）
        ```

        ### 内部执行流程
        - step1: 整体删除 rag_chunks 表（DROP TABLE）——向量列的类型
                焊死在表结构里（vector(512)），连同表结构一起删掉，
                给新维度重建留路；
        - step2: 清空 rag_meta 表的维度记录——这是关键一步：
                不删掉它，下次写入时会拿旧维度和新模型比对、直接报错，
                库就永远锁死在旧模型上。
        """
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS rag_chunks")
            conn.execute("DELETE FROM rag_meta")

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], k: int) -> list[dict]:
        """
        **用查询向量在库中找"意思最近"的 k 个块（KNN 近邻搜索）。**

        用 pgvector 的余弦距离运算符 `<=>` 计算查询向量与每行的距离，
        距离越小越相关，取最小的 k 个按距离从小到大返回。
        与 SQLite 后端"向量表先筛、再 JOIN 回文字表"不同，这里文字和向量
        同表，一次查询直接拿到全部字段。查询前做维度校验，
        避免新旧模型的向量混着算。

        ### 输入/输出示例

        ```python
        search(query_vector, k=4)
        # → [
        #   {"source": "guide.md", "chunk_index": 2, "content": "打开设置页面...", "distance": 0.0512},
        #   {"source": "policy.md", "chunk_index": 0, "content": "密码需包含...", "distance": 0.2301},
        #   ... 共最多 4 个，distance 从小到大
        # ]

        # 知识库是空的（从没灌过向量，表还没建）
        search(query_vector, k=4)
        # → []

        # 查询向量维度与知识库不符（如库里 512 维，传入 1024 维）
        # → 抛出 DimensionMismatchError
        ```

        ### 内部执行流程
        - step1: 读出库里记录的维度。库是空的（None）→ 没有可搜的，返回空列表；
        - step2: 校验查询向量维度与库一致，不一致抛 DimensionMismatchError；
        - step3: 执行向量近邻查询：`embedding <=> 查询向量` 算出每行的
                余弦距离，按距离升序取前 k 行，文字字段同行带出；
        - step4: 返回结果列表（每行是 dict_row 形式的字典）。
        """
        with self._connect() as conn:
            recorded = self._recorded_dim(conn)
            if recorded is None:
                return []
            if recorded != len(query_vector):
                raise DimensionMismatchError(
                    f"向量维度不匹配：知识库是 {recorded} 维，当前模型输出 {len(query_vector)} 维。"
                    "请 modelclaw docs --clear 清空后用新模型重灌"
                )
            return conn.execute(
                """
                SELECT source, chunk_index, content,
                       embedding <=> %s::vector AS distance
                FROM rag_chunks
                ORDER BY distance
                LIMIT %s
                """,
                (_vec_literal(query_vector), k),
            ).fetchall()

    def list_sources(self) -> list[dict]:
        """
        **列出所有已入库的文件，以及每个文件各有多少块。**

        向量表可能还没建过，表不存在时按"没有已入库文件"处理，返回空列表。

        ### 输入/输出示例

        ```python
        list_sources()
        # → [
        #   {"source": "a.md", "chunks": 12},
        #   {"source": "b.md", "chunks": 7},
        # ]
        # 按文件名排序；库为空（或表未建）时返回 []
        ```

        ### 内部执行流程
        - step1: 按 source 分组统计块数，按文件名排序查询；
        - step2: 表不存在 → 捕获异常，返回空列表；
        - step3: 返回结果列表。
        """
        with self._connect() as conn:
            try:
                return conn.execute(
                    "SELECT source, COUNT(*) AS chunks FROM rag_chunks GROUP BY source ORDER BY source"
                ).fetchall()
            except psycopg.errors.UndefinedTable:
                return []

    def count_chunks(self) -> int:
        """
        **返回知识库中块的总数。**

        向量表可能还没建过，表不存在时返回 0。

        ### 输入/输出示例

        ```python
        count_chunks()   # 库中有 137 块 → 137
        # 空库（或表未建）→ 0
        ```

        ### 内部执行流程
        - step1: 统计 rag_chunks 表的总行数并返回；
        - step2: 表不存在 → 捕获异常，返回 0。
        """
        with self._connect() as conn:
            try:
                return conn.execute("SELECT COUNT(*) AS n FROM rag_chunks").fetchone()["n"]
            except psycopg.errors.UndefinedTable:
                return 0

    def all_chunks(self) -> list[dict]:
        """
        **取出库中全部块的完整信息，供 BM25 关键词检索使用。**

        关键词检索不依赖向量，需要拿到每块的原文做分词统计，
        所以这个方法会全量读出所有块。按入库顺序（id 从小到大）返回。
        向量表可能还没建过，表不存在时返回空列表。

        ### 输入/输出示例

        ```python
        all_chunks()
        # → [
        #   {"id": 1, "source": "a.md", "chunk_index": 0, "content": "打开设置页面..."},
        #   {"id": 2, "source": "a.md", "chunk_index": 1, "content": "密码需包含..."},
        #   ...
        # ]
        # 表未建时 → []
        ```

        ### 内部执行流程
        - step1: 全表查询所有块的 id、来源、序号、内容，按 id 排序；
        - step2: 表不存在 → 捕获异常，返回空列表；
        - step3: 返回结果列表。
        """
        with self._connect() as conn:
            try:
                return conn.execute(
                    "SELECT id, source, chunk_index, content FROM rag_chunks ORDER BY id"
                ).fetchall()
            except psycopg.errors.UndefinedTable:
                return []