"""Vector stores for RAG: pluggable backends behind a RagStore contract.

    SqliteVecStore   default, zero-config local file via sqlite-vec
    PgVectorStore    PostgreSQL + pgvector (pgvector_store.py),
                     selected via config rag.backend = "pgvector"

Callers use create_rag_store(cfg) and never construct a backend directly.

Dimension handling: the embedding dimension is never hardcoded. The backend
learns it from the first batch of vectors written, records it in rag_meta,
and raises a friendly "rebuild required" error if a later model disagrees.

Vectors are DERIVED data: they can always be rebuilt from the source
documents by re-running `modelclaw ingest`.
"""

import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

_SCHEMA_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE TABLE IF NOT EXISTS rag_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class DimensionMismatchError(RuntimeError):
    """向量维度不匹配：知识库的维度与当前模型输出的维度不一致。"""
    pass


class RagStore(ABC):
    """向量库的接口约定：所有后端（sqlite-vec / pgvector）都实现这套方法。"""

    @abstractmethod
    def add_chunks(self, source: str, contents: list[str], vectors: list[list[float]]) -> int:
        """
        **把同一个文件的若干文本块连同向量写入知识库，返回写入的块数。**

        实现要求：contents 与 vectors 一一对应（第 i 块内容配第 i 个向量）；
        写入前必须做维度校验，维度不一致时报错拒绝，不允许混入不同维度的向量。

        ### 输入/输出示例

        ```python
        add_chunks("guide.md", ["块A内容", "块B内容"], [向量A, 向量B])
        # → 2
        ```

        ### 内部执行流程（约定）
        - step1: 校验向量维度与知识库一致（不一致抛 DimensionMismatchError）；
        - step2: 逐块写入内容和向量，建立两者的关联；
        - step3: 返回写入的块数。
        """

    @abstractmethod
    def delete_source(self, source: str) -> int:
        """
        **删除一个文件的全部块和向量，返回删除的块数。**

        文件不在库中时什么都不删，返回 0。

        ### 输入/输出示例

        ```python
        delete_source("guide.md")   # 已入库，删了 3 块 → 3
        delete_source("this.md")    # 不在库里        → 0
        ```

        ### 内部执行流程（约定）
        - step1: 找到该文件的全部块及其向量；
        - step2: 两边都删除，不留残留；
        - step3: 返回删除的块数。
        """

    @abstractmethod
    def has_source(self, source: str) -> bool:
        """
        **检查一个文件是否已入库，返回 True / False。**

        ### 输入/输出示例

        ```python
        has_source("guide.md")   # 已入库 → True
        has_source("this.md")    # 未入库 → False
        ```

        ### 内部执行流程（约定）
        - step1: 按 source 查询，查到任一即返回 True，否则 False。
        """

    @abstractmethod
    def clear(self) -> None:
        """
        **清空整个知识库：全部块、全部向量、维度记录一并抹掉。**

        清空后库回到"全新"状态，可换模型重新灌入。

        ### 输入/输出示例

        ```python
        clear()
        # count_chunks() → 0，维度记录消失
        ```

        ### 内部执行流程（约定）
        - step1: 删除向量数据（向量表维度焊死在结构里，通常需连表删除）；
        - step2: 清空全部块；
        - step3: 删除维度记录。
        """

    @abstractmethod
    def search(self, query_vector: list[float], k: int) -> list[dict]:
        """
        **用查询向量做 KNN 近邻搜索，返回距离最近的 k 个块。**

        返回每个块：来源文件、块序号、内容、距离（越小越相关）。

        ### 输入/输出示例

        ```python
        search(query_vector, k=4)
        # → [
        #   {"source": "guide.md", "chunk_index": 2, "content": "...", "distance": 0.05},
        #   ... 共最多 k 个，按距离从小到大
        # ]
        ```

        ### 内部执行流程（约定）
        - step1: 校验查询向量维度与知识库一致（不一致抛 DimensionMismatchError），
                空库返回空列表；
        - step2: 执行近邻搜索，取距离最小的 k 个；
        - step3: 按距离从小到大返回。
        """

    @abstractmethod
    def list_sources(self) -> list[dict]:
        """
        **列出所有已入库的文件及各自的块数。**

        ### 输入/输出示例

        ```python
        list_sources()
        # → [
        #   {"source": "a.md", "chunks": 12},
        #   {"source": "b.md", "chunks": 7},
        # ]
        ```

        ### 内部执行流程（约定）
        - step1: 按 source 分组统计块数，排序后返回。
        """

    @abstractmethod
    def count_chunks(self) -> int:
        """
        **返回知识库中块的总数。**

        ### 输入/输出示例

        ```python
        count_chunks()   # → 137
        ```

        ### 内部执行流程（约定）
        - step1: 统计全部块行数并返回。
        """

    @abstractmethod
    def all_chunks(self) -> list[dict]:
        """
        **取出全部块（id、来源、序号、内容），供 BM25 关键词检索使用。**

        ### 输入/输出示例

        ```python
        all_chunks()
        # → [
        #   {"id": 101, "source": "a.md", "chunk_index": 0, "content": "..."},
        #   ...
        # ]
        ```

        ### 内部执行流程（约定）
        - step1: 全表查询，按 id 顺序返回所有块。
        """


class SqliteVecStore(RagStore):
    """Zero-config local backend: sqlite-vec inside one file."""

    def __init__(self, db_path: str | Path):
        """
        **打开（不存在则创建）本地 SQLite 向量库文件，并初始化基础表结构。**

        数据库文件路径取自参数；所在目录不存在时自动创建。
        初始化只建与模型无关的表（chunks、rag_meta），向量表推迟到
        第一次写入向量时再按维度创建。

        ### 输入/输出示例

        ```python
        store = SqliteVecStore("output/rag.db")
        # 目录 output/ 不存在则自动创建
        # 文件 rag.db 不存在则新建，存在则直接打开
        # chunks / rag_meta 两张表就绪（已存在则跳过）
        ```

        ### 内部执行流程
        - step1: 保存数据库路径，确保所在目录存在；
        - step2: 打开连接，执行建表脚本（chunks 表 + source 索引 + rag_meta 表），
                `IF NOT EXISTS` 保证重复初始化不报错；
        - step3: 连接关闭，对象就绪。注意此时向量表尚未创建——
                它在首次写入向量时按实际维度建出（见 `_check_or_init_dim`）。
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.executescript(_SCHEMA_CHUNKS)

    @contextmanager
    def _db(self):
        """
        **数据库连接的上下文管理器：进入时打开连接，退出时提交并关闭。**

        用法 `with self._db() as conn:`——块内代码正常结束，事务自动提交；
        块内抛出异常，事务自动回滚，已做的修改全部撤销。
        连接在交给调用方前已加载 sqlite-vec 扩展（向量功能的前提）。

        ### 输入/输出示例

        ```python
        with self._db() as conn:
            conn.execute("INSERT ...")   # 块正常结束 → 自动提交
        # 连接已关闭

        with self._db() as conn:
            conn.execute("INSERT ...")
            raise ValueError("出错了")     # 块抛出异常 → 自动回滚，插入被撤销
        # 连接已关闭
        ```

        ### 内部执行流程
        - step1: 打开到 db_path 的连接，查询结果以"行对象"形式返回
                （支持 r["列名"] 取值）；
        - step2: 允许加载扩展，并加载 sqlite-vec 模块——
                之后的 SQL 才能使用向量类型和向量检索语法；
        - step3: 把连接交给 with 块使用；
        - step4: 块结束时：正常 → 提交全部修改；异常 → 回滚；
                无论哪种情况，连接最终都被关闭。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        #加载sqlitevec模块
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        try:
            with conn:
                yield conn
        finally:
            conn.close()

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

        向量表的列类型在建表时就要定死维度（如 float[512]），没法事后改；
        所以这个函数保证：一个知识库永远只用一种维度的向量——
        首次写入按实际维度建表，换模型导致维度变化时给出明确的报错和补救指引，
        而不是让数据库在深处报一个看不懂的错误。

        ### 输入/输出示例

        ```python
        # 例1：首次写入，库是空的 → 建 512 维的向量表，记下维度，正常返回
        self._check_or_init_dim(conn, 512)

        # 例2：再次写入，维度仍是 512 → 校验通过，什么都不发生
        self._check_or_init_dim(conn, 512)

        # 例3：换模型后写入，维度变成 1024 → 抛出 DimensionMismatchError：
        #   向量维度不匹配：知识库是 512 维，当前模型输出 1024 维。
        #   请 modelclaw docs --clear 清空后用新模型重灌
        ```

        ### 内部执行流程
        - step1: 读出库里已记录的维度（`_recorded_dim`）；
        - step2: 没有记录（首次写入）→ 按当前维度创建 vec0 向量虚表，
                并把维度写进 `rag_meta`；
        - step3: 有记录且与当前维度一致 → 什么都不做，直接通过；
        - step4: 有记录但不一致 → 抛出 `DimensionMismatchError`，
                报文中说明两边各是多少维、以及如何重建（先清空再重灌）。
        """
        recorded = self._recorded_dim(conn)
        if recorded is None:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])")
            conn.execute(
                "INSERT INTO rag_meta(key, value) VALUES ('embedding_dim', ?)", (str(dim),))
        elif recorded != dim:
            raise DimensionMismatchError(
                f"向量维度不匹配：知识库是 {recorded} 维，当前模型输出 {dim} 维。"
                "请 modelclaw docs --clear 清空后用新模型重灌"
            )
        ##知识点：
        ##1，如何理解CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])
        ##这是一个虚拟表建表语句，要理解它，首先要知道，在连接数据库时，懒加载了qlitevec模块，即下面代码：
        ##conn.enable_load_extension(True)
        ##sqlite_vec.load(conn)
        ##这使得conn.execute支持该与模块相关的sql语句，即USING vec0(embedding float[{dim}])，而这句虚拟表
        ##建表语句就是启用了vec0这个模块，这又使得和后面的MATCH ... and k = ...这类语句
        ##能够被底层识别而执行对应的操作。
        ##知道了这些，我们再看这个sql语句，他的意思是就是，创建一个虚拟表，加载vec0模块（需要embedding参数，
        ##这里我们实际传入的值为 float[{dim}])，它决定了该表存储向量的维度

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add_chunks(self, source: str, contents: list[str], vectors: list[list[float]]) -> int:
        """
        **把同一个文件的若干文本块连同它们的向量一起写入知识库，返回写入的块数。**

        每个块写进两张表：文字内容进 `chunks` 表，向量进 `vec_chunks` 虚表，
        两边用同一个行号关联。写入前先做维度把关——首次写入会按向量维度
        建好向量表，维度不一致则报错拒绝。

        ### 输入/输出示例

        ```python
        add_chunks("guide.md",
                   ["打开设置页面，点击账户", "密码需包含大小写字母"],
                   [[0.012, -0.339, ...], [-0.221, 0.447, ...]])
        # chunks 表新增 2 行：
        #   id=101  source="guide.md"  chunk_index=0  content="打开设置页面，点击账户"
        #   id=102  source="guide.md"  chunk_index=1  content="密码需包含大小写字母"
        # vec_chunks 虚表对应新增 2 行（rowid 分别为 101、102，各存一个向量）
        # → 返回 2

        # 维度与知识库不符时（如库里是 512 维，传入 1024 维）
        # → 抛出 DimensionMismatchError，一行都不会写入
        ```

        ### 内部执行流程
        - step1: 打开数据库连接（退出时自动提交并关闭）；
        - step2: 若这批向量非空，先做维度检查：首次写入建表并记录维度，
                维度不符直接抛错，中止整个写入；
        - step3: 逐个块配对（第 i 块内容 + 第 i 个向量）：
                先插入 `chunks` 表拿到自增 id，再把这个 id 作为 rowid，
                将序列化后的向量插入 `vec_chunks` 虚表——两张表靠这个
                相同的 id 关联；
        - step4: 返回写入的块数。
        """
        with self._db() as conn:
            if vectors:
                self._check_or_init_dim(conn, len(vectors[0]))
            for i, (content, vector) in enumerate(zip(contents, vectors)):
                cur = conn.execute(
                    "INSERT INTO chunks(source, chunk_index, content) VALUES (?, ?, ?)",
                    (source, i, content),
                )
                conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(vector)),
                )
        return len(contents)

    def delete_source(self, source: str) -> int:
        """
        **把一个文件的全部块（文字 + 向量）从知识库中删除，返回删掉的块数。**

        每个块的数据分散在两张表里（文字在 `chunks`，向量在 `vec_chunks`），
        两张表都要删；先记下该文件所有块的 id，删完文字行后按 id 逐个删向量行，
        保证两边都不残留。文件不在库中时什么都不删，返回 0。

        ### 输入/输出示例

        ```python
        # guide.md 在库中有 3 个块，id 为 101、102、103
        delete_source("guide.md")
        # chunks 表中 source="guide.md" 的 3 行被删除
        # vec_chunks 表中 rowid 为 101、102、103 的 3 行被删除
        # → 返回 3

        # 库里没有 this.md
        delete_source("this.md")
        # → 返回 0
        ```

        ### 内部执行流程
        - step1: 查出该文件所有块的 id 列表；
        - step2: 按 source 删除 `chunks` 表中的文字行；
        - step3: 逐个 id 删除 `vec_chunks` 虚表中的向量行——必须先取 id 再删，
                因为 step2 删完后这些 id 就查不到了，向量行会变成删不掉的残留；
        - step4: 返回删除的块数（文件不存在时为 0）。
        """
        with self._db() as conn:
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM chunks WHERE source = ?", (source,))]
            conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            for cid in ids:
                conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (cid,))
        return len(ids)

    def has_source(self, source: str) -> bool:
        """
        **检查一个文件是否已经入库，返回 True（已入库）或 False（未入库）。**

        只关心"有没有"，不关心有多少块——查到一个块就说明该文件入库过，
        立刻停止查询返回 True，不继续数下去。

        ### 输入/输出示例

        ```python
        has_source("guide.md")   # guide.md 已入库 → True
        has_source("this.md")    # 库里没有     → False
        ```

        ### 内部执行流程
        - step1: 在 `chunks` 表里查该 source 是否存在；
        - step2: `LIMIT 1` 限制最多查一条——存在与否由这一条决定，
                查到就停，不为"有多少块"浪费扫描；
        - step3: 查到了返回 True，没查到（fetchone 结果为 None）返回 False。
        """
        with self._db() as conn:
            return conn.execute(
                "SELECT 1 FROM chunks WHERE source = ? LIMIT 1", (source,)
            ).fetchone() is not None

    def clear(self) -> None:
        """
        **清空整个知识库：删掉全部块、全部向量，连维度记录一起抹掉。**

        用于重建场景——比如换了 embedding 模型（维度变化），旧向量全部作废，
        清空后才能用新模型重新灌入。删完后库回到"全新"状态：
        下次写入会按新模型的维度重新建向量表。

        ### 输入/输出示例

        ```python
        # 库中有 100 个块，维度记录为 512
        clear()
        # chunks 表清空，vec_chunks 表整体删除，rag_meta 里的维度记录删除
        # count_chunks() → 0
        # _recorded_dim() → None（恢复全新状态）
        ```

        ### 内部执行流程
        - step1: 整体删除 `vec_chunks` 虚表（DROP TABLE）——虚表不能只清行，
                连同表结构一起删掉，给新维度重建留路；
        - step2: 清空 `chunks` 表的全部文字行；
        - step3: 清空 `rag_meta` 表的维度记录——这是关键一步：
                不删掉它，下次写入时会拿旧维度和新模型比对、直接报错，
                库就永远锁死在旧模型上。
        """
        with self._db() as conn:
            conn.execute("DROP TABLE IF EXISTS vec_chunks")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM rag_meta")

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], k: int) -> list[dict]:
        """
        **用查询向量在库中找"意思最近"的 k 个块（KNN 近邻搜索）。**

        返回的每个块带距离值（distance），越小表示与问题越相关，
        结果按距离从小到大排好序。查询前做维度校验，避免新旧模型的
        向量混着算。

        ### 输入/输出示例

        ```python
        search(query_vector, k=4)
        # → [
        #   {"source": "guide.md", "chunk_index": 2, "content": "打开设置页面...", "distance": 0.0512},
        #   {"source": "policy.md", "chunk_index": 0, "content": "密码需包含...", "distance": 0.2301},
        #   ... 共最多 4 个，distance 从小到大
        # ]

        # 知识库是空的（从没灌过向量）
        search(query_vector, k=4)
        # → []

        # 查询向量维度与知识库不符（如库里 512 维，传入 1024 维）
        # → 抛出 DimensionMismatchError
        ```

        ### 内部执行流程
        - step1: 读出库里记录的维度。库是空的（None）→ 没有可搜的，返回空列表；
        - step2: 校验查询向量维度与库一致，不一致抛 DimensionMismatchError；
        - step3: 执行向量近邻查询：在 `vec_chunks` 虚表中按向量距离
                找出最近的 k 个，同时按相同的行号（v.rowid = c.id）
                从 `chunks` 表联查出文字内容；
        - step4: 把每行结果转成普通字典返回。
        """
        with self._db() as conn:
            recorded = self._recorded_dim(conn)
            if recorded is None:
                return []
            if recorded != len(query_vector):
                raise DimensionMismatchError(
                    f"向量维度不匹配：知识库是 {recorded} 维，当前模型输出 {len(query_vector)} 维。"
                    "请 modelclaw docs --clear 清空后用新模型重灌"
                )
            rows = conn.execute(
                """
                SELECT c.source, c.chunk_index, c.content, v.distance
                FROM vec_chunks v
                JOIN chunks c ON c.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (sqlite_vec.serialize_float32(query_vector), k),
            ).fetchall()
            ##知识点：
            ##1，vec_chunks中的distance是哪来的？
            ##因为vec_chunks是一个虚拟表（CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks），
            ##启用了向量扩展模块(即vec0，语句为：USING vec0(embedding float[{dim}]))，
            ##而这个模块中，底层对这个虚拟表声明了一个隐藏列“distance”，让sql语句能够直接在查询中引用
            ##
            ##2，如何理解WHERE v.embedding MATCH ? AND k = ?这个sql语句？k哪来的？
            ##因为vec_chunks表是一个启用了vec0模块的虚拟表，这个语句是与这个vec0模块的虚拟表的规定好的“交流方式”
            ##底层会自己识别这些参数，在sql语句表层，这么理解它：MATCH后面的是要比较的“查询向量”，AND k是需要返回
            ##的与“查询向量”最近（distance距离最近）的前top k个。我们可以对启用了vec0模块的虚拟表这么理解，同理
            ##其他不同模块的虚拟表也有对应的“交流方式”
            ##
            ##3,为什么先调sqlite_vec.serialize_float32(query_vector)，而不是直接写query_vector？
            ##与第2点同理，这是启用vec0模块的虚拟表的“交流方式”，底层vec0模块接受的“查询向量”是一个“float32 格式（4 字节）
            ##顺序紧凑地拼成一段字节串”
            

        return [dict(r) for r in rows]

    def list_sources(self) -> list[dict]:
        """
        **列出所有已入库的文件，以及每个文件各有多少块。**

        ### 输入/输出示例

        ```python
        list_sources()
        # → [
        #   {"source": "a.md", "chunks": 12},
        #   {"source": "b.md", "chunks": 7},
        # ]
        # 按文件名排序；库为空时返回 []
        ```

        ### 内部执行流程
        - step1: 按 source 分组统计块数，按文件名排序查询；
        - step2: 把每行结果转成字典返回。
        """
        with self._db() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS chunks FROM chunks GROUP BY source ORDER BY source"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_chunks(self) -> int:
        """
        **返回知识库中块的总数。**

        ### 输入/输出示例

        ```python
        count_chunks()   # 库中有 137 块 → 137
        # 空库 → 0
        ```

        ### 内部执行流程
        - step1: 统计 chunks 表的总行数并返回。
        """
        with self._db() as conn:
            return conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    def all_chunks(self) -> list[dict]:
        """
        **取出库中全部块的完整信息，供 BM25 关键词检索使用。**

        关键词检索不依赖向量，需要拿到每块的原文做分词统计，
        所以这个方法会全量读出所有块。按入库顺序（id 从小到大）返回。

        ### 输入/输出示例

        ```python
        all_chunks()
        # → [
        #   {"id": 101, "source": "a.md", "chunk_index": 0, "content": "打开设置页面..."},
        #   {"id": 102, "source": "a.md", "chunk_index": 1, "content": "密码需包含..."},
        #   ...
        # ]
        ```

        ### 内部执行流程
        - step1: 全表查询所有块的 id、来源、序号、内容，按 id 排序；
        - step2: 把每行结果转成字典返回。
        """
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, source, chunk_index, content FROM chunks ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def create_rag_store(cfg: dict) -> RagStore:
    """
    **根据配置创建对应的向量库后端。调用方只跟这个方法打交道，
    不用关心底层用的是 SQLite 还是 PostgreSQL。**

    配置项 `rag.backend` 决定用哪个：
    "sqlite-vec"（默认）→ 本地 SQLite 文件，零配置；
    "pgvector" → PostgreSQL，只有选了它才会加载对应驱动和模块。
    配置写得不认识时直接报错，列出可选值。

    ### 输入/输出示例

    ```python
    # 配置：{"rag": {}}（没写 backend）
    create_rag_store(cfg)          # → SqliteVecStore("output/rag.db")

    # 配置：{"rag": {"backend": "sqlite-vec", "db_path": "data/kb.db"}}
    create_rag_store(cfg)          # → SqliteVecStore("data/kb.db")

    # 配置：{"rag": {"backend": "pgvector"}, "memory": {"postgres": {...}}}
    create_rag_store(cfg)          # → PgVectorStore(...)

    # 配置：{"rag": {"backend": "mysql"}}
    create_rag_store(cfg)
    # → 抛出 ValueError: 未知的 rag.backend: mysql（可选: sqlite-vec / pgvector）
    ```

    ### 内部执行流程
    - step1: 从配置的 `rag` 段读出 backend，没写就用默认值 "sqlite-vec"；
    - step2: 按 backend 分支创建对应后端的实例：
            sqlite-vec → 用配置里的 db_path（默认 output/rag.db）；
            pgvector → 延迟加载 PgVectorStore 模块
            （只有走这条路才需要安装 PostgreSQL 驱动），
            并传入 memory.postgres 配置；
    - step3: 两个分支都没命中 → 抛出 ValueError，
            报文里说明写错了什么、有哪些可选值。
    """
    rag_cfg = cfg.get("rag", {})
    backend = rag_cfg.get("backend", "sqlite-vec")
    if backend == "sqlite-vec":
        return SqliteVecStore(rag_cfg.get("db_path", "output/rag.db"))
    if backend == "pgvector":
        from pgvector_store import PgVectorStore  # lazy: psycopg only needed here
        return PgVectorStore(cfg.get("memory", {}).get("postgres", {}))
    raise ValueError(f"未知的 rag.backend: {backend}（可选: sqlite-vec / pgvector）")