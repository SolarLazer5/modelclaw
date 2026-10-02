"""Session persistence: pluggable storage backends for chat sessions and messages.

Architecture (Repository pattern): `SessionStore` is the abstract contract that
the CLI and context engine depend on; concrete backends implement it:

    SqliteSessionStore    default, zero-config local file (this module)
    PostgresSessionStore  optional server backend (postgres_store.py),
                          selected via config memory.backend = "postgres"

Callers never construct a backend directly — they use create_session_store(cfg).

Schema (identical across backends, dialect differences isolated per backend):

    sessions(session_id PK, title, summary, created_at, updated_at)
    messages(id PK, session_id FK, role, content, created_at)

`sessions.summary` holds the rolling summary produced by context_engine when
the history grows too large; raw messages stay in `messages` until summarized.
"""

import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path


def _now() -> str:
    """返回格式化后的时间字符串"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


class SessionStore(ABC):
    """Interface contract for session storage backends.

    The 12 methods below are the complete surface used by modelclaw.py and
    context_engine.py — any backend implementing them is drop-in compatible.
    """
    ##知识点：
    # 1，ABC
    # ABC是抽象基类，定义抽象基类时，要继承这个ABC

    @abstractmethod
    def create_session(self, session_id: str | None = None) -> str:
        """Create (or reuse) a session and return its id."""
    ##知识点：
    # 1.@abstractmethod修饰器
    # 继承这个抽象基类的类，其中这个方法必须在子类中被实现

    @abstractmethod
    def find_session(self, fragment: str) -> str:
        """Resolve a session by exact id or unique substring fragment.

        Raises FileNotFoundError when there is no match or several matches.
        """

    @abstractmethod
    def list_sessions(self) -> list:
        """All sessions with user-turn counts, most recently active first.

        Each row supports key access: session_id / title / summary /
        created_at / updated_at / turns.
        """

    @abstractmethod
    def get_session(self, session_id: str):
        """One session row (key-accessible) or None."""

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages."""

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> int:
        """Append a message; touch updated_at; auto-title from first user message.

        Returns the new message id.
        """

    @abstractmethod
    def get_messages(self, session_id: str) -> list[dict]:
        """Raw history as [{'id', 'role', 'content'}, ...] in chronological order."""

    @abstractmethod
    def delete_message(self, message_id: int) -> None:
        """Delete one message by id (used to roll back a failed turn)."""

    @abstractmethod
    def delete_messages_before(self, session_id: str, max_id: int) -> None:
        """Delete all messages with id <= max_id (used after summarization)."""

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Wipe a session's messages and summary (the `/clear` semantics)."""

    @abstractmethod
    def get_summary(self, session_id: str) -> str:
        """The session's rolling summary ('' when absent)."""

    @abstractmethod
    def set_summary(self, session_id: str, summary: str) -> None:
        """Persist the session's rolling summary."""


# ---------------------------------------------------------------------------
# SQLite backend (default)
# ---------------------------------------------------------------------------

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""
##知识点：
# 1,session_id TEXT NOT NULL REFERENCES sessions(session_id)
# REFERENCES代表将该字段作为外键，指向该REFERENCES后面某个表的某个字段（该字段为唯一值“主键”或“UNIQUE”约束）
# 2,session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE
#  ON DELETE CASCADE通常配合REFERENCES使用(它独立的 SQL 语法，它只能出现在外键约束的定义里)，
# 代表删除sessions表中的该session_id字段时，删除当前表所有的session_id值为目标session_id的所有行
# 3.配合REFERENCES ....（写外表的唯一字段如sessions(session_id)） ON DELETE ...(如下所有形式)
# | `ON DELETE CASCADE`                | 跟着一起删                         |
# | `ON DELETE SET NULL`               | 子行的外键列设为 NULL（子行保留）    |
# | `ON DELETE SET DEFAULT`            | 子行的外键列设为默认值              |
# | `ON DELETE RESTRICT` / `NO ACTION` | 有子行引用就**拒绝删除**（默认行为） |


class SqliteSessionStore(SessionStore):
    """Zero-config local backend: one file, no server."""

    def __init__(self, db_path: str | Path):
        """创建数据库文件到db_path,并执行SQLITE_SCHEMA这个SQL语句来创建session表和message表"""
        #创建数据库文件路径属性对象
        self.db_path = Path(db_path)
        #如果数据库文件所在的目录不存在，则创建
        self.db_path.parent.mkdir(parents=True, exist_ok=True)#exist_ok代表如果目录存在，也不报错
        #打开一个数据库连接，取别名conn
        with self._db() as conn:
            #执行建表语句
            conn.executescript(SQLITE_SCHEMA)
        #两个知识点：
        # 1，“conn.executescript(SQLITE_SCHEMA)”
        # 如果执行的这个sql语句发生了错误，抛出的异常会抛给_db函数，再从_db函数中抛给上层           
        # 2，with self._db() as conn:
        ##with后面跟着的对象函数，其对象必须具备下面两个特殊方法：
        ## __enter__()   进入 with 块时调用，返回值赋给 as 后面的 x
        ## __exit__()    走出 with 块时调用（正常或异常都会调）
        ## 3，with语句
        ## 当conn.executescript(SQLITE_SCHEMA)执行完成后，走出with块，会跳转到_db函数的yield处继续执行
        ## 因为_db函数已经被修饰成了一个上下文管理器（@contextmanager装饰器+yeild关键字组成的函数）

    def _connect(self) -> sqlite3.Connection:
        """连接数据库文件，返回sqlite3.Connection连接对象"""
        #连接数据库文件，返回一个sqlite3.Connection连接对象
        conn = sqlite3.connect(self.db_path)
        #使得查询返回的row对象具备直接通过列名取元素,如row["session_id"] 
        conn.row_factory = sqlite3.Row
        return conn

    #使用@contextmanager装饰器+yeild关键字定义一个上下文管理器_db函数
    @contextmanager
    def _db(self):
        """Transaction scope that also CLOSES the connection.

        (`with sqlite3.connect()` alone only commits — it never closes, which
        keeps the file locked on Windows.)
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()
        ##知识点
        ##1，try finally
        ##通过yield conn返回连接对象给外界后，如果外界使用with拿到这个连接对象进行操作时，无论成功还是失败
        ##都会执行到这里的finally语句块，从而关闭连接
        

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    def _new_session_id(self) -> str:
        """以当前时间为基准生成新的session_id值,如果该值在session表中已存在,则加后缀再返回"""
        #以当前时间为准，生成字符串
        base = time.strftime("chat_%Y%m%d_%H%M%S")
        #打开数据库连接，取别名conn，并在抛出with块自动关闭数据库连接
        with self._db() as conn:
            #使用集合推导式创建set对象（特点：查询速度快，无序，去重）
            existing = {
                #执行sql语句，查询得到cursor对象（称为游标，代表查询的结果集），并迭代该游标对象的Row对象，取其session_id键值
                #循环迭代重复过程，得到集合对象
                r["session_id"]
                for r in conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id LIKE ?", (base + "%",)
                )
            }
        #如果sessions表中没有和base相同名字的session_id（精确匹配），则直接返回base（新的session_id值）
        if base not in existing:
            return base
        #如果sessions表有和base完全相同的session_id，则加后缀再返回
        n = 2
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"
    ##知识点
    ##1，集合推导式和列表推导式
    ##略（直接看代码，代码就是集合推导式）
    ##2，base not in existing
    ##这个匹配规则是精确匹配
    ##3，time.strftime("chat_%Y%m%d_%H%M%S")
    ##这个代码是以当前时间为基准生成字符串
    ##4, (base + "%",)
    ##这个base后面的%，代表精确匹配base，但是base后面的值就是随便匹配，如base123就满足这个条件，
    ##但是123base123就不行,这个的满足条件为“%base%”

    def create_session(self, session_id: str | None = None) -> str:
        """创建新的session,并返回该session_id"""
        #如果传入了session_id，则直接用，否则生成一个
        sid = session_id or self._new_session_id()
        now = _now()
        #打开数据库连接，插入新的session行
        with self._db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (sid, now, now),
            )
        #返回创建的session_id
        return sid
    ##知识点
    ##1，INSERT IGNORE INTO
    ## 往表里插一行；如果这行数据违反了表的任何约束（最常见的是插入列中包含的主键/唯一键重复），就放弃这次插入、
    ## 不报错、不中断程序。

    def find_session(self, fragment: str) -> str:
        """查找session,返回该sessionde session_id值,查找策略:精确匹配+模糊匹配"""
        with self._db() as conn:
            #精确匹配查找，如果找到直接返回
            exact = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?", (fragment,)
            ).fetchone()
            if exact:
                return exact["session_id"]
            #模糊匹配查找，然后按照updated_at降序排列
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id LIKE ? ORDER BY updated_at DESC",
                (f"%{fragment}%",),
            ).fetchall()
        #如果模糊匹配后查找到的元素刚好只有一个，直接返回
        if len(rows) == 1:
            return rows[0]["session_id"]
        #如果有多个或者没有，则抛出异常
        raise FileNotFoundError(
            f"找不到会话: {fragment}"
            + (f"（有 {len(rows)} 个模糊匹配，请写完整 ID）" if rows else "")
        )

    def list_sessions(self) -> list:
        """获取每个会话的信息,每个会话包含该会话所含用户消息数目"""
        with self._db() as conn:
            return conn.execute(
                """
                SELECT s.session_id, s.title, s.summary, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM messages m
                         WHERE m.session_id = s.session_id AND m.role = 'user') AS turns
                FROM sessions s
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        ##知识点：
        # SELECT s.session_id, s.title, s.summary, s.created_at, s.updated_at,
        #        (SELECT COUNT(*) FROM messages m
        #          WHERE m.session_id = s.session_id AND m.role = 'user') AS turns
        # FROM sessions s
        # ORDER BY s.updated_at DESC
        # 1，多层查询特点
        # 外层查询先执行，内层查询可以用到外层查询的查询结果集（通过取得别名引用到）。
        # 内层查询出来的结果集的字段将附加再外层结果集的已有字段中
        # 2，SELECT COUNT(*) FROM ....
        # 数查询出来的结果集有多少行
        # 3，AS
        # 给查询出来结果集的字段取别名

    def get_session(self, session_id: str):
        """查询一个session表中某个session的信息"""
        with self._db() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

    def delete_session(self, session_id: str) -> None:
        """删除某个会话session"""
        with self._db() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """往message表追加一条该会话的消息,同时更新session表中该会话的会话信息"""
        now = _now()
        with self._db() as conn:
            #往message消息表中新增一条消息
            cur = conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            #收到一条用户消息时更新会话
            #如果会话还没有标题，就用这条消息的内容当标题(截断到30字):同时刷新最后活跃时间
            if role == "user":
                conn.execute(
                    """UPDATE sessions
                          SET title = CASE WHEN title = '' THEN ? ELSE title END,
                              updated_at = ?
                        WHERE session_id = ?""",
                    (content.replace("\n", " ")[:30], now, session_id),
                )
            #如果不是用户消息，更新会话的“最后活跃时间”即可
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
            #返回最后一条消息的ID
            return cur.lastrowid
        ##知识点：
        ##1,CASE WHEN
        # CASE WHEN ... THEN ... ELSE ... END 是 SQL 里的条件表达式，逻辑和 Python 的三元表达式几乎一样：
        # 即CASE WHEN 条件 THEN 值A ELSE 值B END（条件满足，则取值A，否则取值B）

    def get_messages(self, session_id: str) -> list[dict]:
        """获取所有的消息,并转成dict字典返回"""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_message(self, message_id: int) -> None:
        """从meesage表删除某一条消息"""
        with self._db() as conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    def delete_messages_before(self, session_id: str, max_id: int) -> None:
        """从message表中删除低于某个id值的所有消息"""
        with self._db() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND id <= ?",
                (session_id, max_id),
            )

    def clear_session(self, session_id: str) -> None:
        """删除某个message表中某个会话的全部消息,
        同时更新session表中该会话的信息(清空标题和摘要,同时更新最后活跃时间)"""
        with self._db() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET summary = '', title = '', updated_at = ? WHERE session_id = ?",
                (_now(), session_id),
            )

    # ------------------------------------------------------------------
    # rolling summary
    # ------------------------------------------------------------------

    def get_summary(self, session_id: str) -> str:
        """获取某个session的summary,如果为空,则返回空串"""
        row = self.get_session(session_id)
        return row["summary"] if row else ""

    def set_summary(self, session_id: str, summary: str) -> None:
        """设置某个session的summary为具体设置的值"""
        with self._db() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ? WHERE session_id = ?",
                (summary, session_id),
            )


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def create_session_store(cfg: dict) -> SessionStore:
    """根据config配置信息,创建对应的数据存储后端对象并返回(sqlite或者postgresql)"""
    mem_cfg = cfg.get("memory", {})
    backend = mem_cfg.get("backend", "sqlite")
    if backend == "sqlite":
        return SqliteSessionStore(mem_cfg.get("db_path", "output/sessions.db"))
    if backend == "postgres":
        from postgres_store import PostgresSessionStore  # lazy: psycopg only needed here
        return PostgresSessionStore(mem_cfg.get("postgres", {}))
    raise ValueError(f"未知的 memory.backend: {backend}（可选: sqlite / postgres）")
