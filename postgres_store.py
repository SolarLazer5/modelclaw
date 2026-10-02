"""PostgreSQL backend for session storage (psycopg 3).

Same contract as SqliteSessionStore (see session_store.SessionStore); only the
SQL dialect and connection handling differ:

    SQLite                          PostgreSQL
    ? placeholders                  %s placeholders
    INTEGER ... AUTOINCREMENT       BIGINT GENERATED ALWAYS AS IDENTITY
    INSERT OR IGNORE                INSERT ... ON CONFLICT DO NOTHING
    sqlite3.Row                     psycopg dict_row (real dicts)
    file per database               database created via the maintenance db
    `with conn:` only commits       `with conn:` commits AND closes (psycopg 3)

Connection parameters come from config `memory.postgres`; the password comes
ONLY from the MODELCLAW_PG_PASSWORD env var (.env) — never from config.json.
"""

import os
import time

import psycopg
from psycopg.rows import dict_row

from session_store import SessionStore

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _now() -> str:
    """返回格式化后的当前时间字符串"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


class PostgresSessionStore(SessionStore):
    """Server backend: for shared/multi-user or service deployments."""

    def __init__(self, pg_cfg: dict):
        """初始化pg数据库后端"""
        #讲传入pg config中的数据库后端配置信息保存到属性中
        self.host = pg_cfg.get("host", "127.0.0.1")                 #pg数据库后端地址
        self.port = int(pg_cfg.get("port", 5432))                   #pg数据库后端端口号
        self.user = pg_cfg.get("user", "postgres")                  #pg用户名
        self.dbname = pg_cfg.get("database", "modelclaw")           #要操作的pg数据库名字
        self.password = os.environ.get("MODELCLAW_PG_PASSWORD", "") #通过环境变量获取pg数据库密码

        #如果密码为空，则抛出异常
        if not self.password:
            raise RuntimeError(
                "未找到 MODELCLAW_PG_PASSWORD，请在 .env 文件中配置 PostgreSQL 密码"
            )
        
        self._ensure_database()
        self._init_tables()

    def _connect(self, dbname: str | None = None, autocommit: bool = False) -> psycopg.Connection:
        """获取pg连接对象"""
        return psycopg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=dbname or self.dbname,
            autocommit=autocommit,  #每个sql语句是否直接执行并自动提交
            row_factory=dict_row,   #使得后续查询得到的游标结果集中的每一行对象支持列名访问
        )
    ##知识点
    ## 1.autocommit的作用和pg背景
    ## 每个sql语句默认在pg中是一个事务，多个sql语句就是多个事务，最后关闭pg连接时统一提交，而这个autocommit控制
    ## 每条sql语句是否各自独立执行、执行完立即生效，不挂在任何事务里。
    ## 2.row_factory=dict_row的作用
    ## 等价于sqlite3端的conn.row_factory = sqlite3.Row，使得后续查询得到的游标结果集中的每一行对象支持列名访问


    def _ensure_database(self) -> None:
        """
        确保数据库存在

        ### 执行过程
        1. 获取数据库连接对象，执行数据库查询sql语句
        2. 如果查询的数据库不存在，则执行数据库创建sql语句

        ### 注意
        获取数据库连接对象时，事务设置为了自动提交，因为pg后端不支持数据库操作相关sql语句事物提交模型
        """
        #打开数据库连接，设置自动提交模式
        with self._connect(dbname="postgres", autocommit=True) as conn:
            #执行查询数据库sql语句
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (self.dbname,)
            ).fetchone()
            #如果目标数据库不存在，执行建库语句
            if not exists:
                # Identifiers can't be parameterized; quote instead.
                conn.execute(f'CREATE DATABASE "{self.dbname}"')

    def _init_tables(self) -> None:
        """
        确保数据库中具备所需要的所有表
        """
        #获取pg连接对象，执行建表语句
        with self._connect() as conn:
            conn.execute(PG_SCHEMA)

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    def _new_session_id(self) -> str:
        """创建不重名的新的session_id值"""
        base = time.strftime("chat_%Y%m%d_%H%M%S")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id LIKE %s", (base + "%",)
            ).fetchall()
        existing = {r["session_id"] for r in rows}
        if base not in existing:
            return base
        n = 2
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"

    def create_session(self, session_id: str | None = None) -> str:
        """在数据库sessions表中创建新的session，并返回这个session的session_id值"""
        sid = session_id or self._new_session_id()
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, created_at, updated_at)
                   VALUES (%s, %s, %s) ON CONFLICT (session_id) DO NOTHING""",
                (sid, now, now),
            )
        return sid

    def find_session(self, fragment: str) -> str:
        """从sessions表中查询session_id"""
        with self._connect() as conn:
            exact = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = %s", (fragment,)
            ).fetchone()
            if exact:
                return exact["session_id"]
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id LIKE %s ORDER BY updated_at DESC",
                (f"%{fragment}%",),
            ).fetchall()
        if len(rows) == 1:
            return rows[0]["session_id"]
        raise FileNotFoundError(
            f"找不到会话: {fragment}"
            + (f"（有 {len(rows)} 个模糊匹配，请写完整 ID）" if rows else "")
        )

    def list_sessions(self) -> list:
        """列出包含用户消息轮数的所有sessions信息"""
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT s.session_id, s.title, s.summary, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM messages m
                         WHERE m.session_id = s.session_id AND m.role = 'user') AS turns
                FROM sessions s
                ORDER BY s.updated_at DESC
                """
            ).fetchall()

    def get_session(self, session_id: str):
        """获取sessions表中某个session的详细信息"""
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE session_id = %s", (session_id,)
            ).fetchone()

    def delete_session(self, session_id: str) -> None:
        """删除某个sessions表的某个会话(同步删除messages表中所有该会话的消息)"""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """
        往meesages表中追加一条新的message消息，同时如果该消息对于的会话在sessions表中还没有title\n
        则取该消息的前30个字作为title值
        """
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                """INSERT INTO messages(session_id, role, content, created_at)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (session_id, role, content, now),
            ).fetchone()
            if role == "user":
                conn.execute(
                    """UPDATE sessions
                          SET title = CASE WHEN title = '' THEN %s ELSE title END,
                              updated_at = %s
                        WHERE session_id = %s""",
                    (content.replace("\n", " ")[:30], now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = %s WHERE session_id = %s",
                    (now, session_id),
                )
        return row["id"]

    def get_messages(self, session_id: str) -> list[dict]:
        """获取某个会话的所有message，同时按照id降序排列"""
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, role, content FROM messages WHERE session_id = %s ORDER BY id",
                (session_id,),
            ).fetchall()

    def delete_message(self, message_id: int) -> None:
        """删除某个message消息"""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE id = %s", (message_id,))

    def delete_messages_before(self, session_id: str, max_id: int) -> None:
        """删除某个会话小于等于id值的所有消息"""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = %s AND id <= %s",
                (session_id, max_id),
            )

    def clear_session(self, session_id: str) -> None:
        """清空某个会话的所有消息，并重置标题"""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            conn.execute(
                "UPDATE sessions SET summary = '', title = '', updated_at = %s WHERE session_id = %s",
                (_now(), session_id),
            )

    # ------------------------------------------------------------------
    # rolling summary
    # ------------------------------------------------------------------

    def get_summary(self, session_id: str) -> str:
        """获取某个session的summary摘要"""
        row = self.get_session(session_id)
        return row["summary"] if row else ""

    def set_summary(self, session_id: str, summary: str) -> None:
        """设置某个session的summary摘要"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary = %s WHERE session_id = %s",
                (summary, session_id),
            )
