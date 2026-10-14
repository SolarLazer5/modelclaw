# LangGraph-Checkpoint四表拆解

## 3. checkpoints 表（已收官）

### 3.0.1. SQL语句理解

`TABLESPACE` 是 PostgreSQL 中用来指定数据库对象**物理存储位置**的机制。
它对应文件系统上的一个目录。

sql

```
TABLESPACE pg_default;
```

> 把前面创建的表或索引，存放到名为 `pg_default` 的表空间中。

更准确地说：

> 表空间是 PostgreSQL 对“存储目录”的逻辑命名；
> `TABLESPACE pg_default` 是指定对象使用默认表空间对应的目录来存放数据文件。

### 3.0.2. pg_default是什么

`pg_default` 是 PostgreSQL 内置的默认表空间。

特点：

- PostgreSQL 初始化数据库集群时自动创建；
- 通常对应数据目录下的 `base/` 区域；
- 另一个内置表空间是 `pg_global`，用于存放整个集群共享的系统对象；
- 如果不显式指定 `TABLESPACE`，默认通常也是使用 `pg_default`，除非数据库或会话设置了 `default_tablespace`。

所以：

```
CREATE TABLE ... TABLESPACE pg_default;
```

在默认配置下通常等价于：

```
CREATE TABLE ...;
```

很多工具，比如 `pg_dump`、pgAdmin，会显式写出 `TABLESPACE pg_default`，是为了保留对象的原始存储位置信息。

### 3.1 一句话记住

**一行 = 一步的状态照片。** 图每完成一个节点存一行，行与行用 `parent_checkpoint_id` 首尾相接（第一张的 parent 为 NULL）。

### 3.2 字段

| 字段 | 意思 | 作用 |
|---|---|---|
| thread_id | 会话编号（如 lg_20261013_110026） | 对应我们设计里的 session_id，按它查会话全部快照 |
| checkpoint_ns | 命名空间，默认空串 | 单 agent 用不到，写查询时带上 `= ''` 即可 |
| checkpoint_id | 本张快照的唯一编号 | 每推进一步生成一个新的 |
| parent_checkpoint_id | 上一张快照的编号，可空 | 串成快照链：断点续跑、中断恢复、回溯分叉全靠它 |
| type | 记录类别 | 跑 `SELECT DISTINCT type FROM checkpoints;` 眼见为实 |
| checkpoint(jsonb) | 状态本体（v4 格式） | 只存版本元数据，真实内容在 checkpoint_blobs 表 |
| metadata(jsonb) | 这步怎么来的 | `step`=第几步，`source`=写入时机（loop/input/update），`parents`=父图信息（我们用不到） |

checkpoint jsonb 内部三个关键字段：

| 内部字段 | 意思 |
|---|---|
| channel_versions | 本步后各通道的版本号（如 messages 处于哪一版）——**指向 blobs 表的指针** |
| versions_seen | 各节点运行时看到的版本——判断恢复时哪些节点要重跑 |
| updated_channels | 本步哪些通道变了（messages 在列 = 这一步追加了消息） |

### 3.3 主键与索引

- 主键 `(thread_id, checkpoint_ns, checkpoint_id)`：三列组合唯一 + 非空；PG 自动建一个三列 UNIQUE 索引（副产品，用于查重）。
- 单列索引 `(thread_id)`：独立的普通索引（不唯一），为"按会话拉全部快照"这条最热查询服务。
- 记住：**约束管数据对不对，索引管查得快不快**，两者互不依赖。

---

## 4. checkpoint_blobs 表（已收官）

### 4.1 一句话记住

**一行 = 某个通道某个版本的真实内容。** checkpoints 存指针（版本号），这张表存实物，靠 channel + version 对接。

### 4.2 字段

| 字段 | 意思 | 作用 |
|---|---|---|
| thread_id | 会话编号 | 同 checkpoints 表，关联同一会话 |
| checkpoint_ns | 命名空间，默认空串 | 同上 |
| channel | 通道名 | `messages`（主角）、`__start__`（本轮输入）、`__pregel_tasks`（框架任务队列） |
| version | 通道版本号 | 格式同 checkpoints.channel_versions，**与那张表对接的键** |
| type | 序列化格式 | 实际值 `msgpack`（二进制版 JSON，更小更快），所以 blob 显示为 binary data |
| blob(bytea) | 内容本体 | msgpack 打包的通道值；messages 解开就是消息列表（对话文本都在里面） |

### 4.3 主键与索引

- 主键 `(thread_id, checkpoint_ns, channel, version)`：含 version 不含 checkpoint_id → 值没变不写新行（去重），旧版本永不修改（回溯保障）。
- 单列索引 `(thread_id)`：恢复会话时一次捞回全部内容，内存拼装成 invoke 返回的 result。

---

## 四表关系图

```
checkpoint_writes（明细：每步内每个任务写了什么，先落这里）
        │ checkpoint_id 挂快照名下
        ▼
   checkpoints ──parent_checkpoint_id──▶ 上一张快照（链）
        │ channel_versions（指针：哪个通道在第几版）
        ▼
checkpoint_blobs（实物：channel + version ─▶ msgpack 内容）

checkpoint_migrations ── 表结构版本号账本（setup 时对照升级）
```

读状态的完整链路：`checkpoints` 拿版本号 → `checkpoint_blobs` 取内容 → 内存拼装 → invoke 返回 `result["messages"]`。

## 5. checkpoint_writes 表（已收官）

### 5.1 一句话记住

**一行 = 一步之内、某个任务对某个通道的一次写入明细。** 快照提交前先记流水，崩溃后据此知道进行到哪一步、从哪个任务重放。

### 5.2 字段

| 字段 | 意思 | 作用 |
|---|---|---|
| thread_id | 会话编号 | 同上 |
| checkpoint_ns | 命名空间，默认空串 | 同上 |
| checkpoint_id | 属于哪张快照 | 与 checkpoints 表的 checkpoint_id 对应——明细挂快照名下（对比 blobs 挂版本名下） |
| task_id | 产生写入的任务编号 | 一步内可有多个任务（如并行工具调用），按它区分 |
| idx | 同一任务内的写入序号 | 一个任务写多个通道时逐条编号 |
| channel | 写入哪个通道 | messages、branch:to:agent、__pregel_tasks 等 |
| type | 序列化格式 | `msgpack` 或 `null`（值为空的写，如 branch 通道清空） |
| blob | 写入的内容 | msgpack 二进制 |
| task_path | 写入来源路径 | 标明哪个节点产生的（实际值如 `~__pregel_pull, agent`、`~__pregel_pull, __start__`） |

### 5.3 主键与索引

- 主键 `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`：精确定位"某快照的某任务的第几条写入"。
- 单列索引 `(thread_id)`：按会话捞明细。

## 6. checkpoint_migrations 表（已收官）

### 6.1 一句话记住

**一行 = 表结构的一个版本号。** 记录存档表结构演进到第几版。

### 6.2 字段与实际数据

只有一个字段 `v`（整数，主键）。我查到的实际数据是 0~9 共十行——表示这套表结构已经演进到第 9 版。`checkpointer.setup()` 运行时对照这个版本号，旧了就执行迁移脚本升级到最新，所以 LangGraph 升级后表结构能自动跟上，不用手动改库。
