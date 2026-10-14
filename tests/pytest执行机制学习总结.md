# pytest 执行机制学习总结

基于 modelclaw 项目 M4b 阶段全部测试模块（conftest.py、test_context_engine.py、test_rag_store.py、test_session_store.py、test_storage.py、test_tools.py、test_eval_regression.py）的学习笔记。

## 一、测试模块撰写的方法论

### 核心流程

编写一个测试模块，按以下顺序思考：

1. **先确定测什么功能**：不是"测某个函数"，而是"测某个对外承诺的行为"。例如 test_rag_store.py 测的是"向量库写入能查到、最近邻检索正确、删除和清空生效"这套契约行为。
2. **确定被测对象依赖哪些函数和接口**：这些依赖就是搭建测试环境所需的"原材料"。
3. **设计输入数据**：预先想好传什么数据进去。
4. **定义预期输出**：断言返回什么结果才算功能正常。

### 一个测试函数里通常有三类角色

以 test_context_engine.py 的 `test_build_context_triggers_rolling_summary` 为例：

| 角色 | 本例中的成员 | 特点 |
|------|------------|------|
| 被测对象（SUT） | `build_context` | 唯一"真"的，测试失败就怪它 |
| 测试替身（double） | `FakeStore`、被 monkeypatch 替换的 `complete` | 可控、可预测，只为搭环境 |
| 测试数据工厂 | `_msgs(4)` | 批量制造标准结构的输入 |

**判断标准**：把所有替身的行为都固定下来之后，测试的成败只取决于被测对象自己的逻辑。这样任何一条 assert 失败，问题必然出在被测对象内部，而不是环境。

## 二、pytest 的运行机制

pytest 命令启动后，整个过程分四个阶段：

### 阶段 1：加载 conftest.py（任何测试运行之前）

- pytest 先找到项目里所有 `conftest.py` 并**执行整个文件**。
- conftest.py 的作用是集中定义公共 fixture(作为后续测试函数的参数注入)

### 阶段 2：收集测试（collection）

- 扫描所有 `test_*.py` 文件，找出所有 `test_` 开头的函数，登记成"待执行清单"。

### 阶段 3：逐个执行测试

对清单里的每个测试函数：

1. **fixture 搭建（setup）**：读函数签名里的参数名 → 去 fixture 注册表查找 → 调用对应 fixture 函数 → 把返回值注入参数。
   - `test_xxx(monkeypatch)` → pytest 创建 MonkeyPatch 实例传入
   - `test_xxx(tmp_path)` → pytest 提供本次测试专属的临时目录
   - `test_xxx(sqlite_session_store)` → 执行 conftest.py 里定义的 fixture，返回临时数据库实例
   - 参数名在注册表里找不到 → 报错 `fixture 'xxx' not found`
2. **执行测试函数体**：一条条 assert 跑，任何一条失败即停止并标记失败。
3. **fixture 拆卸（teardown）**：测试结束后自动回收——monkeypatch 的替换被撤销、临时目录被删除，一切恢复原样。需要自定义拆卸时用 `yield` 写在 fixture 里（yield 前是 setup，yield 后是 teardown）；conftest.py 里几个 fixture 没有 yield，因为清理由 pytest 内部代劳。

### 阶段 4：汇总报告

输出 `5 passed, 1 skipped` 之类的总结。

### 参数注入的本质

通过fixture修饰的函数，函数的执行结果将注入给后续测试函数中同名参数的实际值

## 三、MonkeyPatch 的作用和机制

### 作用

在**单个测试运行期间**，把某个模块（或类）里的函数/属性替换成测试指定的假实现，测试结束后**自动恢复原样**。用途：截断对外部依赖（LLM API、系统函数等）的调用，让测试只验证被测对象自己的逻辑。

### 机制

- monkeypatch 是 pytest 的**内置 fixture**，不是普通 Python 函数。测试函数声明 `monkeypatch` 参数，pytest 自动创建一个 `pytest.MonkeyPatch` 类的实例传入。
- 代码层面没有任何魔法：它就是普通参数 + 对象方法调用。`monkeypatch.setattr(目标, "名字", 替换值)` 等价于临时执行 `目标.名字 = 替换值`。
- **作用域仅限当前测试**：出了这个测试函数替换就失效，不污染其他测试（这是它和手动 `setattr` 再手动改回来的本质区别——自动撤销，再稳的实现也难免会忘恢复）。

### 项目实例（test_context_engine.py）

```python
def test_build_context_triggers_rolling_summary(monkeypatch):
    # 把 context_engine 模块里的 complete 换成返回固定字符串的假实现，
    # build_context 内部调用 complete 时，实际调到的就是这个 lambda
    monkeypatch.setattr(context_engine, "complete",
                        lambda client, cfg, messages: "滚动摘要内容")
    ...
```

### 最简示例

```python
import pytest
import my_module

def test_with_patch(monkeypatch):
    monkeypatch.setattr(my_module, "get_current_time", lambda: "2026-01-01")
    assert my_module.greet() == "你好，现在是 2026-01-01"   # greet 内部调用了 get_current_time
# 测试结束后 my_module.get_current_time 自动恢复成原实现
```

要点：`setattr` 的第一个参数是**模块（或类）对象**，第二个参数是目标名字的**字符串**，第三个参数是替换后的新值（可以是 lambda、普通函数或任意对象）。

## 四、两种跳过标记的区分和作用

两者底层是同一机制（往测试身上贴标记、收集时评估），区别只在"贴一个还是贴全部"。

### 写法与作用范围

| 类型 | 写法 | 生效范围 |
|------|------|---------|
| 函数/类级 | `@pytest.mark.skipif(...)` 或 `@pg_required`，贴在 `def`/`class` 正上方 | 只管紧挨着的那个函数或类 |
| 模块级 | `pytestmark = pytest.mark.skipif(...)`（赋值语句，变量名固定） | 文件里所有测试 |

### 函数/类级：装饰器方式

```python
# conftest.py 中定义一次
pg_required = pytest.mark.skipif(not _pg_available(), reason="本地 PostgreSQL 不可用")

# 其他测试文件中使用
@pg_required                        # 可以贴在函数上
class TestPgVectorStore:            # 也可以贴在类上（整个类的测试一起跳过）
    def test_contract(self, pg_cfg):
        ...
```

- 装饰器本质：`test_xxx = pg_required(test_xxx)`，把标记贴到那一个函数对象身上。
- 类装饰器的效果：类里每个测试函数都被贴上标记。

### 模块级：pytestmark 变量

```python
# test_eval_regression.py
pytestmark = pytest.mark.skipif(
    os.environ.get("MODELCLAW_RUN_LLM_TESTS") != "1",
    reason="LLM 回归测试默认跳过（消耗 API 额度），用 MODELCLAW_RUN_LLM_TESTS=1 启用",
)
```

- `pytestmark` 是 pytest 约定的**特殊变量名**，收集文件时 pytest 专门找这个变量，把它身上的标记复制给本文件每个测试函数。换成别的名字 pytest 认不出来。
- 适合"整个文件都需要同一开关"的场景，不用逐个函数贴装饰器。
- 想叠加多个标记时写成列表：`pytestmark = [pytest.mark.skipif(...), pytest.mark.slow]`。

### 生效时机

1. **收集时（阶段 2）**：pytest 评估条件（`_pg_available()` 的结果、环境变量），条件为真就盖上"跳过"章并记录理由。
2. **执行时（阶段 3）**：被盖章的测试**不执行函数体**，直接记为 skipped，连 fixture 都不搭建。

一个细节：探路类检查发生在 conftest.py 加载阶段（pytest 启动时），结果在启动那一刻就定死了——即使运行中途启动了数据库，本次运行也感知不到。

## 判断口诀速查

- 看到 `pytestmark = pytest.mark....` → 整个文件生效
- 看到 `@...mark...` 在 `def`/`class` 上方 → 只管紧挨的一个
- 测试函数参数没人传值 → pytest 按名字从 fixture 注册表注入
- `monkeypatch.setattr(模块, "名字", 新值)` → 只在本测试内替换，结束自动还原
