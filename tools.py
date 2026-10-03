"""Tool registry for the agent: JSON Schemas (the model's manual) + executor functions.

Each tool is a dict entry: {"schema": <OpenAI tools format>, "func": callable}.
The schema's description/parameter docs are what the model reads to decide
WHEN and HOW to call the tool — write them like API documentation, because
that is exactly what they are.

Safety boundaries (deliberate, production-style):
  - calculator:     eval with __builtins__ removed + a small whitelist of names
  - read_local_file: path sandbox — resolved path must stay inside the project
  - web_search:     timeout + truncated results; degrades to an error string
                    the model can react to (instead of crashing the loop)
  - python_repl is NOT provided: arbitrary code execution stays out of scope.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
READ_FILE_MAX_CHARS = 4000
SEARCH_MAX_RESULTS = 5
SEARCH_MAX_CHARS = 3000
WEATHER_MAX_CHARS = 2000

# ---------------------------------------------------------------------------
# tool implementations
# ---------------------------------------------------------------------------

def _get_weather(city: str) -> str:
    """wttr.in 天气查询——无需 API Key；任何错误都降级为字符串返回。"""
    import urllib.parse
    import urllib.request

    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1&lang=zh"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return f"天气查询失败: {type(exc).__name__}: {exc}（可确认地名拼音/英文后重试）"

    try:
        cur = data["current_condition"][0]
        today = data["weather"][0]
        area = data.get("nearest_area", [{}])[0]
        place = area.get("areaName", [{}])[0].get("value", city)
        country = area.get("country", [{}])[0].get("value", "")
    except (KeyError, IndexError) as exc:
        return f"天气数据解析失败: {exc}"

    text = (
        f"{place}{('（' + country + '）') if country else ''} 当前天气：\n"
        f"  状况：{cur['weatherDesc'][0]['value']}\n"
        f"  温度：{cur['temp_C']}°C（体感 {cur['feelsLikeC']}°C）\n"
        f"  湿度：{cur['humidity']}%　风速：{cur['windspeedKmph']} km/h {cur['winddir16Point']}\n"
        f"  今日：{today['mintempC']}°C ~ {today['maxtempC']}°C"
    )
    return text[:WEATHER_MAX_CHARS]

def _calculator(expression: str) -> str:
    """Evaluate a math expression with a whitelist instead of raw eval."""
    safe_names = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
        **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")},
    }
    try:
        return str(eval(expression, {"__builtins__": {}}, safe_names))
    except Exception as exc:
        return f"计算出错: {type(exc).__name__}: {exc}"
    ##知识点：
    ## 1，eval(expression, {"__builtins__": {}}, safe_names)
    ## eval如果不加任何限制的使用，可以做任何事情，这里第二个参数全局命名空间里的内置函数全部清空
    ## 第三个参数使用局部命名空间的批准的少数名字


def _get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def _read_local_file(path: str) -> str:
    """Read a file under the project directory (path sandbox enforced)."""
    target = (PROJECT_ROOT / path).resolve()#展开../，防止路径穿越跳出项目目录
    if not str(target).startswith(str(PROJECT_ROOT)):
        return f"拒绝读取: 路径越界（只允许读取项目目录 {PROJECT_ROOT} 内的文件）"
    if not target.is_file():
        return f"文件不存在: {path}"
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > READ_FILE_MAX_CHARS:
        text = text[:READ_FILE_MAX_CHARS] + f"\n...（已截断，共 {len(text)} 字符）"
    return text


def _web_search(query: str) -> str:
    """DuckDuckGo search — no API key required; fails soft on any error."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "web_search 不可用：未安装 ddgs（pip install ddgs）"
    try:
        results = list(DDGS(timeout=10).text(query, max_results=SEARCH_MAX_RESULTS))
    except Exception as exc:
        return f"搜索失败: {type(exc).__name__}: {exc}（可尝试换关键词重试，或改用已知信息回答）"
    if not results:
        return "没有搜到相关结果"
    text = "\n\n".join(
        f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )
    return text[:SEARCH_MAX_CHARS]


# ---------------------------------------------------------------------------
# registry: schema (what the model sees) + func (what we execute)
# ---------------------------------------------------------------------------

TOOLS = {
    "get_weather": {
        "schema": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询某个城市/地区的当前天气和今日气温范围。"
                               "当用户问到天气、气温、下雨、穿什么衣服时使用。"
                               "地名用中文、英文或拼音均可（如 '杭州'、'hangzhou'）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市或地区名，如 '杭州'、'Tokyo'"},
                    },
                    "required": ["city"],
                },
            },
        },
        "func": _get_weather,
    },
    "calculator": {
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算一个数学表达式的结果。当需要进行任何数学计算时使用此工具。"
                               "输入必须是合法的 Python 数学表达式字符串，例如 '300 * 0.25'。"
                               "百分数请先转换为小数（25% 写成 0.25）；支持 math 模块函数如 sqrt、pi。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式，如 '300 * 0.25'"},
                    },
                    "required": ["expression"],
                },
            },
        },
        "func": _calculator,
    },
    "get_current_time": {
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "返回当前的日期、时间和星期。当问题涉及今天/现在/当前时间时使用。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        "func": _get_current_time,
    },
    "read_local_file": {
        "schema": {
            "type": "function",
            "function": {
                "name": "read_local_file",
                "description": "读取项目目录内的本地文件内容（如 config.json、README.md）。"
                               "路径相对于项目根目录，超长内容会被截断。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对项目根目录的文件路径，如 'config.json'"},
                    },
                    "required": ["path"],
                },
            },
        },
        "func": _read_local_file,
    },
    "web_search": {
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "联网搜索最新信息。当问题涉及你不知道的最新事件、价格、新闻时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                    },
                    "required": ["query"],
                },
            },
        },
        "func": _web_search,
    },
}


def get_tool_schemas(enabled_tools: list[str]) -> list[dict]:
    """Schemas of enabled tools, in config order (unknown names are skipped)."""
    return [TOOLS[name]["schema"] for name in enabled_tools if name in TOOLS]


def run_tool(name: str, arguments_json: str) -> str:
    """Execute one tool call. Never raises — errors go back to the model as text."""
    #raise BaseException
    tool = TOOLS.get(name)
    if not tool:
        return f"未知工具: {name}"
    try:
        kwargs = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return f"工具参数 JSON 解析失败: {exc}"
    try:
        result = str(tool["func"](**kwargs))
    except TypeError as exc:
        return f"工具参数错误: {exc}"
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return f"工具执行失败: {type(exc).__name__}: {exc}"
    logger.info("Tool call: %s(%s) -> %s", name, kwargs, result[:200])
    return result
