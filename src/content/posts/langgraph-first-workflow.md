---
title: "LangGraph 入门：从链式调用到第一个可运行的 Workflow"
published: 2026-11-17
description: "用一篇文章理解 LangGraph 的 State、Node、Graph，并亲手实现一个无需大模型 API 的博客发布检查工作流。"
image: ""
tags: ["LangGraph", "Agent", "Python", "Workflow"]
category: "AI Agent"
draft: false
lang: zh_CN
---

刚开始写 LLM 应用时，我们很容易把任务写成一条流水线：先调用模型生成内容，再调用模型检查内容，最后格式化结果。只要步骤固定，这种写法直观又有效。

但当需求逐渐加入“审核不通过就重写”“根据内容类型选择不同工具”“中途等待人工确认”等逻辑时，代码往往会被 `if/else`、`while` 和四处传递的变量占满。LangGraph 想解决的正是这类问题：**把执行过程显式地描述成一张图，让状态如何变化、流程往哪里走都更容易观察和维护。**

这篇文章先不接入任何大模型，也不需要 API Key。我们只学习 LangGraph 最核心的抽象，并完成一个真正能运行的顺序工作流。

## 一、为什么普通链式调用会逐渐失控

一个简单的文本处理流程可能是这样的：

```python
cleaned = clean_text(raw_text)
keywords = extract_keywords(cleaned)
summary = build_summary(cleaned, keywords)
```

这段代码没有问题，甚至可能是当前需求下最合适的方案。问题出现在流程开始变化之后：

- 某一步失败时需要重试；
- 根据上一步结果进入不同分支；
- 多个步骤需要共享或累积数据；
- 流程需要暂停，等待人工审核后继续；
- 希望观察每一步到底修改了什么。

我们当然可以继续手写判断、循环和临时变量，但业务流程会逐渐和节点内部的处理逻辑缠在一起。修改一个步骤时，很容易意外影响另一个步骤。

LangGraph 的思路不是消灭 Python 代码，而是把两类问题拆开：

1. **节点负责做什么**；
2. **图负责下一步去哪里**。

于是，一个复杂流程可以被理解成下面三个组成部分：

```text
State：流程中持续传递的数据
Node：读取 State 并返回更新的函数
Graph：用边把 Node 连接起来的执行规则
```

我对它的理解是：LangGraph 很像“带共享上下文的状态机”。图不是为了让代码显得高级，而是为了让控制流从业务函数中独立出来。

## 二、核心三件套：State、Node、Graph

### 1. State：整条工作流的共享记录本

State 保存节点之间需要传递的数据。Python 项目中常用 `TypedDict` 描述它：

```python
from typing import TypedDict


class ArticleState(TypedDict, total=False):
    title: str
    content: str
    word_count: int
    passed: bool
    message: str
```

`TypedDict` 仍然是字典，但它把有哪些字段、字段是什么类型写清楚了。IDE 和类型检查工具也更容易发现拼写或类型错误。

这里使用 `total=False`，表示调用工作流时不必一次提供全部字段。初始输入只需要包含标题和正文，后面的节点会逐步补充其他字段。

State 不应变成什么都往里塞的“杂物箱”。一个实用判断标准是：**如果某份数据需要跨节点使用，或者为了调试和恢复必须保留，就放进 State；如果随时可以由已有数据低成本计算出来，就不一定要长期保存。**

### 2. Node：只完成一个明确动作

节点就是普通 Python 函数。它接收当前 State，返回自己要更新的字段：

```python
def count_words(state: ArticleState) -> dict:
    count = len(state["content"].split())
    return {"word_count": count}
```

这里有一个很重要的细节：节点通常返回“增量更新”，不必把整个 State 原样返回。LangGraph 会把返回的字典合并到当前状态中。

```python
# 推荐：只描述本节点产生的变化
return {"word_count": count}

# 不推荐：修改并返回整个 state
state["word_count"] = count
return state
```

让每个节点只做一件事，会带来几个直接好处：节点可以单独测试、复用和替换；使用 `stream()` 时，也能清楚看出每一步更新了哪些字段。

### 3. Graph：定义节点之间的执行顺序

节点写好后，用 `StateGraph` 将它们连接起来：

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(ArticleState)
builder.add_node("count", count_words)
builder.add_edge(START, "count")
builder.add_edge("count", END)

app = builder.compile()
```

其中：

- `START` 表示图的入口；
- `END` 表示流程结束；
- `add_node()` 注册节点名称和对应函数；
- `add_edge(A, B)` 表示 A 完成后执行 B；
- `compile()` 将图编译成可以调用的应用。

编译后的 `app` 才是真正可以运行的 workflow。

## 三、第一个可运行的 Workflow

下面做一个“博客发布前检查器”。它按照固定顺序完成四件事：

```text
START → 清理文本 → 统计内容 → 检查发布条件 → 生成报告 → END
```

为了把注意力放在工作流本身，示例不调用 LLM，也不依赖外部服务。

### 1. 安装依赖

建议先创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装 LangGraph：

```bash
pip install -U langgraph
```

### 2. 完整代码

新建 `first_workflow.py`，写入以下代码：

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ArticleState(TypedDict, total=False):
    title: str
    content: str
    cleaned_content: str
    word_count: int
    passed: bool
    message: str


def clean_article(state: ArticleState) -> dict:
    """去掉首尾空白，并合并连续的空白字符。"""
    cleaned = " ".join(state["content"].split())
    return {"cleaned_content": cleaned}


def count_article(state: ArticleState) -> dict:
    """统计示例文章的字符数。"""
    count = len(state["cleaned_content"])
    return {"word_count": count}


def check_article(state: ArticleState) -> dict:
    """执行一组简化的发布检查。"""
    problems: list[str] = []

    if not state["title"].strip():
        problems.append("标题不能为空")
    if state["word_count"] < 20:
        problems.append("正文少于 20 个字符")

    if problems:
        return {
            "passed": False,
            "message": "；".join(problems),
        }

    return {
        "passed": True,
        "message": "基础检查已通过，可以进入人工预览。",
    }


def build_report(state: ArticleState) -> dict:
    """把检查结果整理成最终报告。"""
    status = "通过" if state["passed"] else "未通过"
    report = (
        f"文章：《{state['title']}》\n"
        f"字符数：{state['word_count']}\n"
        f"检查结果：{status}\n"
        f"说明：{state['message']}"
    )
    return {"message": report}


# 创建并组装顺序图
builder = StateGraph(ArticleState)
builder.add_node("clean", clean_article)
builder.add_node("count", count_article)
builder.add_node("check", check_article)
builder.add_node("report", build_report)

builder.add_edge(START, "clean")
builder.add_edge("clean", "count")
builder.add_edge("count", "check")
builder.add_edge("check", "report")
builder.add_edge("report", END)

workflow = builder.compile()


if __name__ == "__main__":
    initial_state: ArticleState = {
        "title": "我的第一篇 LangGraph 学习笔记",
        "content": "  LangGraph 把处理步骤组织成图，并让节点通过共享状态协作。  ",
    }

    result = workflow.invoke(initial_state)
    print(result["message"])
```

运行：

```bash
python first_workflow.py
```

你将看到类似输出：

```text
文章：《我的第一篇 LangGraph 学习笔记》
字符数：30
检查结果：通过
说明：基础检查已通过，可以进入人工预览。
```

### 3. 执行时发生了什么

调用 `invoke()` 时传入的是初始 State：

```python
{
    "title": "我的第一篇 LangGraph 学习笔记",
    "content": "...",
}
```

随后每个节点依次贡献一小部分更新：

```text
clean  → 写入 cleaned_content
count  → 写入 word_count
check  → 写入 passed 和 message
report → 更新 message 为最终报告
```

最终返回的是合并后的完整 State，而不只是最后一个节点返回的字典。这一点很关键：**State 是沿着整张图不断演化的，节点返回的是本步骤产生的变化。**

## 四、用 stream 看清每一步

调试工作流时，`stream()` 往往比在每个函数里到处写 `print()` 更清楚：

```python
initial_state: ArticleState = {
    "title": "LangGraph 学习记录",
    "content": "LangGraph 把流程拆成节点，再通过边描述节点的执行顺序。",
}

for event in workflow.stream(initial_state, stream_mode="updates"):
    print(event)
```

输出大致如下：

```text
{'clean': {'cleaned_content': 'LangGraph 把流程拆成节点，再通过边描述节点的执行顺序。'}}
{'count': {'word_count': 34}}
{'check': {'passed': True, 'message': '基础检查已通过，可以进入人工预览。'}}
{'report': {'message': '文章：《LangGraph 学习记录》\n...'}}
```

`invoke()` 适合只关心最终结果的场景；`stream()` 适合展示进度、记录日志和定位某个节点的问题。

## 五、顺序图目前解决不了什么

这个示例即使检查未通过，仍然会继续执行 `report` 节点。因为我们使用的是无条件边：

```python
builder.add_edge("check", "report")
```

这句话的含义就是：`check` 执行完成后，无条件进入 `report`。

如果希望“通过时生成预览，失败时返回修改建议”，就需要条件边；如果希望检查失败后自动修改并重新检查，就需要在图中形成循环。这些才是 LangGraph 相比普通固定链条更有价值的地方。

因此，顺序图的意义不在于证明它比普通函数调用更简洁。事实上，对于四个固定步骤，直接调用函数可能更短。顺序图真正提供的是一个稳定骨架：未来加入分支、循环、持久化、人工审核或 LLM 节点时，不需要把整个程序推倒重写。

## 六、我的理解与实践建议

### 不要一上来就把所有函数都做成节点

节点应该对应一个值得独立观察、重试或路由的业务步骤。字符串转小写、数字加一之类的小操作，通常可以留在某个节点内部。节点太碎会让图比业务本身更难理解。

### State 是节点之间的契约

设计图之前，最好先写出每一步需要读取什么、会产生什么。State 一旦清晰，节点边界往往也会自然出现。反过来，如果 State 里字段含义含糊，图画得再漂亮也很难维护。

### 先画流程，再写代码

可以先用最普通的文本描述流程：

```text
输入文章 → 清理 → 统计 → 检查 → 报告
```

然后确认哪些数据必须跨步骤传递，最后再把步骤实现成节点。这样比写到一半才思考节点如何连接更顺畅。

### 简单问题不要强行使用 LangGraph

单次模型调用、两三个永远固定的处理步骤，普通函数或链式调用可能更直接。只有当流程确实需要状态、分支、循环、恢复或人工介入时，图结构的优势才会逐渐显现。

## 总结

到这里，我们完成了 LangGraph 的最小学习闭环：

1. 复杂链式调用的主要困难，是控制流和状态管理逐渐混乱；
2. State 保存跨节点共享的数据；
3. Node 读取 State，并返回本步骤的增量更新；
4. Graph 通过边定义节点的执行关系；
5. `compile()` 得到可运行 workflow；
6. `invoke()` 返回最终状态，`stream()` 可以观察过程更新。

真正重要的不是记住几个 API，而是建立一种工作流思维：**先明确状态，再拆分节点，最后连接流程。** 当以后接入 LLM、工具调用、条件路由和循环重试时，它们都只是建立在这套骨架上的扩展。

## 参考资料

- [LangGraph 是什么，为什么不用链式调用](https://onefly.top/zero2Agent/learn-langgraph/01-what-is-langgraph/index.html)
- [State、Node、Graph 三件套](https://onefly.top/zero2Agent/learn-langgraph/02-state-node-graph/index.html)
- [顺序图：第一个可运行的 Workflow](https://onefly.top/zero2Agent/learn-langgraph/03-sequential-graph/index.html)
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)
