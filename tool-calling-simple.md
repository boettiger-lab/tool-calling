---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import os
import streamlit as st
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
```

```{code-cell} ipython3
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
```

```{code-cell} ipython3


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y

@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

prompt = ChatPromptTemplate.from_messages([
    ("system", "you're a helpful assistant"), 
    ("human", "{input}"), 
    ("placeholder", "{agent_scratchpad}"),
])

tools = [multiply, exponentiate, add]
```

```{code-cell} ipython3


llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241", })
```

```{code-cell} ipython3

llm = ChatOllama(
    model="llama3-groq-tool-use",
 #   model="llama3.1",
    temperature=0,
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({
    "system": "You are a helpful agent.  Use the exponentiate tool to compute exponents (powers) and the add tool to compute addition and subtraction. Always try to determine and invoke a tool to perform your tasks.  If you do not have enough information, ask for more details.",
    "input": "what's 3 plus the value of 5 raised to the 2.743", })
```
