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
import leafmap.maplibregl as leafmap
import numpy as np
from matplotlib import cm
import pandas as pd
import ibis
from ibis import _
```

```{code-cell} ipython3
# h3_parquet = "https://data.source.coop/cboettig/gbif/gbif_us_h3.parquet"
# h3_parquet = "hf://datasets/boettiger-lab/gbif/usa_h3/*/*.parquet"
# h3_parquet = "hf://datasets/boettiger-lab/gbif/gbif_ca_h3.parquet"
# h3_parquet = "https://data.source.coop/cboettig/gbif/gbif_ca.geoparquet"
# h3_parquet = "https://huggingface.co/datasets/boettiger-lab/gbif/resolve/main/gbif_ca.geoparquet"
h3_parquet = "/home/rstudio/huggingface/spaces/gbif/gbif_ca.geoparquet"

con = ibis.duckdb.connect(extensions=["spatial", "httpfs"])
gbif_h3 = con.read_parquet(h3_parquet, "gbif_h3")
```

```{code-cell} ipython3
def get_h3point_df(_df, resolution: float) -> pd.DataFrame:
    column = "h" + str(resolution)
    df = (_df
            .rename(hex = column)
            .group_by(_.hex)
            .agg(n = _.count())
       #     .mutate(wkt =  h3_cell_to_boundary_wkt(_.hex))
            .mutate(v = _.n.log())
            .mutate(normalized_values = _.v / _.v.max())
            .to_pandas()
            )
    rgb = cm.viridis(df.normalized_values)
    rgb_array = np.round( rgb * 255 ).astype(int).clip(0,255).tolist()
    df['rgb'] = rgb_array
    #df['viridis_hex'] = colors.to_hex(rgb)  # not robust?
    df['viridis_hex'] = [f"#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}" for c in rgb]    
    return df
    
def filter_gbif(_df, species="Canis lupus", bbox = [-130., 30., -90., 60.]):
     return (_df
            .filter(_.decimallongitude >= bbox[0],
                    _.decimallongitude < bbox[2],
                    _.decimallatitude >= bbox[1],
                    _.decimallatitude < bbox[3],
                    _.species == species
                   )
      )
```

```{code-cell} ipython3
df = filter_gbif(gbif_h3)
df = get_h3point_df(df, 8)
```

```{code-cell} ipython3
m = leafmap.Map(style="positron",  center=(-121.4, 37.50), zoom=7)
v_scale =1
deck_grid_layer = {
    "@@type": "H3HexagonLayer",
    "id": "my-layer",
    "data": df,
    "getHexagon": "@@=hex",
    "getFillColor": "@@=rgb",
    "getElevation": "@@=normalized_values",
    "elevationScale": 5000 * 10 ** v_scale,
    "elevationRange": [0,1],
}
m.add_deck_layers([deck_grid_layer])
m
```

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
    model="llama3-groq-tool-use:70b",
   # model="llama3.1",
    temperature=0,
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({
    "system": "You are a helpful agent.  Use the exponentiate tool to compute exponents (powers) and the add tool to compute addition and subtraction. Always try to determine and invoke a tool to perform your tasks.  If you do not have enough information, ask for more details.",
    "input": "what's 3 plus the value of 5 raised to the 2.743. Then substract 918.1241", })
```
