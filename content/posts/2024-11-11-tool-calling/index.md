---
title: "Tips to Improve LLM Tool Calling"
date: 2024-11-10T20:37:57-08:00
tags: ["AI"]
author: "Xiaojing Wang"
showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: false
disableHLJS: false
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: false
UseHugoToc: true
---

LLM tool calling is a critical step in any agentic AI workflow. In this post, we build an application with [DuckDB](https://github.com/duckdb/duckdb) to show a few tips to improve LLM tool calling.

## Text-to-SQL Application

We build a simple text-to-SQL application with the [Hacker news](https://motherduck.com/docs/getting-started/sample-data-queries/hacker-news/) dataset compiled by [Mother Duck](https://motherduck.com/). The dataset contains all user posts in most of 2022 along with comments and votes. For simplicity, we only keep story posts with non-null text. The application essentially provides a natural language interface to the underlying dataset. It first translates the input text to a SQL query, then executes the SQL query against the database, and finally communicates the JSON result back to the user in natural language. At the core of the application is the `run_sql` function that executes SQL queries against the Hacker news table `posts`.

```python
import duckdb
from duckdb import DuckDBPyConnection

def get_connection() -> DuckDBPyConnection:
    global con
    if con is None:
        con = duckdb.connect(":memory:")
        con.execute("""
CREATE TABLE IF NOT EXISTS posts AS
SELECT * FROM read_parquet('data/hacker_news.parquet');
""")
    return con


def run_sql(query: str) -> str:
    """Run DuckDB SQL query against Hacker news table `posts` and return the result as a JSON string."""

    con = get_connection()
    df = con.sql(query).fetchdf()

    # Truncate the result if it's too long
    if len(df) > 100:
        logging.warning(
            f"The result contains {len(df)} rows. Only returning the first 100."
        )
        df = df.head(100)

    return df.to_json(orient="records")
```

The application has the following lifecycle:

![alt text](function_call.png)

### Success Example

With input query "What are the most commented posts?", the LLM manages to convert it to SQL query`SELECT title, comments FROM posts ORDER BY comments DESC LIMIT 10;` and returns the result in markdown list

![alt text](most_commented.png)

### Failure Example

While it works for simple aggregation query, the LLM starts to struggle for more complex queries like "What are the most commented posts each month?" which requires a sub-query or more advanced SQL features. Below are a few examples of incorrect SQL queries:

- Incorrect HAVING clause in the SQL query:

  ```sql
  SELECT strftime('%Y-%m', timestamp) as month, MAX(comments) as max_comments, title
  FROM posts
  GROUP BY month, title
  HAVING comments = (SELECT MAX(comments) FROM posts WHERE strftime('%Y-%m', timestamp) = month)
  ORDER BY month;
  ```

- Unnecessary and incorrect outer query

  ```sql
  SELECT strftime(timestamp, '%Y-%m') as month, title, comments
  FROM (
  SELECT month, title, comments,
  ROW_NUMBER() OVER (PARTITION BY month ORDER BY comments DESC) as rn
  FROM (
      SELECT strftime(timestamp, '%Y-%m') as month, title, comments
      FROM posts
  )
  )
  WHERE rn = 1;
  ```

## Tip 1: Retry on Error

LLM tool calling is not always successful on the first attempt, especially when dealing with more complex task. However, it excels when provided with additional context or feedback. By retrying on error, LLM can continue to make progress towards the correct solution. We witnessed this in the recent [computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) for coding demo by Anthropic. As part of the coding task to build a new personal website, Claude first attempted to start a Python server with an error, but then retried with a new version and succeeded.

![alt text](claude_computer_use_for_coding.png)

Let's revisit the scenaior where the LLM fails with input "What are the most commented posts?". Initially, the LLM produces an invalid SQL query

```sql
SELECT STRFTIME(timestamp, '%Y-%m') AS month, title, MAX(comments) as max_comments
FROM posts
GROUP BY month
ORDER BY month;
```

which triggers the following DuckDB execution error

```sh
Binder Error: column "title" must appear in the GROUP BY clause or must be part of an aggregate function. Either add it to the GROUP BY list, or use "ANY_VALUE(title)" if the exact value of "title" is not important.
```

By sending the error message back to the LLM along with the original query, the model has the opportunity to refine its response. In this case, the LLM generates a corrected query:

```sql
SELECT month, title, comments FROM (
    SELECT STRFTIME(timestamp, '%Y-%m') AS month, title, comments,
           ROW_NUMBER() OVER (PARTITION BY STRFTIME(timestamp, '%Y-%m') ORDER BY comments DESC) as rn
    FROM posts
) WHERE rn = 1;
```

## Tip 2: Allow More Thinking Time

There has been a lot of recent talks about the inference scaling law which suggests the performance of a model improves with more computing time spent on inference. Jim Fan [tweeted](https://x.com/DrJimFan/status/1834279865933332752) about it when the OpenAI o1 model was out. Sequoia published a blog post [Generative AI’s Act o1](https://www.sequoiacap.com/article/generative-ais-act-o1/) to look into the future with its implications.

How does this apply to our text-to-SQL application? One simple approach to allow more thinking time is to a pre-tool-calling step. In this step, the model is instructed to first break down the task into logical steps and draft SQL for each step before proceeding to the tool calling and execution.

Consider the same input query "What are the most commented posts each month?". By adding a reasoning step, the model can sketch out a plan to improve the SQL generation. Here's an example captured in the Langfuse [trace](https://us.cloud.langfuse.com/project/cm27ro2si00cd8mi56o0af4bq/traces/69bd6cfc-c8b6-4960-a3ef-08d6f4b06a73), where the model explicitly reasons through the solution

> 1. **Extract the year and month** from the `timestamp` to group the data on a monthly basis.
> 2. **Rank the posts** within each month based on the number of comments to identify the most commented one.
> 3. **Filter** the top-ranked post for each month.

Using these steps, the model constructures subqueries and combines them into the final SQL

```sql
WITH posts_with_month AS (
    SELECT
        title,
        comments,
        EXTRACT(YEAR FROM timestamp) AS year,
        EXTRACT(MONTH FROM timestamp) AS month
    FROM
        posts
),
ranked_posts AS (
    SELECT
        title,
        comments,
        year,
        month,
        ROW_NUMBER() OVER (PARTITION BY year, month ORDER BY comments DESC) AS rank
    FROM
        posts_with_month
)
SELECT
    title,
    comments,
    year,
    month
FROM
    ranked_posts
WHERE
    rank = 1
ORDER BY
    year, month;
```

While this query is slightly verbose, it is accurate and demonstrates the model’s ability to methodically reason through the problem.

## Tip 3: Few-shot Learning

One quick way to improve SQL generation quality is to show LLM some examples. DuckDB's [documentation](https://duckdb.org/docs/sql/introduction) is a rich source of such examples. We can parse the documentation in [markdown](https://github.com/duckdb/duckdb-web/tree/main/docs/sql) format and extract snippets of SQL query examples. Here we leverage the [instructor](https://github.com/instructor-ai/instructor) for structured output with LLM. We simply need to define the output Pydantic model and use the field documentation as prompts for the LLM.

```python
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel, Field


class Example(BaseModel):
    task: str = Field(
        ..., description="Description of the task that is to be solved by the SQL query"
    )
    sql: str = Field(..., description="DuckDB SQL query to solve the task")
    explanation: str = Field(
        ...,
        description="Generic explanation of the query syntax found in the surrounding markdown",
    )


class ExampleBank(BaseModel):
    """
    Parse the input markdown string to extract text-to-sql examples with explanations.
    Extract one example per sql code block.
    Be sure to inspect all sql code blocks.
    The generic explanation must be strictly based on the surrounding markdown not your prior knowledge.
    Avoid include example specific details such table name or column name in the explanation.
    """

    examples: list[Example] = Field(..., description="List of examples")


def parse(client: OpenAI, input: str, model: str = "gpt-4o") -> list[Example]:
    return client.chat.completions.create(
        model=model,
        response_model=ExampleBank,
        messages=[ChatCompletionUserMessageParam(content=input, role="user")],
    )
```

Once we we extract all the examples from the DuckDB documentation, we can create embedding for each example and index those embeddings into a vector database. At inference time, we can retrieve the most similar examples to the input query and use them as additional context for the LLM.

Langfuse [trace](https://us.cloud.langfuse.com/project/cm27ro2si00cd8mi56o0af4bq/traces/e8956c34-6569-4324-ad4e-3b0be153b9e2) shows that the model retrieves 10 most similar examples for the input query "What are the most commented posts each month?". One [example](https://duckdb.org/docs/sql/query_syntax/qualify.html#examples) illustrates the use of the `QUALIFY`. The LLM was able to generate a correct SQL query using the `QUALIFY` clause which is the most succinct and efficient way to solve the problem.

```sql
SELECT DATE_TRUNC('month', timestamp) as month, title, comments
FROM posts
WHERE comments IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('month', timestamp) ORDER BY comments DESC) = 1;
```
