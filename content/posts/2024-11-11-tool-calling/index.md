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

LLM tool calling is a critical step in any agentic AI workflow. In this post, we build a text-to-SQL example with [DuckDB](https://github.com/duckdb/duckdb) to show a few tips to improve LLM tool calling.

## Text-to-SQL Application

We use the [Hacker news](https://motherduck.com/docs/getting-started/sample-data-queries/hacker-news/) dataset compiled by [Mother Duck](https://motherduck.com/). The dataset contains all user posts in most of 2022 along with comments and votes. For simplicity, we only keep the posts of type `story` with non-null text.

We build a simple text-to-SQL application with DuckDB. The application maintains a global connection to the database and provides a `run_sql` function to execute SQL queries. The `run_sql` function takes a SQL query as input and returns the result in a JSON string.

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

When calling it with input text "What are the most commented posts?", the bot manages to convert it to SQL query`SELECT title, comments FROM posts ORDER BY comments DESC LIMIT 10;` and executes it with the `run_sql` function.

![alt text](most_commented.png)

While it works for the simple query, the application may fail for more complex queries. For example, when calling it with input text "What are the most commented posts each month?", the bot may fail due to faulty SQL query generation. We show a few faulty SQL queries below:

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

The first tip is to retry on error. When the application fails to generate a correct SQL query, it can retry with a simpler query. For example, when the bot fails to generate the correct SQL query for the input text "What are the most commented posts each month?", it can retry with a simpler query `SELECT title, comments FROM posts ORDER BY comments DESC LIMIT 10;`.
