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

With input query "What are the most commented posts?", the application manages to convert it to SQL query`SELECT title, comments FROM posts ORDER BY comments DESC LIMIT 10;` and returns the result in markdown list

![alt text](most_commented.png)

### Failure Example

While it works for simple aggregation query, the application starts to struggle for more complex queries like "What are the most commented posts each month?" which requires a sub-query or more advanced SQL features. Below are a few examples of incorrect SQL queries:

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

When the application fails to generate a correct SQL query, just feed the SQL execution error message back to LLM and let it figure out the next step. In many cases, the LLM can self-resolve the issue.

For example, using the same input query "What are the most commented posts?" as above, the application first generates an incorrect SQL query

```sql
SELECT STRFTIME(timestamp, '%Y-%m') AS month, title, MAX(comments) as max_comments
FROM posts
GROUP BY month
ORDER BY month;
```

with an execution error

```sh
Binder Error: column "title" must appear in the GROUP BY clause or must be part of an aggregate function. Either add it to the GROUP BY list, or use "ANY_VALUE(title)" if the exact value of "title" is not important.
```

It is able to retry and recover from the error with the correct SQL query. See the full Langfuse [trace](https://us.cloud.langfuse.com/project/cm27ro2si00cd8mi56o0af4bq/traces/1eafe226-882e-4d52-aef0-390abbc1b181?observation=199d4a6b-6530-47fb-8616-9b7e97c63aa0).

```sql
SELECT month, title, comments FROM (
    SELECT STRFTIME(timestamp, '%Y-%m') AS month, title, comments,
           ROW_NUMBER() OVER (PARTITION BY STRFTIME(timestamp, '%Y-%m') ORDER BY comments DESC) as rn
    FROM posts
) WHERE rn = 1;
```
