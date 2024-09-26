---
title: "Semantic Code Search"
date: 2024-09-24T23:01:05-07:00
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

Recently, there has been significant buzz around Cursor. One of its key features is [codebase index](https://docs.cursor.com/context/codebase-indexing), which transforms Cursor into a context-aware coding assistant. Behind the scenes, the codebase indexing works as follows based on this [post](https://forum.cursor.com/t/codebase-indexing/36):

- Cursor chunks the files in your codebase into small chunks locally.
- These chunks are then sent to Cursor's server to create embeddings either with OpenAI's embedding API or by a custom embedding model.

The embeddings are stored in a remote vector database, along wtih starting / ending line numbers and the relative file path. When you use @Codebase or ⌘ Enter to ask questions about your codebase, Cursor will retrieve the relevant code chunks from the vector database as context in the LLM calls to generate the answer. In other words, Cursor implements a standard Retrieval-Augmented Generation (RAG) model with the codebase index as the retrieval mechanism.

In this post, we replicate the codebase indexing feature and demostrate it by building a semantic code search application. This process involves two key components: an offline ingestion pipeline to store code embeddings in a vector database, and a code search server that performs semantic retrieval from this database.

## Ingestion Pipeline

The ingestion pipeline consists of three main steps: splitting source code files into chunks, creating embeddings for these chunks, and indexing these embeddings in a vector database.

[OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) to create embeddings, and [Qdrant](https://github.com/qdrant/qdrant) as the vector DB for indexing and retrieval. We draw inspiration from Qdrant's own [demo](https://github.com/qdrant/demo-code-search/tree/master) and reuse its frontend code. We will provide our own implementation for the ingestion and backend.

### Splitting

Why splitting source code files? There are two main reasons:

1. **Model Input Limit**: All embedding models have a token limit for input text. For example, OpenAI's [text-embedding-3-small] model has a limit of 8192 tokens. If a code snippet is too long, it needs to be split into smaller chunks.

2. **Semantic Granularity**: Smaller chunks help capture more fine-grained semantics. Long code snippets may contain multiple functions or classes. By splitting, each chunk focuses on a distinct piece of code, which can help improve the quality of retrieval.

We use [code-splitter](https://github.com/wangxj03/code-splitter) (shameless plug: I am the author) to chunk source code files. code-splitter is a Rust library that provides a fast and efficient way to split code into smaller chunks. It utilizes [tree-sitter](https://crates.io/crates/tree-sitter) to parse code into an Abstract Syntax Tree (AST) and merges sibling nodes to create the largest possible chunks without exceeding the chunk limit.

The gist below shows the use of code-splitter [Python bindings](https://pypi.org/project/code-splitter/) to walk through a directory and split Rust files. The choice of `TiktokenSplitter` is to match the tokenizer used by OpenAI's embedding models.

```python
from code_splitter import Language, TiktokenSplitter

def walk(dir: str, max_size: int) -> Generator[dict[str, Any], None, None]:
    splitter = TiktokenSplitter(Language.Rust, max_size=max_size)

    for root, _, files in os.walk(dir):
        for file in files:
            if not file.endswith(".rs"):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir)

            with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            with open(file_path, mode="rb") as f:
                code = f.read()
                chunks = splitter.split(code)

                for chunk in chunks:
                    yield {
                        "file_path": rel_path,
                        "file_name": file,
                        "start_line": chunk.start,
                        "end_line": chunk.end,
                        "text": "\n".join(lines[chunk.start : chunk.end]),
                        "size": chunk.size,
                    }
```

### Creating Embeddings

We use OpenAI's [Embeddings API](https://platform.openai.com/docs/guides/embeddings) to create embeddings. To speed up the process, we batch multiple chunks into a single request. Alternatively, one can also use the [Batch API](https://platform.openai.com/docs/guides/batch/overview) to create embeddings for a large codebase.

### Indexing the Embeddings

We use Qdrant as the vector database to store the embeddings. Qdrant is an open-source vector database that supports fast and efficient similarity search.

## Semantic Code Search
