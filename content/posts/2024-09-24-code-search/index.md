---
title: "Semantic Code Search"
date: 2024-09-27T23:01:05-07:00
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

There’s been a lot of buzz lately about Cursor. One of its key features is [codebase indexing](https://docs.cursor.com/context/codebase-indexing), which turns Cursor into a context-aware coding assistant. This feature, as explained in this [post](https://forum.cursor.com/t/codebase-indexing/36), works as follows:

- Cursor chunks the files from your codebase into small chunks locally.
- These chunks are then sent to Cursor's server, where embeddings are created using either OpenAI's embedding API or a custom embedding model.

The embeddings, along with start/end line numbers and file paths, are stored in a remote vector database. When you use @Codebase or ⌘ Enter to ask about your codebase, Cursor retrieves the relevant code chunks from the vector database to provide context for large language model (LLM) calls. Essentially, Cursor uses a standard Retrieval-Augmented Generation (RAG) model, with the codebase index acting as the retrieval mechanism.

In this post, we'll replicate the codebase indexing feature and demostrate it by building a semantic code search application. This application includes two main components: an offline ingestion pipeline to index code embeddings into a vector database, and a code search server that performs semantic retrieval from this database. We draw heavy inspiration from Qdrant’s [code search demo](https://github.com/qdrant/demo-code-search/tree/master) but provide our own implementations for the ingestion pipeline and code search backend.

## Ingestion Pipeline

The ingestion pipeline has three main steps: splitting source code into chunks, creating embeddings, and indexing them into a vector database.

### Splitting

Why splitting source code files? There are two primary reasons:

1. **Model Input Limit**: All embedding models have token limits. OpenAI's [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) model, for example, has a token limit of 8192. If a code snippet exceeds this limit, it needs to be broken down.

2. **Semantic Granularity**: Smaller chunks offer more precise semantic understanding. Large code snippets often contain multiple functions or classes, and by splitting the code, each chunk can focus on a specific part, enhancing retrieval relevance and quality.

You could split code based on characters, words, or lines, as long as the resulting chunks stay within the token limit. However, a more advanced method is to split based on tokens. Two common tokenizers in the NLP community are:

- [tiktoken](https://github.com/openai/tiktoken): A fast Byte Pair Encoding (BPE) tokenizer for OpenAI models.

- [tokenizers](https://github.com/huggingface/tokenizers): Developed by Huggingface for the Transformers ecosystem.

We use Tiktoken here for compatibility with OpenAI's embedding models.

#### Splitting Strategies

A naive strategy is to split code snippets based on a fixed token count, but this can cut off code blocks like functions or classes mid-way. A more effective approach is to use an intelligent splitter that understands code structure, such as Langchain's [recursive text splitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/). This method uses high-level delimiters (e.g., class and function definitions) to split at the appropriate semantic boundaries and concatenates chunks while staying within token limits. However, this approach is language-specific and struggles with languages that use curly braces.

A more elegant solution is to split the code based on its Abstract Syntax Tree (AST) structure, as outlined in this [blog post](https://docs.sweep.dev/blogs/chunking-2m-files). By traversing the AST depth-first, you can split the code into sub-trees that fit within the token limits. To avoid creating too many small chunks, sibling nodes can be merged into larger chunks as long as they stay under the token constraint. LlamaIndex offers a cleaner Python implementation in its [CodeSplitter](https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.node_parser.CodeSplitter.html) function. Both implementations use [tree-sitter](https://crates.io/crates/tree-sitter) for AST parsing, which supports a wide range of languages.

#### Our Approach

We use [code-splitter](https://github.com/wangxj03/code-splitter) (shameless plug: I' the author!), a Rust re-implementation for added efficiency. Below shows an example of using its [Python bindings](https://pypi.org/project/code-splitter/) to walk through a directory and split Rust files. We use `TiktokenSplitter` to ensure compatibility with OpenAI embedding models.

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

Qdrant's authors used the open-source [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model in their demo. Since this model is primarily trained on natural language tasks, they created a synthetic text-like representation of the code and passed it to the model. The representation captures key elements like function names, signatures, and docstrings.

We opted to use OpenAI’s [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) model instead, which, while not specifically trained on code, still performs reasonable well on code-related tasks. Alternatively, models like Microsoft’s [unixcoder-base](https://huggingface.co/microsoft/unixcoder-base) or Voyage AI’s [voyage-code-2](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/) provide longer context window and are purposedly trained for code-related tasks.

### Indexing

We use [Qdrant](https://github.com/qdrant/qdrant) to index the code chunk embeddings, just like in the original demo. Qdrant, an open-source vector database written in Rust, is optimized to handle high-dimensional vectors at scale.

In the snippet below, we also store metadata such as file paths, start/end line numbers, and chunk sizes. This metadata enables the frontend to display relevant information during search results.

```python
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

df = pd.read_parquet("/data/code_embeddings.parquet")

client = QdrantClient("http://localhost:6333")

client.recreate_collection(
    collection_name="qdrant-code",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=idx,
        vector=row["embedding"].tolist(),
        payload=row.drop(["embedding"]).to_dict(),
    )
    for idx, row in df.iterrows()
]

client.upload_points("qdrant-code", points)
```

Additionally, we index entire code files in a separate Qdrant collection to enable full-file retrieval during search results.

## Semantic Code Search

With the Qdrant database populated with code chunk embeddings and metadata, we can now build a code search server. The architecture of our search application is as follows:

![](code_search_design.svg)

The backend, built with [FastAPI](https://github.com/fastapi/fastapi), handles REST requests and interacts with the Qdrant vector database. It exposes two endpoints:

- **`GET /api/search`**: Search for code snippets based on a query.
- **`GET /api/file`**: Fetch the full content of a file based on its path.

We reuse the [React frontend](https://github.com/qdrant/demo-code-search/blob/master/frontend) from Qdrant's demo. Below is an example query using the UI:

![](code_search_example.svg)

## Wrapping Up

We’ve built a semantic code search application, replicating Cursor’s codebase indexing functionality. This solution gives you full control over each component—from splitting code snippets and generating embeddings to indexing them in a vector database and building a search server. It also provides a flexible, extensible foundation for creating more advanced GenAI applications.

Feel free to check out the complete source code at:
https://github.com/wangxj03/ai-cookbook/tree/main/code-search.
