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

Why splitting source code files? There are two primary reasons:

1. **Model Input Limit**: All embedding models have a token limit for input text. For instance, OpenAI's [text-embedding-3-small] model has a token limit of 8192. If a code snippet exceeds this limit, it must be split into smaller chunks.

2. **Semantic Granularity**: Breaking code into smaller chunks allows for more precise semantic understanding. A long code snippet may contain multiple functions or classes. By splitting the code, each chunk can focus on a specific piece, improving the relevance and quality of code retrieval.

To achieve this, one can simply split based on characters, words, or lines, as long as the chunk size remains below the model's token limit. However, a more sophisticated approach is to split based on tokens directly. Two common tokenizers used in the NLP community are:

- [tiktoken](https://github.com/openai/tiktoken): A fast Byte Pair Encoding (BPE) tokenizer for OpenAI models.

- [tokenizers](https://github.com/huggingface/tokenizers): Developed by Huggingface to use in the Transformers ecosystem. We choose to use the tiktoken for use with OpenAI's embedding models.

#### Splitting Strategies

A navie strategy is to split code snippets based on a fixed number of tokens. This is obviously not ideal because it may cut off in the middle of a semantic code block such as a function or a class. It is desirable for a more intelligent splitter that understands the code's structure and splits at the appropriate semantic boundaries. One such approach is Langchain's [recursive text splitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/). It splits text using top-level delimiters (sunch as class and function definitions) and then concatenates chunks as long as they stay within the character limit. However, this approach requires language-specific delimiters, and its performance suffers with languages that use curly braces heavily. Handling these edge cases can become cumbersome.

A more innovative and elegant approach is to parse the code into an Abstract Syntax Tree (AST) and split based on its structure, as detailed in the blog [post](https://docs.sweep.dev/blogs/chunking-2m-files). By traversing the AST in a depth-first fashion, the code can be split into sub-trees that fit within the token limit. To avoid generating too many small chunks, sibling nodes can be merged into larger chunks, provided they stay within the token constraint. LlamaIndex also provides a cleaner Python implementation of this approach in its [CodeSplitter](https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.node_parser.CodeSplitter.html) function. Both implementations utilize [tree-sitter](https://crates.io/crates/tree-sitter) for AST parsing. tree-sitter supports a wide range of [languages], making it a more generalizable solution.

#### Our Approach

We use [code-splitter](https://github.com/wangxj03/code-splitter) (shameless plug: I' the author) which is a Rust re-implementation for added efficiency. The gist below shows the use of its [Python bindings](https://pypi.org/project/code-splitter/) to walk through a directory and split Rust files. The choice of `TiktokenSplitter` is to for compatibility with OpenAI embedding models.

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

Qdrant's authors selected the open-source [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model for their [code search demo](https://github.com/qdrant/demo-code-search). Since this model is primarily trained on natural language tasks, they created a synthetic text-like representation of the code, including key elements like function names, signatures, and docstrings, which were then passed to the model.

We found this approach cumbersome and opted for OpenAI’s [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) model instead. While this model isn’t specifically designed for code, it still performs effectively on code-related tasks. Alternatively, embedding models tailored for code, such as Microsoft’s [unixcoder-base](https://huggingface.co/microsoft/unixcoder-base) or Voyage AI’s [voyage-code-2](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/), offer advantages like larger context lengths and better semantic retrieval for code.

### Indexing the Embeddings

Similar to the original demo, we utilize Qdrant to index the code chunk embeddings. Qdrant, an open-source vector database written in Rust, is designed to handle high-dimensional vectors for performance and massive-scale AI applications.

In the code snippet below, we also store metadata for each code chunk, such as file path, start and end line numbers, and chunk size. This metadata allows for displaying relevant information during search results on the frontend.

```python
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load code chunk embeddings from a Parquet file
df = pd.read_parquet("/data/code_embeddings.parquet")

# Initialize Qdrant client
client = QdrantClient("http://localhost:6333")

# Create or replace the collection with specific vector parameters
client.recreate_collection(
    collection_name="qdrant-code",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Convert embeddings and metadata into Qdrant-compatible points
points = [
    PointStruct(
        id=idx,
        vector=row["embedding"].tolist(),
        payload=row.drop(["embedding"]).to_dict(),
    )
    for idx, row in df.iterrows()
]

# Upload points to the Qdrant collection
client.upload_points("qdrant-code", points)
```

## Semantic Code Search
