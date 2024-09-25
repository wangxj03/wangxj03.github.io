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

The embeddings are stored in a remote vector DB, along wtih starting / ending line numbers and the relative file path. When you use @Codebase or ⌘ Enter to ask questions about your codebase, Cursor will retrieve the relevant code chunks from the vector DB as context in the LLM calls to generate the answer. In other words, Cursor implements a standard Retrieval-Augmented Generation (RAG) model with the codebase index as the retrieval mechanism.

In this post, we will replicate the codebase indexing feature and demostrate it in a semantic code search application. We will use [`code-splitter`](https://github.com/wangxj03/code-splitter/tree/main/bindings/python) (disclaimer: I am also the author of this tool) to chunk the codebase, [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) to create embeddings, and [Qdrant](https://github.com/qdrant/qdrant) as the vector DB for indexing and retrieval. We draw inspiration from Qdrant's own [demo](https://github.com/qdrant/demo-code-search/tree/master) and reuse its frontend code. We will provide our own implementation for the ingestion and backend.
