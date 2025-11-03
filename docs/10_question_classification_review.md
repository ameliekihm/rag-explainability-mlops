# Question Classification in QA Systems – Literature Review

## 1. Overview
In this note, I summarize recent studies on question classification (QC) for question answering systems.  
My goal is to understand how existing taxonomies compare with the question types in my dataset  
and to see whether those frameworks can be integrated into my own retrieval-augmented generation (RAG) project.


## 2. Related Work

### [Sun et al., 2023 — *Question Classification for Intelligent QA: A Comprehensive Survey*](https://www.mdpi.com/2220-9964/12/10/415)
- Defines four main types of questions:
  1. **Content-based** – factual or descriptive  
  2. **Template-based** – fixed conversational patterns  
  3. **Calculation-based** – numeric or reasoning tasks  
  4. **Method-based** – procedural or “how” questions  
- Also introduces a three-layer structure (*Essence*, *Form*, and *Implementation*).  
  I find this framework useful because it connects question intent with reasoning style.

### [Wang & Mine, 2024 — *One Stone, Four Birds: A Comprehensive Solution for QA System Using Supervised Contrastive Learning*](https://arxiv.org/abs/2407.09011)
- Proposes a **contrastive learning-based model** that jointly handles **question classification**, **intent detection**, and **retrieval ranking**.  
- The model uses **shared embeddings** for both question and answer representations, improving consistency across subtasks.  
- The paper emphasizes that **user intent recognition** helps guide both **retrieval scope** and **answer generation quality**.  
- This framework suggests that **classification is not isolated**, but part of an **integrated feedback loop** in QA systems.  
- I find this useful because it relates to **my idea of adjusting retrieval depth based on question type**.  

### [Classifying the State of Knowledge-Based QA, 2024](https://www.tandfonline.com/doi/full/10.1080/1206212X.2024.2426512)
- Focuses on how **different types of knowledge sources** (structured, semi-structured, unstructured) affect the QA process.  
- The authors argue that classification should also consider **knowledge access methods**, such as **symbolic retrieval**, **neural retrieval**, or **hybrid pipelines**.  
- Their taxonomy links **question type** to **knowledge dependency** — e.g., **factual vs reasoning vs hybrid questions**.  
- This viewpoint can extend **my RAG explainability module** by mapping **which question types rely more on external retrieval** versus **internal reasoning**.  


## 3. My Dataset Taxonomy

I created eight major question categories based on patterns in my current dataset.  
Each category was compared with the frameworks from **Sun et al. (2023)**, **Wang & Mine (2024)**,  
and **Classifying the State of Knowledge-Based QA (2024)** to see how they align.

| My Category | Example Question | Sun et al. (2023) | Wang & Mine (2024) | Knowledge-Based QA (2024) | Notes |
|--------------|------------------|------------------|------------------|------------------|--------|
| **Fact** | When was the first iPod released? | Content-based | User intent: factual lookup | Knowledge: structured fact | Direct factual question |
| **Date** | What year did the iPod Nano come out? | Content-based | User intent: temporal reasoning | Knowledge: structured + retrieval | Requires temporal reasoning |
| **Location** | Where was the first Apple Store opened? | Content-based | User intent: entity lookup | Knowledge: geo-linked retrieval | Spatial entity |
| **Person** | Who founded Apple Inc.? | Content-based | User intent: named entity query | Knowledge: structured database | Entity-based |
| **Reason (Why)** | Why did Kanye West partner with Adidas? | Method-based | User intent: causal / motivation | Knowledge: hybrid retrieval | Causal explanation |
| **Process (How)** | How does solar water disinfection work? | Method-based | User intent: procedural reasoning | Knowledge: hybrid / semi-structured | Step-by-step reasoning |
| **Quantity / Statistics** | How many people live in New York City? | Calculation-based | User intent: numeric retrieval | Knowledge: structured + aggregate | Numeric or statistical |
| **Opinion / Sentiment** | What made *To Kill a Mockingbird* so popular? | Content-based | User intent: evaluative / interpretive | Knowledge: unstructured (text-based) | Evaluative or interpretive |


This taxonomy connects my dataset to existing classification frameworks across three perspectives:  
- **Question form (Sun et al.)**  
- **User intent and reasoning process (Wang & Mine)**  
- **Knowledge dependency (Knowledge-Based QA)**  

I plan to use this mapping as a foundation for refining retrieval and explainability in later stages.



## 4. Reflection
I believe this taxonomy covers the range of questions in my dataset well.  
Each type demands a different reasoning process and retrieval behavior, which I can later explore in more depth.  
For now, my focus is on documenting these categories clearly and connecting them to the explainability aspects of my dashboard.  
If needed later, I can extend this work into a small classification experiment,  
but for now the review and taxonomy comparison are sufficient for my research goals.
