# 1. Mean Similarity Analysis by Question Type
<img width="1191" height="543" alt="Image" src="https://github.com/user-attachments/assets/d79a5a57-9625-45c4-9272-aad2d36f8249" />

## 1-1. Overview
This document analyzes how retrieval quality varies across different question types in our RAG pipeline.  
Specifically, we evaluate **mean embedding similarity** between user queries and retrieved passages across three semantic groups:

- **Factual** (Date, Location, Person, Fact, Statistics, Entity)
- **Reasoning** (Why and How questions, Process)
- **Opinion** (Subjective or preference based questions)

Mean similarity reflects **how closely retrieved passages semantically align with the query**.  
Higher scores indicate that retrieval returned content that embedding space considers more relevant.


## 1-2. Key Findings

### 1-2-1. Mean Similarity by Category
Opinion queries show the highest similarity distributions, while factual and reasoning questions show lower and more constrained ranges.

| Group | Retrieval Difficulty | Interpretation |
|---|---|---|
| Opinion | Lowest | General statements align easily in embedding space, model relies more on internal priors |
| Factual | Medium | Requires precise match for names, dates, locations, factual entities |
| Reasoning | Medium High | Requires conceptual alignment, grounding distributed across multiple sentences |

### 1-2-2. Why Opinion Scores Are Higher
Opinion questions do **not require a specific factual grounding sentence**.  
Therefore, semantic search often retrieves broadly related or generic statements that still score highly in embedding space.

This does **not** mean retrieval is “more accurate” for opinion questions.  
Instead, it means the embedding model finds **loose semantic alignment more easily**, and the generation relies more on **model priors** rather than context evidence.

> Opinion tasks reduce dependency on external grounding, which artificially inflates similarity.


## 1-3. Interpretation Summary

| Metric Behavior | Explanation |
|---|---|
High similarity on opinion | Embeddings reward generic semantic closeness, not factual grounding |
Lower similarity on factual | Requires exact matching of entities, names, dates, places |
Lower but variable similarity on reasoning | Conceptual queries depend on distributed evidence rather than one precise passage |

## 1-4. Research Insight
The results highlight a fundamental RAG limitation:

> When external grounding is optional (like opinion queries), retrieval can appear “good” in metrics but contributes less to the answer quality.

This suggests future evaluation frameworks should consider:
- Grounding necessity by task type
- Confidence vs correctness correlation
- Retrieval contribution to final generation

# 2. Retrieval Variance Analysis by Question Type

<img width="1186" height="538" alt="Image" src="https://github.com/user-attachments/assets/c2ef6828-c9f9-4ae5-acac-20bd2c39c8ba" />

## 2-1. Overview
Retrieval variance measures how consistent the retrieved passages are for each query type.  
Lower variance indicates the retriever consistently fetches similar passages, while higher variance suggests the system pulls context from more diverse or inconsistent regions of the embedding space.

We examine variance across three semantic groups:

- **Factual**
- **Reasoning**
- **Opinion**


## 2-2. Key Findings

### 2-2-1. Retrieval Variance by Category

| Question Type | Variance Level | Interpretation |
|---|---|---|
| Opinion | Low | Semantic space is narrow, model retrieves similarly phrased general statements consistently |
| Factual | Medium + Outliers | Requires exact entity matching; failures lead to large jumps to unrelated context |
| Reasoning | Medium | Explanatory nature introduces broader semantic coverage and expression variety |


### 2-2-2. Why Opinion Queries Show Low Variance
Opinion tasks share broad thematic language (e.g., "remote work," "AI ethics," "social media"),  
so embeddings group many similar passages closely.  
As a result, the retriever repeatedly returns **general, semantically adjacent content**.

> Low variance does not imply strong grounding; it reflects **semantic homogeneity**, not retrieval precision.


### 2-2-3. Why Factual Queries Create Outliers
Factual questions require exact matching for:

- Names
- Dates
- Locations
- Numbers

When matching succeeds, variance remains moderate.  
However, a single failure leads to retrieval of unrelated context, producing **large variance spikes**.


### 2-2-4. Reasoning Queries Show Controlled Spread
Reasoning questions often involve:

- Explanatory or causal phrasing
- Conceptual associations across multiple sentences
- Flexible linguistic structure

Thus, retrieval explores a wider conceptual region, resulting in **medium variance** without extreme spikes.


## 2-3. Interpretation Summary

| Behavior | Explanation |
|---|---|
Low variance on Opinion | Embeddings cluster general opinion language tightly; consistency does not imply correctness |
Outliers in Factual | Precise entity matching failures cause abrupt retrieval shifts |
Medium spread in Reasoning | Explanation tasks draw from broader conceptual neighborhoods |


## 2-4. Research Insight

Retrieval stability varies by task type.  
This shows that RAG evaluation must consider **grounding requirement asymmetry**:

- Opinion tasks benefit little from retrieval consistency
- Factual tasks are sensitive to entity-level matching errors
- Reasoning tasks demand conceptual rather than literal alignment


# 3. Generation Length Analysis by Question Type

<img width="1180" height="534" alt="Image" src="https://github.com/user-attachments/assets/eb3fdcc8-542f-4551-a252-b3c64ab53392" />

## 3-1. Overview
Generation length represents the number of tokens produced in the model’s answer.  
Longer outputs generally correspond to tasks requiring explanation or multi-step reasoning, while shorter outputs appear in direct factual recall scenarios.


## 3-2. Key Findings

| Type | Length Pattern | Interpretation |
|---|---|---|
| Factual | Shortest | Direct recall; answers are concise (e.g., "1945", "Einstein") |
| Reasoning | Longest + broad spread | Explanation tasks require multi-sentence, stepwise reasoning |
| Opinion | Mostly short, occasional long spikes | Opinion stance is usually concise, but model may ramble when exploring reasoning style |


### 3-2-1. Why Factual Answers Are Short
Factual questions often ask for a **single entity, date, or factual point**.  
Since the retrieval context already provides the necessary information structure, the model produces minimal text.

> Factual tasks reflect pure information lookup behavior.


### 3-2-2. Why Reasoning Answers Expand
Reasoning queries (why/how/process) trigger:

- Step-by-step explanation
- Causal or conceptual reasoning
- Multi-sentence narrative structure

Thus, generation length naturally increases, and variance widens.

> Reasoning tasks activate the model's internal logical scaffolding.


### 3-2-3. Opinion Responses Stay Short — With Spikes
Opinion queries typically elicit brief stance statements:

- Personal-position style output
- Subjective phrasing without strict grounding
- Short declarative tone

However, occasional spikes appear when the model chooses to elaborate, producing **“ramble bursts”** where the model spontaneously justifies its view.

> Opinion tasks often bypass structured reasoning and rely on model priors.


## 3-3. Interpretation Summary

| Behavior | Explanation |
|---|---|
Short factual output | Retrieval provides exact info → minimal generation needed |
Long reasoning output | Requires structured explanation and step-by-step logic |
Opinion brevity with spikes | Subjective stance often brief, sometimes expands to justification |


## 3-4. Research Insight
Generation length reveals cognitive mode shifts in RAG systems:

- **Factual tasks** → lookup behavior
- **Reasoning tasks** → internal reasoning and explanation expansion
- **Opinion tasks** → stance expression with optional elaboration

This suggests evaluation frameworks should incorporate **generation discipline** metrics alongside correctness, especially for reasoning tasks.

# 4. Context Attention Analysis by Question Type

<img width="1183" height="528" alt="Image" src="https://github.com/user-attachments/assets/05252fe1-0364-4b07-a805-8990a83948a4" />

## 4-1. Overview
Context attention represents how strongly the model attends to retrieved passages when generating an answer.  
Values near 1.0 indicate heavy reliance on context.  
Lower values indicate the model answers more from internal knowledge rather than retrieved documents.

We analyze attention across three groups

* Factual
* Reasoning
* Opinion


## 4-2. Key Findings

| Group | Attention Pattern | Interpretation |
|---|---|---|
| Opinion | Extremely high attention near 1.0 but often superficial | Model answers mostly from internal priors, uses context as a stylistic anchor rather than grounding |
| Factual | High and stable with occasional dips | Typically grounded in retrieved evidence but may sometimes answer from memory instead |
| Reasoning | Medium average with wide variance | Model balances retrieval with internal reasoning depending on question framing |


### 4-2-1. Opinion Questions

* Attention nearly maxed out around 1.0
* Indicates minimal need to inspect context deeply
* Model likely uses retrieved text only loosely while generating answer from internal knowledge

**Summary**

> Opinion questions are answered primarily from internal priors  
> with context acting as a light reference rather than grounding.


### 4-2-2. Factual Questions

* High and stable attention scores
* Shows that retrieved context guides the final answer
* Occasional low attention outliers indicate cases where model attempts to answer without checking the evidence

**Summary**

> Factual questions mostly rely on context  
> but sometimes the model shortcuts and guesses from memory.


### 4-2-3. Reasoning Questions

* Medium attention values
* Wide spread across samples
* Reflects mixture of retrieval use and internal reasoning

**Summary**

> Reasoning tasks combine retrieved context and model logic  
> leading to mixed attention patterns.


## 4-3. Interpretation Summary

| Behavior | Explanation |
|---|---|
Opinion near one point zero | Context is acknowledged but internal model knowledge drives the response |
Factual high with dips | Retrieval is essential but shortcut hallucination attempts appear |
Reasoning medium with variance | Hybrid mode between grounded reference and reasoning from priors |


## 4-4. Research Insight
Attention behavior reveals different grounding strategies across task types

* Opinion tasks lean heavily on internal priors
* Factual tasks require context validation but are vulnerable to memory hallucination
* Reasoning tasks operate in a hybrid grounding reasoning mode


