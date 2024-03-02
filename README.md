# Knowledge Editing

<!-- Reference: https://github.com/jonschlinkert/markdown-toc -->

<!-- toc -->

- [Motivation](#motivation)
- [Problem Definition](#problem-definition)
  * [Description](#description)
  * [Definition](#definition)
  * [Task definition](#task-definition)
- [Evaluation for Knowledge Editing](#evaluation-for-knowledge-editing)
  * [Reliability](#reliability)
  * [Generalization](#generalization)
  * [Portability (Robust generalization)](#portability-robust-generalization)
  * [Locality (Specificity)](#locality-specificity)
  * [Other stuff](#other-stuff)
- [Methodology](#methodology)
  * [Mechanism of Knowledge Storage of LLM (Transformer-based)](#mechanism-of-knowledge-storage-of-llm-transformer-based)
  * [Knowledge Editing for LLMs](#knowledge-editing-for-llms)
  * [Expose LLM to New Knowledge During Inference](#expose-llm-to-new-knowledge-during-inference)
    + [Method Type 1](#method-type-1)
  * [Learning Knowledge Through LLM Parameters](#learning-knowledge-through-llm-parameters)
    + [Method Type 2: Utilize additional parameters](#method-type-2-utilize-additional-parameters)
    + [Method Type 3: Direct edit of intrinsic knowledge (Locating and Editing)](#method-type-3-direct-edit-of-intrinsic-knowledge-locating-and-editing)
  * [Beyond Editing Factual Knowledge](#beyond-editing-factual-knowledge)
- [Future Prospects, Challenges, and Opportunities](#future-prospects-challenges-and-opportunities)

<!-- tocstop -->

## Motivation
- LLM contains unwanted info
    - Bias
    - Misinfo
    - Harmful content (e.g. toxicity, offensive content, privacy issue)
    - Outdated info
- Goal:
    - **Alter** the unwanted info
    - **Maintain** other knowledge
- Why edit?
    - It is too expensive to re-train a LLM

## Problem Definition
### Description
**Change** the LLM's behavior for a given knowledge efficiently **without compromising other cases**

### Definition
- Given a function space $f:\mathbb{X}\rightarrow\mathbb{Y}$ estimated by an initial base model $f_\theta$ and a set of specified edit descriptor $Z_e=\{(x_e,y_e)\}$ s.t. $f_\theta(x_e')\neq y_e'\ \forall (x_e',y_e')\in Z_e$, efficiently create an edited model $f_{\theta_e}$ s.t. $f_{\theta_e}(x_e')=y_e'\ \forall (x_e',y_e')\in Z_e$.
- In addition to the edit descriptor $(x_e,y_e)$ itself, we define two sets that is related to **editing scope**:
    - In-scope samples $I(x_e)$: Includes $x_e$ and its related samples (neighborhood) $N(x_e)$
    - Out-of-scope samples $O(x_e)$: Includes samples that are not related to $x_e$
    - We expect the edited function $f_{\theta_e}$ to behave like the following:

```math
f_{\theta_e}(x)=\begin{cases}y_{\text{updated}} &\text{if }x\in I(x_e) \\
f_{\theta}(x) &\text{if }x\in O(x_e)\end{cases}
```

### Task definition
Usually consists of the following 3 tasks:
- Knowledge insertion: Inserts knowledge that LLMs have not seen before.
- Knowledge modification: Update outdated/incorrect knowledge.
- Knowledge erasure: Erase unwanted knowledge

## Evaluation for Knowledge Editing
### Reliability
- Description: 
    <br> Success rate of editing **based on the edit description** $Z_e$, which is evaluated using the **accuracy** of the post-edit model $f_{\theta_e}$
- Definition:

```math
\mathbb{E}_{(x_e',y_e')\sim Z_e}\mathbb{1}\left\{\mathrm{argmax}_y [p_{\theta_e}(y|x_e')]=y_e'\right\}
```

### Generalization
- Description:
    <br> Success rate (accuracy) within input set containing **in-scope samples** $(x_e',y_e')\sim I_e(x_e)$
- Definition:

```math
\mathbb{E}_{(x_e',y_e')\sim I_e(x_e)}\mathbb{1}\left\{\mathrm{argmax}_y [p_{\theta_e}(y|x_e')]=y_e'\right\}
```

### Portability (Robust generalization)
- Description:
    <br> Success rate of editing when **transferring knowledge to related content** $(x_e',y_e')\sim P_e(x_e)$. Similar to generalization, but is **evaluated on factual reasoning** (one-hop, synonym, subject-replace, reverse-relation, one-to-one relation) not predicted label.
- Definition:

```math
\mathbb{E}_{(x_e',y_e')\sim P_e(x_e)}\mathbb{1}\left\{\mathrm{argmax}_y [f_{\theta_e}(y|x_e')]=y_e'\right\}
```

- Factual reasoning tasks: **WIP ...**
    - One-hop: 
    - Synonym: 
    - Subject-replace: 
    - Reverse-relation: 
    - One-to-one relation: 
    - Multi-hop:
### Locality (Specificity)
- Description:
    <br> Evaluates if model's output **changes only within the editing scope $I_e(x_e)$, without affecting out-of-scope samples $O_e(x_e)$**. Checks if the edited model $f_{\theta_e}$ output remains aligned with the original model $f_{\theta}$.
- Definition:

```math
\mathbb{E}_{(x_e',y_e')\sim O_e(x_e)}\mathbb{1}\left\{p_{\theta_e}(y|x_e')=p_{\theta}(y|x_e')\right\}
```


### Other stuff
- Performance as an LLM: Fluency, robustnes, etc.
- Efficiency: Time/GPU/memory consumption

## Methodology

### Mechanism of Knowledge Storage of LLM (Transformer-based)
- **FFN (MLP)** layers are similar to **Neural Memory Network**.
- **Multi-head self-attention (MHSA)** layers aggregates knowledge from the previous layer through self-attention.
- Knowledge retrieved from FFN at (a), then brought to output token (b) by self-attention
    ![image](imgs/MLP.svg)
    <br> (Reference: https://rome.baulab.info/)
- Steps:
    1. Early (Bottom) MLP layers in the last-subject position encodes many subject-related attirubtes.
    2. MHSA aggregates and propagates the attributes to the prediction position.
    3. Middle layer MHSA in the prediction position queries from previouse layer and extract information.
    4. Top layer MLPs are also associated with semantic/fact kowledge
![layers](imgs/layers.png)
<br> (Reference: https://arxiv.org/pdf/2401.01286.pdf)

### Knowledge Editing for LLMs
- Requirements:
    - Analyze model behavior
    - Accurately locating the area to edit
    - Design efficient and low-cost methods

![overview](imgs/overview.jpg)
(Reference: https://github.com/zjunlp/KnowledgeEditingPapers)

### Expose LLM to New Knowledge During Inference
#### Method Type 1
- [SERAC](https://arxiv.org/abs/2206.06520)
    <br> In addition to the original base model, this method uses an **edit memory storage** (directly stores edit descriptions), a **scope classifier** that determines if the input is related to edit descriptions, and a **Retrieval-Augmented Counterfactual Model** that deals with input samples that are related to edit descriptions.
    - Advantage: Workable with small scope classifier and counterfactual models, doesn't require tuning the original base model. Can handle multiple edits.
    - Disadvantage: Constrained by the edit memory storage (not scalable). In other words, this method is not applicable when there is a large amount of edit descriptions
- [IKE](https://arxiv.org/abs/2305.12740)
    <br> Apply in-context learning to knowledge editing. Uses edit description, in-scope samples, and out-of-scope samples to demonstrate when the model should update its predictions.
    - Advantage: Better locality and generalization. Applicable to many LLMs.
    - Disadvantage: Not applicable to large amount of edit descriptions as the input length of LLMs is limited.
- [MeLLo](https://arxiv.org/abs/2305.14795) (Also the paper that proposed MQuAKE)
    <br> Inspired by SERAC, uses additional memory to store edit descriptions. They make the model decompose the multi-hop questions into sub-questions, generate tentative answers, then iteratively self-check if the answers are consistent with the edit descriptions and correct the answers if not.
    - Advantage: Improved generalization, easy to add/remove edit descriptions.
    - Disadvantage: Similar to SERAC, limited by memory and retrieval accuracy/relevance.
- [DeepEdit](https://arxiv.org/abs/2401.10471)
    <br> Proposed to view knowledge editing as decoding with contraints. They proposed to use a DFS-based progressive decoding method (multi-step reasoning) with information retrieval that can be applied to blackbox LLMs.
    - Proposed constraints:
        - Uniqueness: Ensure each reasoning step introduces new information.
        - Coherence: Ensure the current reasoning step is relevant to the previous step.
        - Awareness: Ensure the current reasoning step does not contradict with edit descriptions. Uses information retrieval to obtain relevant edit descriptions instead of looping through everything to save time.
        - Relevance: Ensures each reasoning step is helpful for finding the final answer. 
    - Advantage: More succinct and faithful reasoning while enforcing the new knowledge. Great quantitative improvement.
    - Disadvantage: *Not sure, but looks like it would cost a lot of time & resource*
### Learning Knowledge Through LLM Parameters
#### Method Type 2: Utilize additional parameters
- [CaliNET](https://arxiv.org/abs/2210.03329)
- [T-Patcher](https://arxiv.org/abs/2301.09785)
- [GRACE](https://arxiv.org/abs/2211.11031)

#### Method Type 3: Direct edit of intrinsic knowledge (Locating and Editing)
- [Knowledge editor](https://arxiv.org/abs/2104.08164)
- [MEND](https://arxiv.org/abs/2110.11309)
- [ROME](https://arxiv.org/abs/2202.05262)
- [Knowledge neuron](https://arxiv.org/abs/2104.08696)
- [MEMIT](https://arxiv.org/abs/2210.07229)
- [PMET](https://arxiv.org/abs/2308.08742)
- [Meta learning](https://arxiv.org/abs/2311.04661)

### Beyond Editing Factual Knowledge
- [Task Arithmetic](https://arxiv.org/abs/2212.04089)
- [DUNE](https://arxiv.org/abs/2311.16087)
- [LEME](https://arxiv.org/abs/2402.09394)
- Concept editing: To appear 2024 according to [this](https://drive.google.com/file/d/1fkTbVeRJSWmU7fBDeNf1OhHEkLSofQde/view)
- [EVEDIT](https://arxiv.org/abs/2402.11324)
- [Relation-based](https://arxiv.org/abs/2311.09053)
- [Temporal Knowledge Editing](https://arxiv.org/abs/2312.05497)
- [WilKE](https://arxiv.org/abs/2402.10987)
- [Multi-lingual](https://arxiv.org/abs/2309.08952)
- [Multi-modal](https://arxiv.org/abs/2310.08475)

## Future Prospects, Challenges, and Opportunities
**WIP ...**
