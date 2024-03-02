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

$$
f_{\theta_e}(x)=\begin{cases}y_{\text{updated}} &\text{if }x\in I(x_e) \\
f_{\theta}(x) &\text{if }x\in O(x_e)\end{cases}
$$

### Task definition
Usually consists of the following 3 tasks:
- Knowledge insertion: Inserts knowledge that LLMs have not seen before.
- Knowledge modification: Update outdated/incorrect knowledge.
- Knowledge erasure: Erase unwanted knowledge

## Evaluation for Knowledge Editing
### Reliability
- Description: 
    Success rate of editing **based on the edit description** $Z_e$, which is evaluated using the **accuracy** of the post-edit model $f_{\theta_e}$
- Definition:

```math
\mathbb{E}_{(x_e',y_e')\sim Z_e}\mathbb{1}\left\{\mathrm{argmax}_y [p_{\theta_e}(y|x_e')]=y_e'\right\}
```

### Generalization
- Description:
    Success rate (accuracy) within input set containing **in-scope samples** $(x_e',y_e')\sim I_e(x_e)$
- Definition:

```math
\mathbb{E}_{(x_e',y_e')\sim I_e(x_e)}\mathbb{1}\left\{\mathrm{argmax}_y [p_{\theta_e}(y|x_e')]=y_e'\right\}
```

### Portability (Robust generalization)
- Description:
    Success rate of editing when **transferring knowledge to related content** $(x_e',y_e')\sim P_e(x_e)$. Similar to generalization, but is **evaluated on factual reasoning** (one-hop, synonym, subject-replace, reverse-relation, one-to-one relation) not predicted label.
- Definition:

```math
\mathbb{E}_{(x_e',y_e')\sim P_e(x_e)}\mathbb{1}\left\{\mathrm{argmax}_y [f_{\theta_e}(y|x_e')]=y_e'\right\}
```

- Factual reasoning tasks: **WIP ...**
    - one-hop: 
    - synonym: 
    - subject-replace: 
    - reverse-relation: 
    - one-to-one relation: 
    - **<span style="font-size:1.2em;">Multi-hop:</span>**
### Locality (Specificity)
- Description: Evaluates if model's output **changes only within the editing scope $I_e(x_e)$, without affecting out-of-scope samples $O_e(x_e)$**. Checks if the edited model $f_{\theta_e}$ output remains aligned with the original model $f_{\theta}$.
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
    <br>(Reference: https://rome.baulab.info/)
- Steps:
    1. Early (Bottom) MLP layers in the last-subject position encodes many subject-related attirubtes.
    2. MHSA aggregates and propagates the attributes to the prediction position.
    3. Middle layer MHSA in the prediction position queries from previouse layer and extract information.
    4. Top layer MLPs are also associated with semantic/fact kowledge
![layers](imgs/layers.png)
<br>(Reference: https://arxiv.org/pdf/2401.01286.pdf)

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
- [IKE](https://arxiv.org/abs/2305.12740)
- [MQuAKE](https://arxiv.org/abs/2305.14795)
- [DeepEdit](https://arxiv.org/abs/2401.10471)

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
- Concept editing: To appear 2024
- [EVEDIT](https://arxiv.org/abs/2402.11324)
- [Relation-based](https://arxiv.org/abs/2311.09053)
- [Temporal Knowledge Editing](https://arxiv.org/abs/2312.05497)
- [WilKE](https://arxiv.org/abs/2402.10987)
- [Multi-lingual](https://arxiv.org/abs/2309.08952)
- [Multi-modal](https://arxiv.org/abs/2310.08475)

## Future Prospects, Challenges, and Opportunities
**WIP ...**
