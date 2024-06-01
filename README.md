# Editing the Mind of Giants: An In-Depth Exploration of Pitfalls of Knowledge Editing in Large Language Models


<!-- Reference: https://github.com/jonschlinkert/markdown-toc -->

## Table of Content

<!-- toc -->

- [Overview of Knowledgge Editing](#overview-of-knowledgge-editing)
  * [Problem Definition](#problem-definition)
    + [Reliability](#reliability)
    + [Generalization](#generalization)
    + [Locality](#locality)

<!-- tocstop -->

## Overview of Knowledgge Editing

### Problem Definition
We denote the input and output space as $\mathbb{X}$ and $\mathbb{Y}$, respectively. The function space $\mathbb{F}: \mathbb{X} \rightarrow \mathbb{Y}$ is estimated by the base model $f_{\theta_0}$ parameterized by $\theta_0 \in \Theta$. Finally, let $Z_e = \{ (x_e, y_e) \ | \ f_{\theta_0}(x_e) \neq y_e \}$ be the set of edit queries we would like to apply to the base model.
% The goal of knowledge editing is to efficiently update the model to the edited model $f_{\theta_e}$ satisfying:
The goal of knowledge editing is to efficiently derive the edited model $f_{\theta_e}$ from the base model that satisfies the following:
```math
f_{\theta_e}(x_e) = y_e, \forall (x_e, y_e) \in Z_e
```

As shown in the figure below, the ideal edited model $f_{\theta_e}$ should satisfy three properties: **reliability**, **generalization**, and **locality**.
<img src="imgs/basic_metric.png" width="250" align=center>

#### Reliability
Given an edit query $(x_e, y_e)$, the edited model $f_{\theta_e}$ should output the target answer $y_e$ when given the target input $x_e$, i.e. $f_{\theta_e}(x_e) = y_e$. The reliability of an editing method is measured by calculating the average edit success rate:
```math
\mathbb{E}_{(x_e', y_e')\sim Z_e} \mathbb{1}\{ f_{\theta_e}(x_e') = y_e' \}
```

#### Generalization
The edited model should generalize the edited knowledge to relevant instances. The generalization metric is commonly formulated as the average success rate on the neighboring set:
```math
\mathbb{E}_{(x_e', y_e')\sim N(x_e, y_e)} \mathbb{1} \{ f_{\theta_e}(x_e') = y_e' \},
```
where $N(x_e, y_e)$ is the set of neighboring instances of an edit query $(x_e, y_e)$. Earlier works evaluate this metric by rephrasing the input prompts.

#### Locality
The editing process should not affect instances unrelated to the edit queries. The locality set of an edit query $(x_e, y_e)$ can be defined as $L(x_e) = \{ (x_{loc}, y_{loc}) \in \mathbb{X} \times \mathbb{Y}\ \mathrm{s.t}\ x_{loc} \notin N(x_e, y_e) \land f_{\theta_0}(x_{loc}) = y_{loc} \}$. The locality, also known as specificity, of a editing method is measured by calculating the level of invariance of model output before and after the edits, which can be calculated as follows:
```math
\mathbb{E}_{(x_{loc}, y_{loc})\sim L(x_e)} \mathbb{1} \{ f_{\theta_e}(x_{loc}) = y_{loc} \}
```
