# Paper Pool for Papers Related to Side-effects of Knowledge Editing

## General (Paul Huang)
- Unveiling the Pitfalls of Knowledge Editing for Large Language Models (https://arxiv.org/abs/2310.02129)
    - Problems:
        - Knowledge conflict:
            <br> In sequential edit scenarios, editing related knowledge may cause inconsistencies in model predictions, especially when the edits are contradicting.
        - Knowledge distortion:
            <br> Methods that modify the model parameters may harm the knowledge structure/general ability of LLM. For example, if we edit $a\rightarrow b$ then $b\rightarrow a$, the model would not perform the same even though we would expect it to perform the same as original model.
    - Discussed Editing Methods: Fine-tuning (FT), MEND, ROME, MEMIT
    - Benchmarks / Datasets:
        - ConflictEdit:
            - Task:
                - Reverse edit: Provide contradicting edits using reverse relations.
                - Composite edit: Knowledge contained in 2 or more edits need to be linked using knowledge contained in the original model to get the result.
            - Evaluation Metrics:
                <br> Let $P_{\theta^m}$ and $P_{\theta'}$ be the probability estimated by intermediate (after old edit but before new edit) and final (after old edits) model, respectively.
                - Conflict Score (CS):
                    <br> Suppose that new edit contradicts with old edit, we would expect the model to update its knowledge to adhere to the newer knowledge. This metric is designed to evaluate the percentage (success rate) that the edited LLM predicts newer edit to have higher probability than older edit.

```math
\mathrm{CS} = \mathbb{E}\mathbb{1}\left[P_{\theta'}(k_{\text{new}}) > P_{\theta'}(k_{\text{old}})\right]
```

                - Conflict Magnitude (CM):
                    <br> The idea is similar to CS, but instead of success rate, this metric aims to evaluate the decrease of probability of older knowledge after the new edit.

                    ```math
                    \mathrm{CM} = \frac{P_{\theta^m}(k_{\text{old}}) - P_{\theta^'}(k_{\text{old}})}{P_{\theta^m}(k_{\text{old}})}
                    ```

                - Tied Fact Damage (TFD):
                    <br> Similar to CM, but includes new terms for composite edits. The metric aims to evaluate how is the original knowledge (related to edits but should not be changed after edit) influenced by the edits. Ideally, the influence should be as small as possible.
            - Results:
                - MEND and MEMIT fails to achieve simple coverage evaluations.
                - Reverse edit: FT and MEND successfully erased older edit when new edit arrived; ROME and MEMIT completely failed.
                - Composite edit: FT and MEND perform well; ROME and MEMIT successfully edit the model, but does not generalize well to related stuff. TFD shows that most methods damages the knowledge structures in LLMs.
        - Round-Edit:
            - Task (Round-Edit):
                <br> Edit a knowledge and **un-edit** it by introducing a new edit that changes the knowledge back. Then, evaluate how is the knowledge structure (next token probability) of LLM influenced.
            - Metrics:
                - Distortion (D):
                    <br> Evaluates the JS divergence of the predicted distribution before and after knowledge editing.
                - Ignore rate (IR):
                    <br> Evaluates what percentage of knowledge related to the edit (set $\text{Obj}$) is discarded or overlooked.
                    
                    ```math
                    \mathrm{IR} = \sum_(o\in\text{Obj})\mathbb{E}(P_{\theta}(o)>P_{\theta'}(o))
                    ```
                    
                - Failure rate (FR):
                    <br> Evaluates the percentage of cases where IR is greater than $50%$.
            - Results:
                - Knowledge distortion is more significant in FT and MEND.
                - ROME and MEMIT showed high success rate in each edit of Round-Edit while minimally damaging the inner knowledge structure.
    - Proposed a multi-label edit (MLE) method to mitigate the knowledge distortion problem.
- Navigating the Dual Facets: A Comprehensive Evaluation of Sequential Memory Editing in Large Language Models (https://arxiv.org/abs/2402.11122)
- Editing Large Language Models: Problems, Methods, and Opportunities(https://arxiv.org/abs/2305.13172)
- Emptying the Ocean with a Spoon: Should We Edit Models? (https://arxiv.org/abs/2310.11958)

## Catastrophic forgetting (Hank)
- Model Editing at Scale leads to Gradual and Catastrophic Forgetting(https://arxiv.org/abs/2401.07453)
- An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning (https://arxiv.org/abs/2308.08747)
- Model Editing Can Hurt General Abilities of Large Language Models (https://arxiv.org/abs/2401.04700)(downstream evaluation)
- The Butterfly Effect of Model Editing: Few Edits Can Trigger Large Language Models Collapse (https://arxiv.org/abs/2402.09656)

## Ripple effect (Hung-Chieh)
- Propagation and Pitfalls: Reasoning-based Assessment of Knowledge Editing through Counterfactual Tasks (https://arxiv.org/abs/2401.17585)
- MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions
- Evaluating the Ripple Effects of Knowledge Editing in Language Models (https://arxiv.org/abs/2307.12976)
- Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models (https://arxiv.org/abs/2401.10647)
- Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge (https://arxiv.org/abs/2305.01651)

## Bad specificity/locality (Liao)
- Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark (https://aclanthology.org/2023.findings-acl.733/)
- Is it Possible to Edit Large Language Models Robustly? (https://arxiv.org/abs/2402.05827)

## Mismatch between success rate and current findings (Jim)
- Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models (https://arxiv.org/abs/2301.04213)
- What does the Knowledge Neuron Thesis Have to do with Knowledge? (https://openreview.net/forum?id=2HJRwwbV3G)
- Journey to the Center of the Knowledge Neurons: Discoveries of Language-Independent Knowledge Neurons and Degenerate Knowledge Neurons (https://arxiv.org/abs/2308.13198)(unsure)


## Others (Liao)
- Large Language Models Relearn Removed Concepts (https://arxiv.org/abs/2401.01814)
