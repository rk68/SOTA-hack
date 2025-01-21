# Hackathon Submission

Welcome to the **Neuroscope** hackathon submission! This repository showcases a notebook demonstrating **dynamic text augmentation** and **iterative text rewriting** driven by *neuron-level activations* in a language model. 

We combine:
- An **interpretability model** (a [Transformers-NeuronLens / EasyTransformer-like model](https://github.com/neelnanda-io/TransformerLens)) that we can query for per-token *neuron* activations.
- A **masked language model** (e.g. DistilBERT) for single-token replacements.
- A **generative model** (placeholder: `google/gemma-2b`) for full-sentence or multi-token rewrites.

Our goal is to discover text edits or rewrites that *increase* or *decrease* activation of a specific neuron, using the techniques of:
1. **Pruning**: Finding a minimal substring that still elicits high activation from the target neuron.
2. **Augmentation**: Iteratively modifying tokens via an MLM (or by generating rewrites with a generative model) to see how the neuron's activation changes.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Examples](#examples)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

In this notebook:
1. We **load** a hooking/interpretability model (which we refer to as `model`) that supports:
   - `model.to_tokens(text)`: to tokenize a string.
   - `model.run_with_cache(tokens)`: forward pass returning `(logits, cache)`, where `cache` lets us extract internal activations at any layer or neuron.
2. We **prune** large text snippets to isolate minimal triggers. 
   - We measure the target neuron’s activation on each substring until we find a short sequence that still yields high activation.
3. We **iteratively augment** the pruned text, performing token replacements with a masked language model (MLM) or generating rewrites with a generative model. 
4. We classify each new variant as having *increased*, *neutral*, or *decreased* activation based on thresholds.

The high-level objective is to:
- Identify the textual elements that matter for a specific *neuron*’s activation.
- Generate additional positive or negative examples that confirm or deny that neuron’s learned triggers.

---

## Features

- **Neuron-Specific Pruning**  
  Function `prune()` attempts to remove unneeded tokens while keeping a high neuron activation.

- **Iterative MLM-Based Augmentation**  
  Class `IterativeMultiTokenAugmenter` replaces tokens one by one using a masked language model (e.g. DistilBERT). It measures the new activation each time and classifies the augmented text as positive or negative relative to the neuron’s original activation.

- **Generative Rewrites**  
  Function `generative_augment()` prompts a generative model (like GPT) to rewrite the snippet, then measures each rewrite’s neuron activation.

- **Activation Thresholding**  
  We define thresholds such as `inclusion_threshold` and `exclusion_threshold`. If the new text’s activation drop is above/below these, we classify it as a *positive* or *negative* rewrite.

---

## Usage

1. **Install Dependencies**  
   In addition to standard libraries (`torch`, `numpy`, `tqdm`, etc.), you need:
   ```bash
   pip install transformer_lens
   pip install transformers
   pip install huggingface_hub
   ```
2. **Download / Set Up Models**  
   - A hooking/interpretability model (like [TransformerLens models](https://github.com/neelnanda-io/TransformerLens)).  
   - A masked language model (e.g. `distilbert-base-uncased`).  
   - (Optional) A generative model (e.g. `google/gemma-2b` placeholder) for rewriting.

3. **Run the Notebook**  
   - Load your hooking model as `model`.
   - Specify the layer and neuron to analyze (`layer_name`, `neuron_index`).
   - Provide text snippets you want to prune and augment.
   - Call the augmentation/pruning functions to observe the results.

4. **Interpret Results**  
   - Check which tokens or rewrites *increase* or *decrease* neuron activation.
   - See minimal triggers (from pruning) that keep activation high.

---

## Code Explanation

Below is a **high-level** overview of the core classes and functions:

### 1. `prune(model, layer, neuron, prompt, ...)`
- **Purpose**: Find a substring of the prompt that still triggers the neuron’s maximum activation (above a certain proportion of the original).
- **Method**:
  1. Compute initial activation + identify the token of maximum activation.
  2. Iteratively remove tokens/sentences while ensuring the max activation remains fairly close to the original.
- **Parameters**:
  - `model`: hooking model with `.run_with_cache`.
  - `layer`, `neuron`: layer name and neuron index to track.
  - `proportion_threshold`: how much activation can drop before we stop pruning.
  - `window`: size of the textual window around the max-activation token we try removing.

### 2. `ContextualAugmenter`
- **Purpose**: Use a masked LM to replace *one token at a time* in a given prompt. 
- **Method**:
  1. Insert `[MASK]` for each token in sequence.
  2. Query the MLM for top candidates.
  3. Filter out duplicates or the same token. 
  4. Generate new text, measure activation change, classify as positive/negative.

### 3. `IterativeMultiTokenAugmenter`
- **Purpose**: A slightly refined approach for iterative MLM replacements. 
- **Key Steps**:
  - Uses `_measure_activation(text)` to compute the target neuron’s activation.
  - `_mask_and_replace(...)` to find top `[MASK]` replacements.
  - Replaces tokens sequentially up to a certain number of steps, each time re-checking activation.

### 4. `generative_augment(...)`
- **Purpose**: Generate rewrites via a generative model (e.g., GPT), measure neuron activation for each. 
- **Implementation**:
  - Takes `original_prompt` plus a rewrite-instruction prefix (e.g., `"Rewrite this text:"`).
  - Samples multiple completions from the generative model.
  - For each rewrite, measures max activation on the hooking model and classifies.

### 5. `run_iterative_multi_token_on_snippets(...)`
- **Purpose**: Example helper to loop over multiple snippets, run `prune` then run iterative augmentation, printing the final *positive* and *negative* lists.

---

## Examples

We provide several snippet arrays (`snippets_3_1`, `snippets_3_2`, etc.). Each snippet is pruned to a minimal substring that retains high neuron activation, and then iteratively augmented.

### Sample Workflow

1. **Pruning**:
   ```python
   pruned_prompt = prune(
       model, "blocks.3.mlp.hook_mid", 1, snippet,
       proportion_threshold=-0.5, window=3
   )
   ```

2. **Iterative MLM Augmentation**:
   ```python
   iterative_augmenter = IterativeMultiTokenAugmenter(
       mlm_model, mlm_tokenizer, model, "blocks.3.mlp.hook_mid", neuron_index=1
   )
   positive, negative = iterative_augmenter.iterative_augment(pruned_prompt, steps=3)
   ```

3. **Generative Rewriting**:
   ```python
   pos_gens, neg_gens = generative_augment(
       hooking_model=model,
       layer_name="blocks.3.mlp.hook_mid",
       neuron_index=1,
       original_prompt=pruned_prompt,
       n=4
   )
   ```

Check the console/log outputs for classification of each rewrite (with measured activation changes).

---

## License

This project is provided under the [MIT License](LICENSE). Note that any pretrained models (MLM or generative) have their own respective licenses per the Hugging Face Model Hub or original project.

---

## Acknowledgments

- **TransformerLens** by [Neel Nanda](https://github.com/neelnanda-io/TransformerLens) for the hooking/interpretability interface.
- **Hugging Face Transformers** for the masked LM and generative model APIs.
- The **hackathon organizers** for the prompt to explore advanced neuron-level analysis and text transformations.

---

**Happy neuron probing and text rewriting!** If you have questions or suggestions, feel free to open an issue or submit a pull request. Contributions are welcome.
