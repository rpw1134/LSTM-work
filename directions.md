# MiniProject 4: Learning with Sequence Data (RNNs and Transformers)

Please read this entire document before beginning the assignment.

## Preamble

This mini-project is to be completed in groups of three. All members of a group will receive the same grade except when a group member is not responding or contributing to the project. If this is the case and there are major conflicts, please reach out to the assignment TA for help and flag this in the submitted report. It is not expected that all team members contribute equally, but every team member should make integral contributions, be aware of the content, and understand the full solution.

Submit your report (PDF) and code (zip file or a single notebook `.ipynb`) on MyCourses as a group. You must register your group on MyCourses, and any group member can submit.

We recommend using Overleaf for writing your report and Google Colab for coding and running experiments. You should use Python for this and subsequent mini-projects.

Please refer to the course LLM policy and make sure to precisely declare any use. All group members are accountable for this declaration.

## Background and Learning Objectives

The assignment aims to help you understand and implement sequence modeling techniques. By the end of this project, you should be familiar with:

- Training a sequence classifier using an LSTM in PyTorch.
- Fine-tuning a pretrained Transformer for a classification task.
- Understanding and correctly implementing the practical steps needed to feed text sequences into models, including tokenization and padding.

A key point of this project is that the preprocessing pipeline is naturally different in the two settings:

- When training an LSTM from scratch, you typically build your own vocabulary and map tokens to IDs yourself.
- When fine-tuning a pretrained Transformer, you must use the pretrained model's tokenizer, because the model was trained with a specific subword vocabulary and tokenization scheme.

This assignment is designed so that you learn both workflows.

## Dataset

You will use the AG News dataset (4-class news topic classification). It comes with an official train/test split.

- Training set: 120,000 examples
- Test set: 7,600 examples
- You will create a validation split from the training set.
- Use the label set as provided by the dataset.

Recommended access: Hugging Face Datasets (`ag_news`). The dataset provides the fields `text` and `label`.

## Reproducibility

Initialize `np.random.default_rng(RANDOM_SEED=2026)` and set the same seed value for PyTorch, Python `random`, and any DataLoader worker seeds you use.

### Practical note on runtime

We are not enforcing a strict runtime limit, but you should aim for experiments that run comfortably on a single GPU (e.g., Colab). If training is slow, you can shorten runtime by:

- reducing the maximum sequence length,
- using fewer epochs,
- using a smaller pretrained model (e.g., DistilBERT),
- using mixed precision (optional).

## A Short Primer: Tokenization and Padding

Neural models do not directly consume raw text. You must turn text into a sequence of integers.

### Tokenization

Tokenization breaks text into pieces called tokens and maps each token to an integer ID.

- **Word-level tokenization** (typical for LSTM-from-scratch): tokens are words or word-like units. You build a vocabulary from the training set and assign IDs.
- **Subword tokenization** (typical for Transformers): tokens are subword pieces. You must use the tokenizer that matches the pretrained model.

### Padding (Why It Exists)

Sequences have different lengths, but tensors in a batch must have the same shape. So you pad shorter sequences with a special PAD token up to the length of the longest sequence in the batch or a fixed max length.

- For LSTMs, you must ensure PAD tokens do not affect your representation (using sequence lengths or masking).
- For Transformers, you must provide an attention mask so attention ignores PAD tokens.

Suggested references (short and practical) are listed near the end of this handout.

## Task 1: LSTM Classifier Trained From Scratch

Your goal is to build a full text classification pipeline and train an LSTM on AG News.

### 1. Data Split

Starting from the official training set, create:

- a training subset and a validation subset (disjoint, using a 90/10 split),
- a fixed random seed,
- keep the official test split of 7,600 examples for final reporting.

### 2. Tokenization and Vocabulary (Word-Level)

Implement a word-level tokenizer and vocabulary builder.

Minimum expectations:

- Lowercasing and a reasonable tokenization scheme. You can write a simple regex (for example, split on whitespace and punctuation) or use an existing word-level tokenizer such as `nltk.word_tokenize` or `torchtext.data.utils.get_tokenizer("basic_english")`.
- To limit vocabulary size, consider only the words with a frequency higher than 2 in the training set. Words that appear only once rarely contain important information.
- Special tokens: at least `PAD` and `UNK` (for padding and unknown words). You can add more if you like (for example, `SOS`, `EOS`).
- Vocabulary is built using training text only, not validation or test text.
- Convert each example into a list of token IDs.
- Choose a maximum sequence length (recommended: 128). Truncate longer sequences.

### 3. Padding and LSTM Input Handling

Sequences in a batch have different lengths, but PyTorch tensors require a uniform shape. To handle this, you pad shorter sequences with the PAD token ID so that every sequence in the batch has the same length (either the longest sequence in the batch or the fixed maximum length if you are truncating).

For example, if a batch contains sequences of length 5, 3, and 7, all three would be padded to length 7.

The problem: an LSTM processes every time step it receives, including PAD tokens. If you naively take the final hidden state of a padded sequence, you get the hidden state after several meaningless PAD steps, not the state after the last real word. You must account for this.

Two standard approaches are acceptable:

- **Length-aware LSTM processing:** record each sequence's true length before padding. Use `pack_padded_sequence` before the LSTM and `pad_packed_sequence` after it so PyTorch skips PAD positions entirely during the forward pass.
- **Mask-aware pooling:** run the LSTM on the full padded sequences, but when computing the final representation (for example, mean pooling), use a boolean mask to exclude hidden states at PAD positions.

### 4. Model Architecture

A typical LSTM classifier looks like this:

1. `Embedding(vocab_size, d_emb)`
2. `LSTM(d_emb, d_hidden, num_layers=..., dropout=..., bidirectional=(True/False))`
3. A way to convert the sequence of hidden states into a single vector (see the pooling strategy below)
4. `Linear(d_hidden, 4)` to produce logits over 4 classes

#### What is a pooling strategy?

An LSTM produces a hidden state at every time step. Classification needs a single fixed-size vector.

Common choices:

- **Last hidden state (length-aware):** use the hidden state at the last real token, not PAD. With variable lengths, this means the last token before padding.
- **Mean pooling with a mask:** average hidden states over time, but only over real tokens and ignore PAD states.
- **Max pooling with a mask:** less common, but also acceptable.

Pick one strategy and explain it clearly. If you compare two strategies, that is a nice extension, but not necessary.

### 5. Training

Train with cross-entropy loss.

Suggested starting hyperparameters:

- max sequence length: 128
- embedding dimension: your choice
- hidden size: your choice
- number of layers: 1 (or 2 if stable)
- dropout: 0.1–0.3
- optimizer: Adam
- learning rate: your choice
- batch size: 64
- epochs: 6
- gradient clipping (recommended): clip global norm to 1.0

Track the following in each training epoch:

- training loss
- validation loss
- validation accuracy

### 6. Test Model

Evaluate the trained model on the test set.

- Compute and report test accuracy.
- Display several misclassified examples (for example, 5–10), including:
  1. input text
  2. true label
  3. predicted label

## Task 2: Fine-Tune a Pretrained Transformer

Now solve the same classification task using a pretrained Transformer encoder, and compare it with your LSTM.

### 1. Tokenization, Padding, Truncation, and Attention Masks

For Transformer fine-tuning, do not reuse your word-level tokenizer. Use the tokenizer that matches the pretrained model.

What you should understand and demonstrate in your write-up:

- Transformer tokenizers output token IDs for subwords, not words.
- They usually add special tokens (for example, `[CLS]` and `[SEP]` for BERT-like models).
- They handle unknown words by splitting into subwords rather than mapping everything to `UNK`.

Pick a short example sentence and show:

- your word-level tokens from Task 1,
- the Transformer tokenizer's token list (subwords),
- and briefly explain why they differ.

Transformers require:

- `input_ids` (token IDs),
- `attention_mask` (`1` for real tokens, `0` for PAD),
- truncation to a maximum length.

Set a Transformer maximum length (recommended: 128).

If you use a library `DataCollator`, you should still explain what padding and the attention mask are doing.

### 2. Load Model

Recommended model (fast, strong baseline):

- `distilbert-base-uncased` with a sequence classification head

Then print and show the number of model parameters.

### 3. Train Model

Suggested starting hyperparameters:

- max length: 128
- batch size: 16 (reduce if you run out of memory)
- epochs: 3
- learning rate: your choice
- optimizer: AdamW
- weight decay: 0.05
- linear warmup: optional

To accelerate training, you may enable mixed precision (`fp16=True`) and disable saving checkpoints (`save_strategy="no"`).

Track the following in each training epoch:

- training loss
- validation loss
- validation accuracy

### 4. Test Model

Evaluate the trained model on the test set.

- Compute and report test accuracy.
- Display several misclassified examples (for example, 5–10), including:
  1. input text
  2. true label
  3. predicted label

## Deliverables

Submit two files to MyCourses:

- `code.ipynb` or `code.zip`
  - If using a notebook, keep outputs and plots visible.
  - If using a zip, include a clear `README.md` with run instructions.
  - Ensure plots and reported numbers match the write-up.
- `writeup.pdf`
  - Maximum 5 pages, excluding references or appendix if you choose to include them.
  - Single-spaced, at least 11pt font, at least 0.5 inch margins.

### Important Note

Only the work that is clearly presented in both the written report and the Colab notebook will be considered for grading. Any additional results (for example, plots or experiments) that appear only in the Colab notebook but are not discussed in the report will not be graded. Please ensure that all relevant results are clearly included and explained in the report.

## Project Write-Up Structure

Your report must include the following sections:

### Abstract

100–250 words. Summarize the task, the two approaches (LSTM vs Transformer), and the main results.

### Introduction

At least 5 sentences. Describe the dataset and why sequence modeling choices matter (tokenization, padding, pretrained vs scratch).

### Methods

Include:

- **LSTM pipeline:** word-level tokenization, vocabulary construction, padding, model architecture, and how you produce a fixed-size vector from the sequence (pooling strategy).
- **Transformer pipeline:** pretrained tokenizer, attention masks, truncation length, model choice, and fine-tuning setup.
- **Training setup:** optimizer, learning rate, batch size, number of epochs, and any regularization or clipping.

### Results and Analysis

This section is the main section and should clearly report and compare the two tasks at a high level.

At minimum, include:

1. **Performance**
   - Validation accuracy for LSTM and Transformer (final or best)
   - Test accuracy for LSTM and Transformer

2. **Training behavior**
   - Learning curves for LSTM:
     - training loss vs. epoch
     - validation accuracy vs. epoch
   - Learning curves for Transformer:
     - training loss vs. epoch
     - validation loss vs. epoch
     - validation accuracy vs. epoch

3. **A direct comparison table**
   Include a small table comparing:
   - validation accuracy (final or best)
   - test accuracy
   - approximate number of parameters (order-of-magnitude is fine)
   - training time per epoch or total training time (rough timing is acceptable; report the environment such as Colab T4 or local GPU)

4. **Qualitative error analysis**
   Provide a short error analysis for each model:
   - show around 5–10 misclassified test examples per model,
   - for each, include a short excerpt, true label, and predicted label,
   - add 1–2 sentences summarizing patterns you notice (for example, ambiguity, named entities, long inputs, etc.).

### Discussion and Conclusion

At least 5 sentences. Reflect on:

- Where the LSTM struggled vs. the Transformer, and vice versa.
- What tokenization choice seems to buy you (word-level vs subword).
- Practical trade-offs: implementation complexity, training speed, memory.
- One concrete improvement you would try next, even if you did not run it.

### Statement of Contributions

1–3 sentences. Briefly describe how all team members contributed.

### Statement on the Use of LLMs and Other Resources

1–3 sentences. Explain how you used LLMs or other resources, if at all.

## Going Beyond the Minimum

You must demonstrate a level of creativity and go beyond the required items listed above for the full mark. This is true for both Task 1 and Task 2. Below are some examples of additional steps you might take, but you are encouraged to come up with your own ideas.

### For the LSTM

- Compare pooling strategies for the LSTM (last hidden vs. masked mean pooling).
- Train a GRU instead of, or in addition to, the LSTM while keeping everything else fixed.
- Compare LSTM with Bidirectional LSTM.

### For the Transformer

- Compare full fine-tuning vs. freezing the encoder and training only the classifier head.
- Explore a parameter-efficient fine-tuning method (for example, LoRA) using Hugging Face PEFT.
- Analyze the effect of maximum sequence length on accuracy vs. speed.

### ...
