# llms-start

A learning project for training a small GPT from scratch, learning semantic IDs, and fine-tuning a Hugging Face language model for classification.

## Set up the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run this from the repository root:

```sh
uv sync --locked
```

uv uses Python 3.12 and installs the project into `.venv`. The default PyTorch installation uses CPU wheels on Linux and Windows. To include the Hugging Face dependencies, run `uv sync --locked --extra classification`.

## Run in VS Code

1. Open the repository folder and install the Microsoft Python and Python Debugger extensions.
2. Run **Python: Select Interpreter** and select the interpreter in the project’s `.venv`. On Linux and macOS it is `.venv/bin/python`. On Windows it is `.venv\Scripts\python.exe`.
3. Once your training code is in place, select **Train tiny GPT** in **Run and Debug** and press **F5**.

## Add the experiments

- **From scratch** — Put your book-based dataloader, `GPTModel`, and training code in `src/llms_start/from_scratch/`. The launch profile expects a runnable `train.py` in that package. Use the book’s `tiktoken` encoding, and use `examples/tiny_corpus.txt` as a sample input. The VS Code profile starts in the project root, so relative input paths resolve from there. Add training arguments to the profile once the script’s interface is known.
- **Semantic IDs** — Put the item embedding and quantization code in `src/llms_start/semantic_ids/`.
- **Classification** — Put the Hugging Face fine-tuning code in `src/llms_start/classification/`.

Once the from-scratch training module is present, the equivalent terminal command is:

```sh
uv run --locked python -m llms_start.from_scratch.train
```

Use `data/` for downloaded data and `runs/` for metrics and checkpoints. Both directories are ignored by Git.
