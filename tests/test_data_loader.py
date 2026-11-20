from pathlib import Path

import tiktoken
import torch

from llms_start.from_scratch.data_loader import create_dataloader_v1

def test_first_training_batch_contains_shifted_token_windows():
    corpus_path = Path(__file__).resolve().parents[1] / "examples" / "the-verdict.txt"
    text_data = corpus_path.read_text(encoding="utf-8")

    dataloader = create_dataloader_v1(
        text_data, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    inputs, targets = next(iter(dataloader))

    token_ids = tiktoken.get_encoding("gpt2").encode(
        text_data, allowed_special={"<|endoftext|>"}
    )
    torch.testing.assert_close(inputs, torch.tensor(token_ids[:32]).reshape(8, 4))
    torch.testing.assert_close(targets, torch.tensor(token_ids[1:33]).reshape(8, 4))
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)

