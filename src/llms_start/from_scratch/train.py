
import tiktoken

from llms_start.from_scratch import small_model
from llms_start.from_scratch.data_loader import create_dataloader_v1
from llms_start.from_scratch.data_processing import generate_text_simple, text_to_token_ids, token_ids_to_text

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

start_model = small_model.SmallModel(GPT_CONFIG_124M)
print("model architecture: {}".format(start_model))

print("starting training: vocab size: {}".format(GPT_CONFIG_124M["vocab_size"]))

file_path = "examples/the-verdict.txt"  # Path to the text file
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

print("Text data loaded from {}: {}".format(file_path, text_data[:100]))  # Print first 100 characters

dataloader = create_dataloader_v1(text_data, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)