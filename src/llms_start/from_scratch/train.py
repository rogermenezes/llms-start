
import tiktoken

from llms_start.from_scratch import small_model
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

small_text = "This is a small text corpus for training a GPT model from scratch."
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = text_to_token_ids(small_text, tokenizer)  # Convert text to token IDs
print("Token IDs: {}".format(token_ids))

decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())  # Convert token IDs back to text
print("Decoded text: {}".format(decoded_text))

start_model.eval()  # Set the model to evaluation mode

token_ids = generate_text_simple(
    model=start_model,
    idx=text_to_token_ids(small_text, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
