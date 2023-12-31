import torch
from glassonion.models.gpt2 import GPT2LMModel, GPT2Config, GPT2Tokenizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Example input text
    text = "This is an example sentence."

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids).to(device)
    # Forward pass up to layer 0, modify the output, and continue
    output_1 = model(
        input_ids,
        intervene_at_layer=0,
        attention_mask=attention_mask,
        intervene_at_position=0,
        return_dict=True,
    ).logits
    output_2 = model(
        input_ids,
        intervene_at_layer=None,
        attention_mask=attention_mask,
        intervene_at_position=0,
        return_dict=True,
    ).logits

    print(torch.allclose(output_1, output_2))