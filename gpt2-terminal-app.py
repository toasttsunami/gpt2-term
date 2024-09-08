import torch
import torch.nn.functional as F
from simplified_gpt2 import SimplifiedGPT2

# Hyperparameters
VOCAB_SIZE = 256  # Using ASCII characters
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
MAX_SEQ_LEN = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Character Level tokenizer
def tokenize(text):
    return [ord(c) for c in text]


def detokenize(tokens):
    return "".join([chr(t) for t in tokens])


# Load and preprocess data
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def create_datasets(text, seq_len):
    data = torch.tensor(tokenize(text), dtype=torch.long)
    n = len(data)
    X = torch.stack([data[i : i + seq_len] for i in range(n - seq_len)])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in range(n - seq_len)])
    return X, y


# Training loop
def train(model, X, y, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(num_epochs):
        for i in range(0, len(X), BATCH_SIZE):
            batch_X = X[i : i + BATCH_SIZE].to(device)
            batch_y = y[i : i + BATCH_SIZE].to(device)

            logits = model(batch_X)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


# Text generation
def generate_text(model, start_text, max_length=100):
    model.eval()
    tokens = tokenize(start_text)
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor(tokens[-MAX_SEQ_LEN:]).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze().item()
            tokens.append(next_token)
            if next_token == ord("\n"):
                break
    return detokenize(tokens)


# Main application
def main():
    # Load and preprocess data
    text = load_data("your_text_file.txt")  # Replace with your text file
    X, y = create_datasets(text, MAX_SEQ_LEN)

    # Initialize and train the model
    model = SimplifiedGPT2(
        VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN
    ).to(device)
    train(model, X, y, NUM_EPOCHS)

    # Interactive loop
    while True:
        prompt = input("Enter a prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
        generated_text = generate_text(model, prompt)
        print("Generated text:")
        print(generated_text)


if __name__ == "__main__":
    main()
