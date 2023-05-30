import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import json

# Load the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, summary = self.data[index]
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512).squeeze(0).to(device)
        output_ids = tokenizer.encode(summary, return_tensors='pt', truncation=True, max_length=64).squeeze(0).to(device)
        return input_ids, output_ids

# Initialize empty lists for abstract and non-abstract texts
abstract_texts = []
non_abstract_texts = []

# Read the JSON file
with open("data.json", "r") as json_file:
    for line in json_file:
        # Parse each line as a JSON object
        data = json.loads(line)

        # Extract the category and text
        category = data.get("category")
        text = data.get("text")

        # Add the text to the corresponding list based on the category
        if category == "abstract":
            abstract_texts.append(text)
        elif category == "non-abstract":
            non_abstract_texts.append(text)

# Define the training data
train_data = []
min_len = min(len(abstract_texts), len(non_abstract_texts))
for i in range(min_len):
    train_data.append((non_abstract_texts[i], abstract_texts[i]))

# Define the training dataset and dataloader
train_dataset = TextDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Define the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=1e-5)

# Define the training loop
def train(epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (input_ids, output_ids) in enumerate(train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Generate the summary using the BART model
            summary_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)

            # Compute the loss and backpropagate
            loss = model(input_ids, labels=output_ids)[0]
            loss.backward()
            optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()

        # Print the average loss at the end of each epoch
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch}, Average Loss {avg_loss}')

# Train the model for 10 epochs
train(epochs=10)

# Save the trained model
model.save_pretrained("model")

# Load the saved model
model = BartForConditionalGeneration.from_pretrained("model")
model.to(device)

# Test the model on a string
input_text = 'updated_text'         # Replace with your own text, can be extracted from the PDF file
input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=1024).to(device)
summary_ids = model.generate(input_ids, max_length=1024, num_beams=6, early_stopping=True)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"Summary text: {summary_text}")
