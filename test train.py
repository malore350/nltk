import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from pypdf import PdfReader
import re

# Derive the main text for testing
# Read in the research paper data
pdf_file = open(r'C:\Users\kamra\DataspellProjects\nltk_test\file2.pdf', 'rb')

# Create a PDF reader object
pdf_reader = PdfReader(pdf_file)
num_pages = len(pdf_reader.pages)

# Loop through all the pages in the PDF document and extract the text
text = ""
for i in range(num_pages):
    # Get the page object
    page = pdf_reader.pages[i]

    # Extract the text from the page
    page_text = page.extract_text()

    # Append the text to the document text
    text += page_text

# Close the PDF file
pdf_file.close()


# Define a regular expression to match the word "References" and everything that follows it
references_pattern = re.compile(r'References(.*)', re.DOTALL)

# Use the regular expression to remove the text after "References"
text = references_pattern.sub('', text)

# Define a regular expression to match equations
equations_pattern = re.compile(r'.*?=.+')


updated_text = re.sub(r".*Abstract[^:]+:", "", text, flags=re.DOTALL)






# Load the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('./bert-large-tokenizer')


# Load the saved model
model = BartForConditionalGeneration.from_pretrained("model")  # replace "model" with the path to your .bin file if necessary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Test the model on a string
input_text = updated_text
input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=1024)
summary_ids = model.generate(input_ids, max_length=512, num_beams=6)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"Summary text: {summary_text}")