import re
from pypdf import PdfReader
import json
import os

# Specify the folder path containing the PDF files
folder_path = r"C:\Users\kamra\DataspellProjects\nltk_test\PDFs"

# Get the list of PDF files in the folder
pdf_files = [file for file in sorted(os.listdir(folder_path)) if file.endswith('.pdf')]
# Iterate through all the PDF files in the folder
for pdf_file_name in pdf_files:
    try:
        pdf_file_path = os.path.join(folder_path, pdf_file_name)

        # Read in the research paper data
        pdf_file = open(pdf_file_path, 'rb')

        # Create a PDF reader object
        pdf_reader = PdfReader(pdf_file, strict = False)
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
    except:
        print("error")
        continue


    # Define a regular expression to match the word "References" and everything that follows it
    references_pattern = re.compile(r'References(.*)', re.DOTALL)

    # Use the regular expression to remove the text after "References"
    text = references_pattern.sub('', text)

    # Define a regular expression to match equations
    equations_pattern = re.compile(r'.*?=.+')


    updated_text = re.sub(r".*Abstract[^:]+:", "", text, flags=re.DOTALL)

    # Create a dictionary to store the non-abstract text
    data = {
        "category": "non-abstract",
        "text": updated_text.strip()
    }

    # Save the non-abstract data to JSON file
    with open("data1.json", "a", encoding = 'utf8') as json_file:
        json.dump(data, json_file)
        json_file.write('\n')




    # NOW WE GO THROUGH THE ABSTRACTS SECTION
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


    # Use regular expressions to extract the content of the Abstract section
    abstract_match = re.search(r"Abstract\s*([^:]+)", text)

    if abstract_match:
        abstract_content = abstract_match.group(1).strip()
    else:
        print("neyise")
        # Read the JSON file
        with open('data1.json', 'r', encoding = 'utf8') as file:
            json_data = file.readlines()
        json_data = json_data[:-1]
        with open('data1.json', 'w', encoding = 'utf8') as file:
            file.writelines(json_data)


    # Create a dictionary to store the abstract text
    data = {
        "category": "abstract",
        "text": abstract_content
    }

    # Save the abstract data to JSON file
    with open("data1.json", "a", encoding = 'utf8') as json_file:
        json.dump(data, json_file)
        json_file.write('\n')
