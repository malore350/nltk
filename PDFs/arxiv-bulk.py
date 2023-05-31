import requests
import xml.etree.ElementTree as ET
import os
import re
import uuid

script_directory = os.path.dirname(os.path.abspath(__file__))

api_url = 'http://export.arxiv.org/api/query'
category = 'cs.AI'
max_results = 100

params = {
    'search_query': f'cat:{category}',
    'max_results': max_results
}

response = requests.get(api_url, params=params)
xml_content = response.content

root = ET.fromstring(xml_content)
entries = root.findall('{http://www.w3.org/2005/Atom}entry')

os.chdir(script_directory)

# Download the PDF files
for entry in entries:
    pdf_link = entry.find('{http://www.w3.org/2005/Atom}link[@title="pdf"]').attrib['href']
    pdf_response = requests.get(pdf_link)

    # Generate a unique filename for each downloaded PDF
    filename = str(uuid.uuid4()) + '.pdf'

    with open(filename, 'wb') as f:
        f.write(pdf_response.content)
        print(f'Downloaded: {filename}')


