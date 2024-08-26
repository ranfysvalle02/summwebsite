import json
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import tiktoken

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Define constants
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = "" 
az_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,api_version="2023-07-01-preview",api_key=AZURE_OPENAI_API_KEY)

r = requests.get('https://arxiv.org/html/1706.03762v7')
# Parsing the HTML
soup = BeautifulSoup(r.content, 'html.parser')
RAW_HTML = soup.get_text()
# Break the text into chunks (adjust chunk size as needed)
chunk_size = 10000
# check status code for response received
# success code - 200
print(r)
print("total tokens: " + str(len(tokenizer.encode(RAW_HTML))))
chunks = [RAW_HTML[i:i+chunk_size] for i in range(0, len(RAW_HTML), chunk_size)]
print(len(chunks))
# Define function to summarize each chunk
def summarize_chunk(chunk):
    print(f"Starting to process chunk at index {chunks.index(chunk)}; chunk_size=" + str(len(tokenizer.encode(chunk))))
    msgs2send = [
        {"role": "system", "content": "You are a helpful assistant that summarizes the CONTENT of SCRAPED HTML."},
        {"role": "user", "content": """=
Your main objective is to summarize the content in such a way that the user does not have to visit the website.
Please provide a bullet list (-) summary to the best of your ability of the CONTENT in this WEBSITE:

IMPORTANT: PAY CLOSE ATTENTION TO THE CONTENT OF THE WEBSITE. IGNORE THINGS LIKE NAVIGATION, ADS, ETC.
REMEMBER! YOUR GOAL IS TO SUMMARIZE THE CONTENT OF THE WEBSITE IN A WAY THAT THE USER DOES NOT HAVE TO VISIT THE WEBSITE.
[web_content]
         
"""},
        {"role": "user", "content": str(chunk)},
        {"role": "user", "content": """
[response format]
should be in bullet list, valid markdown format. IMPORTANT! must be valid markdown string!
Max. 3 sentences per bullet point.
Max. 10 bullet points.
Min. 5 bullet points.
[sample_response]

- bullet_point_1

- bullet_point_2

...
         
- bullet_point_n


"""}
    ]
    ai_response = az_client.chat.completions.create(
        model="gpt-35-turbo",
        messages=msgs2send,
    )
    tmp_sum = str(ai_response.choices[0].message.content.strip())
    print(f"Finished processing chunk at index {chunks.index(chunk)}")
    return tmp_sum

# Use ThreadPoolExecutor to parallelize the summarization
with concurrent.futures.ThreadPoolExecutor() as executor:
    summaries = list(executor.map(summarize_chunk, chunks))

# Summarize the summaries
summary_of_summaries = az_client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role":"system", "content":"You are a helpful assistant who can strategically summarize multiple summaries together into one coherent summary."},
        {"role": "user", "content": "[summaries]"},
        {"role": "user", "content": str(summaries)},
        {"role": "user", "content": """
[response format]
Summary should be in bullet list, valid markdown format. IMPORTANT! must be valid markdown string!
Max. 3 sentences per bullet point.
Max. 10 bullet points.
Min. 5 bullet points.
[sample_response]

- bullet_point_1

- bullet_point_2

...
         
- bullet_point_n
         
[task]
    - Please provide a complete, comprehensive summary using the individual summaries provided in [summaries]
"""}
    ],
)
print("summary_of_summaries:")
print(
    summary_of_summaries.choices[0].message.content.strip()
)
