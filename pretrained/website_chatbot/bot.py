# =====================================================================
# STEP 1: SCRAPE AND STORE THE DATA (No training needed!)
# =====================================================================
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Fetch sitemaps
response = requests.get('https://australianpremiumsolar.co.in')
soup = BeautifulSoup(response.content, 'xml')
sitemap_links = [l.text for l in soup.find_all('loc')]

link_map = []
for sitemap in sitemap_links:
    try:
        res = requests.get(sitemap)
        sub_soup = BeautifulSoup(res.content, 'xml')
        link_map.extend([u.text for u in sub_soup.find_all('loc')])
    except:
        pass

# 2. Extract text from the first few key pages to build our Knowledge Base
knowledge_base = ""
for page_url in link_map[:10]:
    try:
        res = requests.get(page_url)
        page_soup = BeautifulSoup(res.content, 'html.parser')
        paragraphs = [p.text.strip() for p in page_soup.find_all('p') if p.text.strip()]
        page_text = " ".join(paragraphs)
        if len(page_text.split()) > 10:
            knowledge_base += f"\n--- Source: {page_url} ---\n{page_text}\n"
    except:
        pass

print("Knowledge Base built successfully!")

# =====================================================================
# STEP 2: LOAD THE BASE CHAT MODEL
# =====================================================================
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# =====================================================================
# STEP 3: RAG INFERENCE (Injecting data directly into the prompt)
# =====================================================================
def ask_chatbot(question):
    # We pass the scraped website text directly into the system context!
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant for the company. Answer the user's question accurately using ONLY the provided company data below. If the answer cannot be found in the data, say 'I cannot find that in the website data.' Do not explain your reasoning.\n\nCOMPANY DATA:\n{knowledge_base}"
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100, # Increased slightly for complete answers
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.1 # Low temperature makes the model strictly stick to the text
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f'Question: {question}')
    print(f'Generated Output: {response.strip()}\n')

# Test the RAG chatbot
ask_chatbot("What is the name of this company?")
ask_chatbot("What does this company do?")
ask_chatbot("What are the contact details of the company?")
ask_chatbot("What is the company address?")
