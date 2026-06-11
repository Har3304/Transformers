# Website RAG Chatbot using Qwen 2.5 and Live Website Data

## Overview

This project demonstrates a lightweight Retrieval-Augmented Generation (RAG) chatbot built using a pretrained Qwen 2.5 language model and live website content.

Instead of fine-tuning a model, the chatbot dynamically scrapes website data, constructs a temporary knowledge base, and injects that information directly into the model's context during inference. This allows the chatbot to answer company-specific questions using information extracted from the website.

The implementation provides a simple and effective approach for creating domain-specific chatbots without additional model training.

---

## Features

* Automatic website content extraction
* Sitemap discovery and URL crawling
* Dynamic knowledge base generation
* Retrieval-Augmented Generation (RAG) style prompting
* Qwen 2.5 Instruct model integration
* No model training required
* Company-specific question answering
* GPU acceleration support via PyTorch

---

## How It Works

### Step 1: Website Crawling

The script:

1. Accesses the target website.
2. Extracts sitemap URLs.
3. Retrieves page URLs from sitemap files.
4. Downloads selected webpages.
5. Extracts paragraph content.

Example:

```text
Homepage
    ↓
Sitemap
    ↓
Website URLs
    ↓
Page Content
    ↓
Knowledge Base
```

---

### Step 2: Knowledge Base Construction

All extracted text is combined into a single knowledge base.

Example:

```text
--- Source: https://example.com/about-us ---
Company information...

--- Source: https://example.com/services ---
Service information...
```

This knowledge base becomes the source of truth used by the chatbot.

---

### Step 3: Load the Language Model

The project uses:

Model:

Qwen/Qwen2.5-1.5B-Instruct

Benefits:

* Instruction tuned
* Lightweight compared to larger LLMs
* Good reasoning capabilities
* Efficient inference on consumer GPUs

---

### Step 4: RAG-style Inference

The website data is injected directly into the system prompt.

```text
System Prompt
      +
Knowledge Base
      +
User Question
      ↓
Qwen 2.5
      ↓
Response
```

The model is instructed to:

* Answer only using website data
* Avoid hallucinations
* Return "I cannot find that in the website data" when information is unavailable

---

## Example Questions

```python
ask_chatbot("What is the name of this company?")
ask_chatbot("What does this company do?")
ask_chatbot("What are the contact details of the company?")
ask_chatbot("What is the company address?")
```

Example output:

```text
Question:
What does this company do?

Generated Output:
The company provides solar energy solutions including installation and maintenance services.
```

---

## Project Structure

```text
.
├── chatbot.py
├── requirements.txt
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute:

```bash
python chatbot.py
```

The script will:

1. Crawl the website
2. Build the knowledge base
3. Load the Qwen model
4. Answer predefined questions

---

## Customization

To use another website:

```python
response = requests.get("https://yourwebsite.com")
```

Replace the URL with the target website.

You may also modify:

```python
link_map[:10]
```

to crawl more pages.

Example:

```python
link_map[:50]
```

for a larger knowledge base.

---

## Limitations

* Large websites may exceed the model context window.
* Content is loaded into memory at runtime.
* No vector database is used.
* No semantic retrieval is performed.
* Performance depends on website quality and available content.

---

## Future Improvements

* FAISS vector database integration
* Embedding-based retrieval
* FastAPI backend
* Web chat interface
* Conversation memory
* Multi-website support
* Streaming responses
* Automatic website updates

---

## Technologies Used

* Python
* PyTorch
* Transformers
* Qwen 2.5
* Requests
* BeautifulSoup4

---

## Author

Harnish Gajjar

AI/ML Developer focused on Natural Language Processing, Large Language Models, Retrieval-Augmented Generation (RAG), and Machine Learning solutions.
