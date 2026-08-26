# CapybaraAI

A lightweight, self-learning conversational chatbot built in PyTorch. It extracts facts, user preferences, and question-answer pairs directly from chat and retrains its neural network on-the-fly.

---

## Features

- **Conversational Knowledge Extraction**: Automatically parses user profile details (*name, location, favorites*) and factual statements (*"X is Y"*, *"X was created by Y"*) from natural conversation.
- **On-the-Fly Neural Retraining**: Dynamically updates its vocabulary and retrains a deep PyTorch neural network in milliseconds when new knowledge is learned.
- **Active Learning**: When asked an unknown question, it asks for the answer and permanently commits your explanation to memory.
- **Persistent Storage**: Saves all learned knowledge and profile attributes to `knowledge.json`.

---

## Quickstart

### Prerequisites
- Python 3.9+
- PyTorch

```bash
pip install torch
```

### Run the Chatbot
```bash
python main.py
```

---

## How It Learns

### 1. User Profile & Preferences
```text
You: My name is Ravi and I live in Tokyo
  [Brain Update] Learned user name: Ravi
  [Brain Update] Learned user location: Tokyo

You: What is my name?
Bot: Your name is Ravi!
```

### 2. Facts & Definitions
```text
You: Python was created by Guido van Rossum
  [Brain Update] Learned fact: Python -> created by Guido van Rossum

You: Who created Python?
Bot: Python was created by Guido van Rossum.
```

### 3. Active Learning (Teaching Unknown Queries)
```text
You: What is PyTorch?
Bot: I haven't learned about that yet (confidence: 42.1%). What is the correct answer?
You: PyTorch is an open-source machine learning framework created by Meta.
  [Learning] Retraining neural brain on your answer... Done!

You: What is PyTorch?
Bot: PyTorch is an open-source machine learning framework created by Meta.
```

---

## Architecture Overview

```
User Input
    │
    ├──► Pattern Matcher & Knowledge Extractor
    │         └── Extracts entities, facts, and profile data
    │
    ├──► Tokenizer & Bag-of-Words Vectorizer
    │
    ├──► PyTorch Neural Classifier (ChatbotBrain)
    │         ├── Linear(vocab_size, 128) + ReLU + Dropout(0.15)
    │         ├── Linear(128, 128) + ReLU + Dropout(0.15)
    │         └── Linear(128, num_intents)
    │
    └──► Knowledge Store (`knowledge.json`)
              └── Persists intents, patterns, and facts across runs
```

---

## Project Structure

```
.
├── main.py            # Neural chatbot, knowledge extraction & interactive CLI
├── knowledge.json     # Persistent database for learned facts & user profiles
└── README.md
```

---

## License

MIT
