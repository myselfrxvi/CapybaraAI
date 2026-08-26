import os
import re
import json
import random
import torch
import torch.nn as nn

KNOWLEDGE_FILE = "knowledge.json"

DEFAULT_KNOWLEDGE = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "whats up", "greetings", "hhiii", "wsp", "wassup", "yo"],
        "responses": ["Hello! How can I help you?", "Hey there!", "Hi! What are we working on?"]
    },
    "how_are_you": {
        "patterns": ["how are you", "how are you doing", "how is it going", "hows it going", "are you ok", "hows life"],
        "responses": ["I'm doing great and constantly learning new things!", "I'm doing well, thank you! How are you?"]
    },
    "acknowledgement": {
        "patterns": ["i know", "ok", "okay", "cool", "nice", "got it", "understood", "makes sense", "great", "okayy", "alright"],
        "responses": ["Glad to hear!", "Awesome!", "Sounds good! Let me know if you want to teach me anything else."]
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "thx", "appreciate it"],
        "responses": ["You're very welcome!", "Anytime! Happy to help."]
    },
    "goodbye": {
        "patterns": ["bye", "see you", "goodbye", "exit", "quit"],
        "responses": ["Goodbye! Have a great day!", "See you later!", "Bye!"]
    },
    "identity": {
        "patterns": ["who are you", "what is your name", "what are you"],
        "responses": ["I am your custom PyTorch neural chatbot!", "I'm an adaptive self-learning AI built by you!"]
    },
    "python": {
        "patterns": ["tell me about python", "do you know python", "what is python", "who created python"],
        "responses": ["Python is the world's most popular language for AI and Data Science created by Guido van Rossum!"]
    }
}

user_memory = {}

def load_knowledge():
    global user_memory
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                kb = saved.get("knowledge", DEFAULT_KNOWLEDGE)
                user_memory = saved.get("user_memory", {})
                return kb
        except Exception:
            return DEFAULT_KNOWLEDGE
    return DEFAULT_KNOWLEDGE

def save_knowledge(knowledge_base):
    try:
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump({"knowledge": knowledge_base, "user_memory": user_memory}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Failed to save knowledge: {e}")

def build_vocab(knowledge_base):
    all_words = []
    tags = list(knowledge_base.keys())
    for tag, data in knowledge_base.items():
        for pattern in data["patterns"]:
            words = pattern.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").split()
            all_words.extend(words)
    vocab = sorted(list(set(all_words)))
    return vocab, tags

def text_to_bow(sentence: str, vocabulary: list) -> torch.Tensor:
    words = sentence.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").split()
    bow = [1.0 if w in words else 0.0 for w in vocabulary]
    return torch.tensor([bow], dtype=torch.float32)

class ChatbotBrain(nn.Module):
    def __init__(self, vocab_size: int, num_intents: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_intents)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def train_model(vocab, tags, knowledge_base):
    X_train = []
    y_train = []
    
    for tag_idx, (tag, data) in enumerate(knowledge_base.items()):
        for pattern in data["patterns"]:
            bow = text_to_bow(pattern, vocab).squeeze(0)
            X_train.append(bow)
            y_train.append(tag_idx)
            
    X_tensor = torch.stack(X_train)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    
    model = ChatbotBrain(len(vocab), len(tags), hidden_dim=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model

def extract_knowledge_from_conversation(text: str, knowledge_base: dict) -> list:
    learned_items = []
    text_clean = text.strip()
    
    is_question = text_clean.endswith("?") or any(
        text_clean.lower().startswith(w) for w in ["what", "who", "where", "why", "how", "when", "is it", "can you", "tell me"]
    )

    profile_patterns = [
        (r"\bmy name is ([A-Za-z0-9_'\s\-]+?)(?:\.|$|,|\band\b)", "name", "Your name is {}!"),
        (r"\bi live in ([A-Za-z0-9_'\s\-]+?)(?:\.|$|,)", "location", "You live in {}."),
        (r"\bmy favorite (\w+) is ([A-Za-z0-9_'\s\-]+?)(?:\.|$|,)", "favorite_{}", "Your favorite {0} is {1}."),
        (r"\bi work as an? ([A-Za-z0-9_'\s\-]+?)(?:\.|$|,)", "job", "You work as a {}."),
        (r"\bi like ([A-Za-z0-9_'\s\-]+?)(?:\.|$|,)", "likes", "You mentioned that you like {}."),
        (r"\bi love ([A-Za-z0-9_'\s\-]+?)(?:\.|$|,)", "likes", "You mentioned that you love {}.")
    ]

    for pat, key, resp_template in profile_patterns:
        m = re.search(pat, text_clean, re.IGNORECASE)
        if m:
            if "{}" in key:
                prop = m.group(1).strip().lower()
                val = m.group(2).strip()
                tag = f"user_fav_{prop}"
                user_memory[f"favorite_{prop}"] = val
                reply = resp_template.format(prop, val)
                query_patterns = [
                    f"what is my favorite {prop}",
                    f"do you know my favorite {prop}",
                    f"my favorite {prop}"
                ]
            else:
                val = m.group(1).strip()
                if val.lower() in ["a", "an", "the", "fine", "ready", "happy", "sure", "thinking"]:
                    continue
                tag = f"user_{key}"
                user_memory[key] = val
                reply = resp_template.format(val)
                query_patterns = [
                    f"what is my {key}",
                    f"who am i" if key == "name" else f"where do i live" if key == "location" else f"what is my {key}",
                    f"do you know my {key}"
                ]

            knowledge_base[tag] = {
                "patterns": query_patterns,
                "responses": [reply]
            }
            learned_items.append(f"Learned user {key}: {val}")

    if not is_question and len(text_clean) > 8 and not learned_items:
        fact_patterns = [
            (r"^([A-Z][A-Za-z0-9\s\-]{1,35})\s+(is an?|is the|is|are|was|were)\s+([A-Za-z0-9\s\,\-\'\(\)]+)$", "definition"),
            (r"^([A-Za-z0-9\s\-]{2,30})\s+(was created by|was invented by|was developed by)\s+(.+)$", "creator"),
            (r"^the capital of ([A-Za-z0-9\s\-]+?)\s+is\s+([A-Za-z0-9\s\-]+)$", "capital"),
            (r"^(?:remember|note|learn)(?:\s+that)?\s+(.+?)\s+(is|are|means|equals)\s+(.+)$", "explicit")
        ]

        for pat, kind in fact_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                if kind == "capital":
                    subj = m.group(1).strip()
                    obj = m.group(2).strip().rstrip(".")
                    tag = f"fact_capital_{subj.lower().replace(' ', '_')}"
                    q_patterns = [f"what is the capital of {subj}", f"capital of {subj}", f"where is the capital of {subj}"]
                    reply = f"The capital of {subj} is {obj}."
                elif kind == "creator":
                    subj = m.group(1).strip()
                    verb = m.group(2).strip()
                    obj = m.group(3).strip().rstrip(".")
                    tag = f"fact_{subj.lower().replace(' ', '_')}"
                    q_patterns = [f"who {verb.replace('was ', '')} {subj}", f"who created {subj}", f"tell me about {subj}"]
                    reply = f"{subj} {verb} {obj}."
                elif kind == "explicit":
                    subj = m.group(1).strip()
                    pred = m.group(2).strip()
                    obj = m.group(3).strip().rstrip(".")
                    tag = f"fact_{subj.lower().replace(' ', '_')}"
                    q_patterns = [f"what is {subj}", f"what does {subj} mean", f"tell me about {subj}"]
                    reply = f"{subj} {pred} {obj}."
                else:
                    subj = m.group(1).strip()
                    pred = m.group(2).strip()
                    obj = m.group(3).strip().rstrip(".")
                    if subj.lower() in ["it", "this", "that", "there", "what", "something", "you", "i", "we", "they", "my name", "my favorite", "i live"]:
                        continue
                    tag = f"fact_{subj.lower().replace(' ', '_')}"
                    q_patterns = [f"what is {subj}", f"who is {subj}", f"tell me about {subj}", f"what are {subj}"]
                    reply = f"{subj} {pred} {obj}."

                knowledge_base[tag] = {
                    "patterns": q_patterns,
                    "responses": [reply]
                }
                learned_items.append(f"Learned fact: {subj} -> {obj}")
                break

    return learned_items

def chat_loop():
    knowledge = load_knowledge()
    vocab, tags = build_vocab(knowledge)

    print("[System] Training neural chatbot brain...", end="", flush=True)
    model = train_model(vocab, tags, knowledge)
    print(" Ready!\n")

    print("=" * 65)
    print("  SELF-LEARNING NEURAL CHATBOT (PyTorch AI)")
    print("  * Talks with you using its neural intent classifier.")
    print("  * Automatically grabs knowledge & facts from conversations.")
    print("  * Retrains its neural brain dynamically upon learning.")
    print("=" * 65)

    last_unanswered_query = None

    while True:
        try:
            user_msg = input("\nYou: ").strip()
            if not user_msg:
                continue
                
            if user_msg.lower() in ["quit", "exit", "bye"]:
                print(f"Bot: {random.choice(knowledge['goodbye']['responses'])}")
                break

            if last_unanswered_query:
                if not user_msg.endswith("?") and len(user_msg) > 1:
                    new_tag = f"learned_{last_unanswered_query.lower().replace(' ', '_')[:25]}"
                    knowledge[new_tag] = {
                        "patterns": [last_unanswered_query, f"tell me about {last_unanswered_query}"],
                        "responses": [user_msg]
                    }
                    save_knowledge(knowledge)
                    vocab, tags = build_vocab(knowledge)
                    print("[Learning] Retraining neural brain on your answer...", end="", flush=True)
                    model = train_model(vocab, tags, knowledge)
                    print(" Done!")
                    
                    print(f"Bot: Got it! When asked '{last_unanswered_query}', I will now answer: \"{user_msg}\"")
                    last_unanswered_query = None
                    continue
                else:
                    last_unanswered_query = None

            learned = extract_knowledge_from_conversation(user_msg, knowledge)
            if learned:
                save_knowledge(knowledge)
                vocab, tags = build_vocab(knowledge)
                model = train_model(vocab, tags, knowledge)
                for item in learned:
                    print(f"  [Brain Update] {item}")

            user_tensor = text_to_bow(user_msg, vocab)
            num_matched_words = user_tensor.sum().item()
            
            with torch.no_grad():
                output = model(user_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
                conf_val = confidence.item()

            if num_matched_words > 0 and conf_val > 0.60:
                intent_tag = tags[predicted_idx.item()]
                bot_reply = random.choice(knowledge[intent_tag]["responses"])
                print(f"Bot ({intent_tag} - {conf_val*100:.1f}% confidence): {bot_reply}")
            else:
                is_q = user_msg.endswith("?") or any(user_msg.lower().startswith(w) for w in ["what", "who", "where", "why", "how", "when", "tell me"])
                if is_q:
                    last_unanswered_query = user_msg
                    print(f"Bot: I haven't learned about that yet (confidence: {conf_val*100:.1f}%). What is the correct answer?")
                else:
                    print(f"Bot: Got it! Tell me more or teach me something new.")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

if __name__ == "__main__":
    chat_loop()
