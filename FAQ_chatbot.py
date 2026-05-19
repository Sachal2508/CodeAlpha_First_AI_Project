
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

# Download NLTK data (only runs once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ============================================================
# FAQ DATA — Questions and Answers
# (You can replace these with your own topic)
# ============================================================
faqs = [
    {"question": "What is artificial intelligence?",
     "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."},

    {"question": "What is machine learning?",
     "answer": "Machine Learning is a subset of AI where machines learn from data to improve their performance without being explicitly programmed."},

    {"question": "What is deep learning?",
     "answer": "Deep Learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns in data."},

    {"question": "What is a neural network?",
     "answer": "A neural network is a system of algorithms modeled after the human brain that recognize patterns and learn from data."},

    {"question": "What is natural language processing?",
     "answer": "NLP is a branch of AI that enables computers to understand, interpret, and generate human language."},

    {"question": "What is Python?",
     "answer": "Python is a popular, beginner-friendly programming language widely used in AI, data science, and web development."},

    {"question": "What is a dataset?",
     "answer": "A dataset is a collection of data used to train and test machine learning models."},

    {"question": "What is overfitting?",
     "answer": "Overfitting happens when a model learns the training data too well and performs poorly on new/unseen data."},

    {"question": "What is supervised learning?",
     "answer": "Supervised learning is a type of ML where the model is trained on labeled data (input-output pairs)."},

    {"question": "What is unsupervised learning?",
     "answer": "Unsupervised learning is a type of ML where the model finds patterns in data without labeled answers."},

    {"question": "What is reinforcement learning?",
     "answer": "Reinforcement learning is a type of ML where an agent learns to make decisions by receiving rewards or penalties."},

    {"question": "Hello",
     "answer": "Hello! How can I help you today?"},

    {"question": "Who are you?",
     "answer": "I am an AI FAQ chatbot. I can answer questions related to Artificial Intelligence and Machine Learning."},

    {"question": "Bye",
     "answer": "Goodbye! Have a great day!"},
]

# ============================================================
# TEXT PREPROCESSING
# ============================================================
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Preprocess all FAQ questions
faq_questions = [preprocess(f["question"]) for f in faqs]
faq_answers   = [f["answer"] for f in faqs]

# ============================================================
# FIND BEST MATCHING ANSWER
# ============================================================
def get_answer(user_input):
    cleaned_input = preprocess(user_input)

    # Combine user input with all FAQ questions for vectorization
    all_texts = faq_questions + [cleaned_input]

    # Convert to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Cosine similarity between user input and all FAQ questions
    user_vector = tfidf_matrix[-1]             # Last vector = user input
    faq_vectors = tfidf_matrix[:-1]            # All others = FAQs

    similarities = cosine_similarity(user_vector, faq_vectors).flatten()

    best_idx   = similarities.argmax()          # Index of most similar FAQ
    best_score = similarities[best_idx]         # Similarity score (0 to 1)

    # If similarity is too low, say we don't know
    if best_score < 0.1:
        return "Sorry, I don't understand that. Could you rephrase your question?"

    return faq_answers[best_idx]

# ============================================================
# BUILD THE CHAT UI
# ============================================================
def send_message(event=None):
    user_text = entry.get().strip()
    if not user_text:
        return

    # Show user message
    chat_box.config(state="normal")
    chat_box.insert(tk.END, f"You:  {user_text}\n", "user")

    # Get bot response
    response = get_answer(user_text)
    chat_box.insert(tk.END, f"Bot:  {response}\n\n", "bot")

    chat_box.see(tk.END)
    chat_box.config(state="disabled")
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("AI FAQ Chatbot")
root.geometry("620x520")
root.configure(bg="#0f0f1a")

tk.Label(root, text="🤖 AI FAQ Chatbot", font=("Arial", 16, "bold"),
         bg="#0f0f1a", fg="#7dcfff").pack(pady=10)

# Chat display area
chat_box = scrolledtext.ScrolledText(root, state="disabled", width=72, height=22,
                                      font=("Arial", 10), bg="#1a1a2e", fg="#c0caf5",
                                      relief="flat", wrap=tk.WORD)
chat_box.pack(padx=15, pady=5)
chat_box.tag_config("user", foreground="#9ece6a")
chat_box.tag_config("bot", foreground="#7dcfff")

# Starter message
chat_box.config(state="normal")
chat_box.insert(tk.END, "Bot:  Hello! Ask me anything about AI/ML.\n\n", "bot")
chat_box.config(state="disabled")

# Input row
input_frame = tk.Frame(root, bg="#0f0f1a")
input_frame.pack(pady=5)

entry = tk.Entry(input_frame, width=52, font=("Arial", 11),
                 bg="#1a1a2e", fg="#c0caf5", insertbackground="white", relief="flat")
entry.grid(row=0, column=0, padx=5, pady=5, ipady=6)
entry.bind("<Return>", send_message)  # Press Enter to send

tk.Button(input_frame, text="Send", command=send_message,
          bg="#7dcfff", fg="#0f0f1a", font=("Arial", 11, "bold"),
          padx=15, pady=5, relief="flat").grid(row=0, column=1, padx=5)

root.mainloop()
