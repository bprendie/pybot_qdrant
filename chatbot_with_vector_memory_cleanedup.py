import os
import datetime
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from cryptography.fernet import Fernet
from qdrant_client import QdrantClient, models as rest
from langchain_ollama.embeddings import OllamaEmbeddings

# Constants
OLLAMA_URL = "http://ollamaurl:11434"
OLLAMA_MODEL = "hermes3"
HASH_FILE = "hash.txt"
SYSTEM_PROMPT_FILE = "systemprompt.txt"
DETAILS_FILE = "details.txt"
INNER_THOUGHTS_FILE = "inner_thoughts.txt"
QDRANT_HOST = "qdrant_url"
QDRANT_PORT = 6333
COLLECTION_NAME = "chat_history_01"
INNER_THOUGHTS_COLLECTION = "inner_thoughts_01"
DETAILS_COLLECTION = "details_01"
#QDRANT_API_KEY = "your_api_key"

# Initialize QdrantClient with HTTP
client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# Ensure the Qdrant collections are set up correctly
for collection in [COLLECTION_NAME, INNER_THOUGHTS_COLLECTION, DETAILS_COLLECTION]:
    try:
        client.get_collection(collection)
        client.delete_collection(collection)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE)
    )
    print(f"Qdrant collection '{collection}' created with vector size 768.")

# Initialize Ollama embeddings from LangChain
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")

# --- Function Definitions ---

def decrypt_message(encrypted_message, key):
    """Decrypts an encrypted message using Fernet."""
    f = Fernet(key)
    return f.decrypt(encrypted_message).decode()

def encrypt_message(message, key):
    """Encrypts a message using Fernet."""
    f = Fernet(key)
    return f.encrypt(message.encode())

def generate_key():
    """Generates a new Fernet key and saves it to hash.txt."""
    key = Fernet.generate_key()
    with open(HASH_FILE, "wb") as key_file:
        key_file.write(key)
    print(f"New Fernet key generated and saved to {HASH_FILE}")

def load_or_generate_key():
    """Loads the encryption key from file or generates a new one if it doesn't exist."""
    if not os.path.exists(HASH_FILE):
        generate_key()
    with open(HASH_FILE, "r") as f:
        return f.read().strip()

def load_system_prompt():
    """Loads the system prompt from file or prompts the user to enter a new one."""
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            system_prompt = f.read()
        agent_name = system_prompt.split("AGENT_NAME=")[1].split("\n")[0]
        if input("Use default system prompt? (y/n): ").lower() != "y":
            agent_name = input("Enter agent name: ")
            system_prompt = input("Enter your system prompt: ")
            with open(SYSTEM_PROMPT_FILE, "w") as f:
                f.write(f"AGENT_NAME={agent_name}\n{system_prompt}")
            print("System prompt saved to systemprompt.txt")
        else:
            print("Loaded system prompt from systemprompt.txt")
    else:
        agent_name = input("Enter agent name: ")
        system_prompt = input("Enter your system prompt: ")
        with open(SYSTEM_PROMPT_FILE, "w") as f:
            f.write(f"AGENT_NAME={agent_name}\n{system_prompt}")
        print("System prompt saved to systemprompt.txt")
    return agent_name, system_prompt

def concatenate_histories(hash_key):
    """Concatenates all conversation history files into history.txt."""
    conversation_files = sorted(
        [f for f in os.listdir() if f.startswith("conversation_") and f.endswith(".txt")]
    )
    with open("history.txt", "w") as outfile:
        for fname in conversation_files:
            try:
                with open(fname, "rb") as infile:
                    encrypted_content = infile.read()
                    decrypted_content = decrypt_message(encrypted_content, hash_key)
                    outfile.write(decrypted_content + "\n")
            except Exception as e:
                print(f"Error reading or decrypting {fname}: {e}")

def summarize_history():
    """Summarizes the concatenated history and saves it to history_summary.txt."""
    try:
        with open("history.txt", "r") as f:
            full_history = f.read()
        summary_prompt = f"Summarize this detailed conversation history:\n\n{full_history}\n\nSummary:"
        summary = llm.predict(summary_prompt)
        with open("history_summary.txt", "w") as f:
            f.write(summary)
        print("History summarized and saved to history_summary.txt")
    except Exception as e:
        print(f"Error summarizing history: {e}")

def load_history_summary():
    """Loads the history summary from history_summary.txt."""
    try:
        with open("history_summary.txt", "r") as f:
            summary = f.read()
        print("\nLoaded History Summary:")
        print(summary)
        return summary
    except FileNotFoundError:
        print("No history summary found.")
        return None

def import_conversation_summary(hash_key):
    """Imports the latest conversation summary if it exists."""
    conversation_files = [f for f in os.listdir() if f.startswith("conversation_") and f.endswith(".txt")]
    if conversation_files:
        latest_file = max(conversation_files, key=os.path.getctime)
        print(f"Found a previous summary: {latest_file}")
        if input("Import this summary? (y/n): ").lower() == "y":
            try:
                with open(latest_file, "rb") as f:
                    encrypted_summary = f.read()
                decrypted_summary = decrypt_message(encrypted_summary, hash_key)
                print("\nImported Summary:")
                print(decrypted_summary)
                return decrypted_summary
            except Exception as e:
                print(f"Error decrypting summary: {e}")
    else:
        print("No previous summaries found.")
    return None

def save_conversation_summary(summary, hash_key):
    """Saves the conversation summary to a file."""
    encrypted_summary = encrypt_message(summary, hash_key)
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{current_datetime}.txt"
    with open(filename, "wb") as f:
        f.write(encrypted_summary)
    print(f"Conversation saved to {filename}")

def chunk_history(history, chunk_size=500):
    """Chunks the conversation history into smaller segments."""
    return [history[i:i + chunk_size] for i in range(0, len(history), chunk_size)]

def save_to_qdrant(history, is_details=False, collection_name=COLLECTION_NAME):
    """Saves the conversation history or personal details from a file to the Qdrant database."""
    try:
        chunks = chunk_history(history)
        points = []
        for i, chunk in enumerate(chunks):
            embeddings_vector = embeddings.embed_query(chunk)
            timestamp_id = int(datetime.datetime.now().timestamp()) + i  # Ensure unique IDs
            points.append(
                rest.PointStruct(
                    id=timestamp_id,
                    vector=embeddings_vector,
                    payload={"history": chunk, "is_details": is_details},
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        print(f"Data saved to Qdrant collection '{collection_name}'.")
    except Exception as e:
        print(f"Error saving to Qdrant: {e}")

def save_transcript(history, filename="transcript.txt"):
    """Appends the conversation history to a plain text file."""
    try:
        with open(filename, "a") as f:
            f.write(history + "\n")
        print(f"Transcript appended to {filename}")
    except Exception as e:
        print(f"Error saving transcript: {e}")

def load_transcript(filename="transcript.txt"):
    """Loads the transcript file."""
    try:
        with open(filename, "r") as f:
            transcript_content = f.read()
        print(f"Transcript loaded from {filename}")
        return transcript_content
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return ""

def query_qdrant(query_text, limit=5, collection_name=COLLECTION_NAME):
    """Queries the Qdrant database to retrieve relevant conversation history or personal details."""
    try:
        query_vector = embeddings.embed_query(query_text)
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return [point.payload for point in search_result]
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []

def load_personal_details():
    """Loads personal details from details.txt."""
    try:
        with open(DETAILS_FILE, "r") as f:
            details = f.read()
        print("\nLoaded Personal Details:")
        #print(details)
        return details
    except FileNotFoundError:
        print("No personal details found.")
        return None

def integrate_personal_details(user_input, relevant_history):
    """Integrates personal details into the agent's response."""
    personal_details = [chunk["history"] for chunk in relevant_history if chunk.get("is_details", False)]
    if personal_details:
        personal_details_str = "\n".join(personal_details)
        integrated_input = f"{user_input}\n\nPersonal Details:\n{personal_details_str}"
        return integrated_input
    return user_input

def save_inner_thoughts(inner_thoughts):
    """Appends the inner thoughts to a plain text file and saves them to Qdrant."""
    try:
        with open(INNER_THOUGHTS_FILE, "a") as f:
            f.write(inner_thoughts + "\n")
        print(f"Inner thoughts appended to {INNER_THOUGHTS_FILE}")

        # Save inner thoughts to Qdrant without encryption
        save_to_qdrant(inner_thoughts, is_details=False, collection_name=INNER_THOUGHTS_COLLECTION)
    except Exception as e:
        print(f"Error saving inner thoughts: {e}")

def load_inner_thoughts():
    """Loads the inner thoughts file."""
    try:
        with open(INNER_THOUGHTS_FILE, "r") as f:
            inner_thoughts_content = f.read()
        print(f"Inner thoughts loaded from {INNER_THOUGHTS_FILE}")
        return inner_thoughts_content
    except Exception as e:
        print(f"Error loading inner thoughts: {e}")
        return ""

def query_inner_thoughts(query_text, limit=5):
    """Queries the Qdrant database to retrieve relevant inner thoughts."""
    try:
        query_vector = embeddings.embed_query(query_text)
        search_result = client.search(
            collection_name=INNER_THOUGHTS_COLLECTION,
            query_vector=query_vector,
            limit=limit
        )
        return [point.payload for point in search_result]
    except Exception as e:
        print(f"Error querying Qdrant for inner thoughts: {e}")
        return []

def query_details(query_text, limit=5):
    """Queries the Qdrant database to retrieve relevant personal details from the 'details_01' collection."""
    try:
        query_vector = embeddings.embed_query(query_text)
        search_result = client.search(
            collection_name=DETAILS_COLLECTION,
            query_vector=query_vector,
            limit=limit
        )
        return [point.payload for point in search_result]
    except Exception as e:
        print(f"Error querying Qdrant for personal details: {e}")
        return []

# --- Initialize ---
llm = OllamaLLM(base_url=OLLAMA_URL, model=OLLAMA_MODEL, temperature=0.7)
ConversationChain.verbosity = 0

# --- Load or Generate Encryption Key ---
hash_key = load_or_generate_key()

# --- Load System Prompt ---
agent_name, system_prompt = load_system_prompt()

# --- Get User's Name ---
user_name = input("Enter your name: ")

# --- Create Conversation Chain ---
template = PromptTemplate(
    input_variables=["history", "input"],
    template=f"{system_prompt}\n\n{{history}}\n{user_name}: {{input}}\n{agent_name}:",
)
memory = ConversationBufferWindowMemory(k=100000)
conversation = ConversationChain(llm=llm, memory=memory, prompt=template)

# --- Concatenate and Summarize History ---
concatenate_histories(hash_key)
summarize_history()

# --- Load History Summary ---
decrypted_summary = load_history_summary()
if decrypted_summary:
    conversation.memory.save_context(
        {"input": "Here's a summary of our previous conversations:"},
        {"output": decrypted_summary},
    )
    print("Previous conversation summary imported.")

# --- Load and Embed Personal Details ---
personal_details = load_personal_details()
if personal_details:
    save_to_qdrant(personal_details, is_details=True, collection_name=DETAILS_COLLECTION)

# --- Main Conversation Loop ---
while True:
    user_input = input(f"{user_name}: ")
    if user_input.lower() == "/byebye":
        history = conversation.memory.load_memory_variables({})["history"]

        # Save the transcript
        save_transcript(history)

        # Load and chunk transcript.txt
        transcript_content = load_transcript()
        save_to_qdrant(transcript_content)

        # Save inner thoughts to Qdrant (already done in save_inner_thoughts)

        summary_prompt = f"This is the conversation we just had:\n\n{history}\n\nIn your own voice, summarize the key points so we can continue talking later."
        summary = llm.predict(summary_prompt)
        save_conversation_summary(summary, hash_key)
        break
    elif user_input.lower() == "/thoughts":
        relevant_thoughts = query_inner_thoughts(conversation.memory.load_memory_variables({})["history"])
        if relevant_thoughts:
            print("Inner Thoughts:")
            for thought in relevant_thoughts:
                print(thought["history"])
        else:
            print("No relevant inner thoughts found.")
        continue

    # Query Qdrant for relevant past conversations and personal details
    relevant_history = query_qdrant(user_input)
    relevant_details = query_details(user_input)

    # Integrate personal details into the user input
    integrated_input = integrate_personal_details(user_input, relevant_details)

    response = conversation.invoke({"input": integrated_input})

    # Generate "inner thoughts"
    inner_thoughts_prompt = f"""Inner Thoughts: I just generated the response: '{response['response']}'.
    Now, reflect on why you said that. What factors influenced your response?
    What information did you find most important? What were your goals in
    formulating this response? Keep your inner thoughts concise and insightful."""

    inner_thoughts = llm.predict(inner_thoughts_prompt)

    # Save inner thoughts
    save_inner_thoughts(inner_thoughts)

    print(f"{agent_name}: {response['response']}")

