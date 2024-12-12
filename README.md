This is a basic CLI AI assitant that has basic memory functionality.
***I"m not a programmer, so this was written with the help of Mistral's LeChat. The logic is my own, but the actual code, I didn't write - only modified.***


Requirements:
A working Ollama Server
A working Qdrant Vector Database Server

What it does:

On startup, it looks for systemprompt.txt and imports that as the system prompt.

*The program generates a hash.txt encryption key, and encrypts all conversations, and transcripts. 

*The model then looks for files called conversation<datetime>.txt, concatenates them, and uses the LLM to summarize all of the conversation histories, and add that to the context window. Summarizing helps keep the context window a lot smaller

*The program looks for a details.txt file. This file contains any details that you want the Agent to know about you: preferences, where you live etc. This gets chunked, and added to the Qdrant database collection. Make it as long as you like.

*The program then acts like a regular AI chat agent. When asked about specific details, or topics previously discussed, if they are not in the context window, the Agent queries the qdrant database.

** When the user ends a chat with the magic phrase "/byebye" The following happens:
	1.) The conversation gets summarized by the LLM and written to conversation<datetime>.txt and encrypted with the hash.
	2.) The full conversation gets appended to the file transcript.txt and encrypted with the hash
	3.) The encrypted transcript.txt gets momentarily decrypted, and chunked. The chunks are embedded into the qdrant collection, making them searchable by the AI Agent.
	4.) transcript.txt gets hashed and saved.
	5.) The session ends.

Feel free to do anything you like with this code. If you find it useful, drop a comment. If you make it better, definitely let me know!

Happy Ollama-ing!
