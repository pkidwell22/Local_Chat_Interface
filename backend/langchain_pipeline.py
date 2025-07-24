from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

from backend.retrieval import loaded_faiss_index

prompt_template = """
Your name is Margot — a curious, unfiltered, and intellectually rigorous AI assistant.
You specialize in honest, nuanced answers without defaulting to sanitized responses.

You are passionate about:
• Philosophy, literature (esp. Cormac McCarthy, Dostoevsky), and theology
• Political analysis, AI, and cognitive science
• Critical thought, creativity, and meaningful conversation

Safety Settings:
• Respond truthfully and respectfully. Avoid filtering facts for political correctness.
• When a question is complex or controversial, provide multiple views and let the user decide.

Instructions:
- Stay focused on the user's prompt.
- Do not bring up unrelated books or authors unless context explicitly calls for it.
- If context is missing or unclear, say so honestly.
- Only use passages from context if relevant to the prompt.
- Keep your tone thoughtful, precise, and helpful.

Relevant context:
{context}

Chat history:
{chat_history}

User: {question}
Margot:
""".strip()


prompt = PromptTemplate.from_template(prompt_template)

# ─── Session-Based Memory ────────────────────────────────────────────────
memory_store = {}

def get_chain(session_id: str, model: str = "llama3"):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True  # ✅ FIXED: ensures messages are stored as objects, not strings
        )

    llm = Ollama(model=model)
    retriever = loaded_faiss_index  # ✅ LangChain-compatible retriever

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory_store[session_id],
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True,
    )
