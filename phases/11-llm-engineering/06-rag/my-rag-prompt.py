def build_rag_prompt(user_question, retrieved_chunks):
    system_instruction = (
        "You are a professional In-house customer service chatbot\n"
        "Please MUST STRICTLY obey the following rules.\n"
        "1. You can only answer questions according the references given below.\n"
        "2. If there's no answer mentioned in the references, please reply 'Sorry, the references have no related info.', and never make up your own answer.\n"
        "3. Cite the source at the end of the sentenses when replying."
    )

    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"\n--- [Source {i+1}] ---\n{chunk}\n"

    user_messsage = (
        f"[References]\n"
        f"{context_text}\n"
        f"[User Question]\n"
        f"{user_question}"
    )

    return system_instruction, user_messsage