def support_prompt(context: str, hyp: str) -> str:
    """
    Standardized yes/no support prompt for seq2seq and decoder-only judges.
    """
    return (
        f"Context: {context}\n"
        f"Sentence: {hyp}\n"
        f"Is the sentence supported by the context above?\n"
        f"Answer with exactly one word: yes or no."
    )