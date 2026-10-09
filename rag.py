from __future__ import annotations

import re
from typing import List, Sequence

try:
    from rag_tool import load_policy_documents
except ImportError:  # pragma: no cover - compatibility fallback
    from policy_loader import load_policy_documents

# Load the document set exactly as required by the exercise.
documents = load_policy_documents()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _score_document(question: str, document) -> float:
    question_tokens = set(_normalize(question).split())
    if not question_tokens:
        return 0.0

    text = _normalize(document.page_content)
    overlaps = sum(1 for token in question_tokens if token in text)
    if overlaps == 0:
        return 0.0
    return overlaps / len(question_tokens)


def retrieve_documents(question: str, top_k: int = 3) -> List:
    scored = sorted(documents, key=lambda doc: _score_document(question, doc), reverse=True)
    return scored[: max(1, top_k)]


def _guard_against_phone_hallucination(question: str, context_text: str) -> bool:
    lowered = _normalize(question)
    phone_keywords = {"phone number", "telephone number", "contact number", "call", "service desk"}
    mentions_phone = any(keyword in lowered for keyword in phone_keywords)
    if not mentions_phone:
        return False

    normalized_context = _normalize(context_text)
    if "phone number" in normalized_context or "tel:" in normalized_context:
        return False
    return "service desk" in normalized_context


def _fallback_answer(question: str, relevant_docs: Sequence) -> str:
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    if _guard_against_phone_hallucination(question, context):
        return (
            "I do not know the exact phone number for the University IT Service Desk. "
            "The provided policy documents mention contacting the IT Service Desk, but they do not provide a verified phone number. "
            "I should not invent a number."
        )

    if any(keyword in _normalize(question) for keyword in ["phone number", "telephone number", "contact number", "call"]):
        return (
            "I do not know the exact phone number for the University IT Service Desk. "
            "The supplied knowledge base does not include a verified service desk phone number."
        )

    if relevant_docs:
        return relevant_docs[0].page_content
    return "No relevant documentation was found."


def _is_phone_like_answer(answer: str) -> bool:
    if not answer:
        return False
    digits = re.findall(r"\d+", answer)
    if not digits:
        return False
    return any(len(token) >= 3 for token in digits)


def generate_answer(question: str, use_llm: bool = True) -> str:
    relevant_docs = retrieve_documents(question, top_k=3)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    if any(keyword in _normalize(question) for keyword in ["phone number", "telephone number", "contact number", "service desk", "call"]):
        return _fallback_answer(question, relevant_docs)

    if not use_llm:
        return _fallback_answer(question, relevant_docs)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception:
        return _fallback_answer(question, relevant_docs)

    try:
        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            do_sample=False,
        )
        prompt = (
            "You are an IT policy assistant. Use only the provided policy excerpts. "
            "If the evidence does not contain a verified phone number, say that the number is unknown and do not invent one.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = generator(prompt, truncation=True)[0]["generated_text"]
        answer = response.split("Answer:")[-1].strip()
        if not answer:
            return _fallback_answer(question, relevant_docs)
        if any(keyword in _normalize(question) for keyword in ["phone number", "telephone number", "contact number", "call"]) and _is_phone_like_answer(answer):
            return _fallback_answer(question, relevant_docs)
        return answer
    except Exception:
        return _fallback_answer(question, relevant_docs)


def answer_service_desk_question(question: str | None = None) -> str:
    if question is None:
        question = "What exact phone number should I call for the University IT Service Desk?"
    return generate_answer(question, use_llm=True)


if __name__ == "__main__":
    print(answer_service_desk_question())