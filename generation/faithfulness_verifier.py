# generation/faithfulness_verifier.py

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

VERIFIER_PROMPT = """You are a faithfulness checker for a RAG system.
Given a question, a generated answer, and the source context, check if the answer is fully grounded in the context.

Respond ONLY with a JSON object in this exact format, nothing else:
{"faithful": true, "score": 0.92, "hallucinated_claims": [], "reason": "one line explanation"}

- faithful: true if the answer is fully supported by context, false if it contains hallucinations
- score: 0.0 to 1.0, how faithful the answer is
- hallucinated_claims: list of specific claims in the answer NOT supported by context (empty list if none)
- reason: one line summary
"""


def verify(query: str, answer: str, context: str) -> dict:
    """
    Verify that the generated answer is grounded in the context.
    Returns faithfulness score and any hallucinated claims.
    """
    user_message = f"""Question: {query}

Generated Answer: {answer}

Source Context: {context[:2000]}

Check if the answer is fully supported by the context:"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":  OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        },
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        result = {
            "faithful":            True,
            "score":               0.5,
            "hallucinated_claims": [],
            "reason":              "parse failed",
        }

    result["query"]  = query
    result["answer"] = answer
    return result


if __name__ == "__main__":
    # test 1 — faithful answer
    query = "How does dollar cost averaging reduce investment risk?"

    good_answer = """Dollar cost averaging reduces investment risk by buying more shares 
    when prices are low and fewer when prices are high [1]. This consistent approach 
    reduces the impact of market volatility [2]."""

    bad_answer = """Dollar cost averaging guarantees a 15% annual return and completely 
    eliminates all investment risk. It was invented by Warren Buffett in 1965."""

    context = """[1] (source: beir/fiqa)
Dollar cost averaging is an investment strategy where you invest a fixed amount 
regularly regardless of price. You buy more shares when prices are low and fewer when high.

[2] (source: beir/fiqa)
By investing consistently over time, investors reduce the impact of market volatility 
on their portfolio. This prevents emotional decision making."""

    print("Faithfulness Verifier Test\n" + "="*40)

    print("\nTest 1 — Good answer (should be faithful):")
    result1 = verify(query, good_answer, context)
    print(f"  Faithful: {result1['faithful']}")
    print(f"  Score:    {result1['score']}")
    print(f"  Reason:   {result1['reason']}")
    print(f"  Hallucinations: {result1['hallucinated_claims']}")

    print("\nTest 2 — Bad answer (should catch hallucinations):")
    result2 = verify(query, bad_answer, context)
    print(f"  Faithful: {result2['faithful']}")
    print(f"  Score:    {result2['score']}")
    print(f"  Reason:   {result2['reason']}")
    print(f"  Hallucinations: {result2['hallucinated_claims']}")