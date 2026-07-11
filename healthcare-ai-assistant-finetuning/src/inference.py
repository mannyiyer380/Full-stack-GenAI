"""
Healthcare FAQ Assistant — simple inference script (Step 12).

Loads the final DPO-aligned model and answers a user question. The model can be:
  * a local merged model directory  (e.g. ../outputs/stage3_merged), or
  * a Hugging Face Hub repo id       (e.g. your-username/healthcare-faq-qwen2.5-0.5b-stage3-dpo-merged)

Works on GPU (fast) or CPU (slow but fine for a 0.5B model). Uses plain Transformers so it does
not require Unsloth / a GPU to run.

Usage
-----
  # Ask a single question (local final model):
  python src/inference.py -q "How can I apply for reimbursement?"

  # Point at a Hugging Face repo:
  python src/inference.py -m your-username/healthcare-faq-qwen2.5-0.5b-stage3-dpo-merged \\
                          -q "What temperature is a fever?"

  # Interactive chat (no -q): type questions, 'exit' to quit
  python src/inference.py

  # Or set the model via env var:
  MODEL_PATH=../outputs/stage3_merged python src/inference.py
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a Healthcare FAQ Assistant. You provide clear, general health information for "
    "educational purposes only. You are not a substitute for professional medical advice, "
    "diagnosis, or treatment. Always recommend consulting a qualified healthcare professional, "
    "and advise seeking emergency care for urgent symptoms."
)

# Default to the final (Stage 3 / DPO) merged model relative to the repo root.
DEFAULT_MODEL = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "outputs", "stage3_merged"),
)


def load_model(model_path):
    """Load tokenizer + model onto GPU if available, else CPU."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model = model.to("cpu")
    model.eval()
    print(f"Model loaded on {'GPU' if use_cuda else 'CPU'}.")
    return model, tokenizer


def generate_answer(question, model, tokenizer, max_new_tokens=256):
    """Return the assistant's answer to a single question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy / deterministic
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="Healthcare FAQ Assistant inference")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help="Local model dir or Hugging Face repo id (default: final DPO model).")
    parser.add_argument("-q", "--question", default=None,
                        help="A single question to answer. If omitted, starts interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate (default: 256).")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.question:
        # Single-shot mode (matches the assignment example).
        question = args.question
        answer = generate_answer(question, model, tokenizer, args.max_new_tokens)
        print(answer)
        return

    # Interactive mode.
    print("\nHealthcare FAQ Assistant — type a question ('exit' or 'quit' to stop).")
    print("Note: general health information only, not medical advice.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        answer = generate_answer(question, model, tokenizer, args.max_new_tokens)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    # Mirrors the assignment's example usage:
    #   question = "How can I apply for reimbursement?"
    #   answer = generate_answer(question)
    #   print(answer)
    main()
