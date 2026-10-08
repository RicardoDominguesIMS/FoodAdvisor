"""Hello world: one prompt in, one answer out. Replace with your project."""
import sys

from llm import ask

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    print(ask(prompt))
