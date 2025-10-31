import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.generator import Generator

# Load generator model
generator = Generator(model_name="google/flan-t5-base")

# Hard coded context for testing
context = "Albert Einstein was a theoretical physicist who developed the theory of relativity"
query = input("Enter your question: ")

# Generate answer
answer, logits, attention = generator.generate_answer(query, context, return_details=True)

print("Answer:")
print(answer)
