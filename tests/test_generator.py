import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from src.generation.generator import Generator 

def test_simple_generation():
    gen = Generator()
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital city is Paris."
    answer = gen.generate_answer(question, context)
    print("Q:", question)
    print("A:", answer)

if __name__ == "__main__":
    test_simple_generation()
