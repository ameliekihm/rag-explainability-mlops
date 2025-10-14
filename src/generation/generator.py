from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Generator:
    def __init__(self, model_name="google/flan-t5-base"): 
        print(f"[INFO] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        ).to(torch.device("cpu"))
        print("[INFO] Model loaded on CPU")

    def generate_answer(self, question, context):
        prompt = f"Question: {question}\nContext: {context[:800]}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=60)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
