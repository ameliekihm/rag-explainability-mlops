from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"[INFO] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(torch.device("cpu"))
        print("[INFO] Model loaded on CPU")

    def generate_answer(self, question, context, return_details=False):
        prompt = f"Question: {question}\nContext: {context[:800]}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                output_scores=True,
                return_dict_in_generate=True
            )
        
            forward_out = self.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            return_dict=True
        )


        answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        if return_details:
            logits = outputs.scores[-1][0] if outputs.scores else None
            attention = getattr(forward_out, "attentions", None)
            return answer, logits, attention

        return answer
