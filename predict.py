from transformers import AutoTokenizer, AutoModelForCausalLM

class Predictor:
    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

    def predict(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
