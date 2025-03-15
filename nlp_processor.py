# nlp_processor.py
from transformers import pipeline, set_seed

class NLPProcessor:
    def __init__(self, model_name="./jarvis_model", seed=42):
        self.generator = pipeline("text-generation", model=model_name)
        set_seed(seed)

    def generate_response(self, user_message, max_length=150):
        prompt = f"User: {user_message}\nJr Jarvis:"
        output = self.generator(
            prompt,
            max_length=max_length,
            pad_token_id=self.generator.tokenizer.eos_token_id,
            truncation=True,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        response = output[0]['generated_text'][len(prompt):].split("\nUser:")[0].strip()
        return response

if __name__ == "__main__":
    processor = NLPProcessor()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Jr Jarvis:", processor.generate_response(user_input))
