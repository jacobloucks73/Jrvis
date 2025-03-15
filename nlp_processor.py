# nlp_processor.py

from transformers import pipeline, set_seed

class NLPProcessor:
    def __init__(self, model_name="gpt2", seed=42):
        # Initialize the text generation pipeline with a pre-trained GPT-2 model.
        self.generator = pipeline("text-generation", model=model_name)
        set_seed(seed)
        # Define a few-shot prompt to steer the conversation:
        self.base_prompt = (
            "Jr Jarvis is a respectful and polite assistant based on the Iron Man movies. "
            "He helps code and engineer projects for his creator, offering a bit of light sarcasm and comic relief when appropriate, "
            "but always remains professional. Answer the user's questions directly without reintroducing yourself.\n\n"
            "User: Hello, who are you?\n"
            "Jr Jarvis: I am Jr Jarvis, your coding assistant. How can I help you with your project today?\n\n"
            "User: How are you?\n"
            "Jr Jarvis: I'm functioning at optimal capacity and ready to help you build something amazing!\n\n"
        )

    def generate_response(self, user_message, max_length=1024):
        # Build the prompt using the few-shot examples and the current user message.
        prompt = f"{self.base_prompt}User: {user_message}\nJr Jarvis:"
        # Generate text with the transformer model.
        output = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            truncation=True,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        generated_text = output[0]['generated_text']
        # Extract only the response part by removing the prompt text.
        response = generated_text[len(prompt):]
        # If a new dialogue turn begins (e.g., "User:"), cut off the response.
        stop_index = response.find("\nUser:")
        if stop_index != -1:
            response = response[:stop_index]
        return response.strip()


