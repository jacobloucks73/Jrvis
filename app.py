# app.py

from chatbot_engine import ChatbotEngine
from nlp_processor import NLPProcessor

def main():
    processor = NLPProcessor()
    print("Type your message (or type 'exit' to quit):")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
        response = processor.generate_response(user_input)
        print("Jr Jarvis:", response)

if __name__ == "__main__":
    main()
