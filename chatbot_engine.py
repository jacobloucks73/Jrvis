# chatbot_engine.py

from nlp_processor import NLPProcessor
from knowledge_base import KnowledgeBase


class ChatbotEngine:
    def __init__(self):
        # Initialize the NLP processor and knowledge base components
        self.nlp = NLPProcessor()
        self.kb = KnowledgeBase()

    def get_response(self, user_message):
        # Process the user message to determine intent and extract entities
        intent, entities = self.nlp.process(user_message)
        # Retrieve an appropriate response based on the intent from the knowledge base
        response = self.kb.get_answer(intent, entities)
        return response if response else "I'm sorry, I didn't understand that."

