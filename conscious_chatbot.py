import mlx.core as mx
from cognitive_architecture import BicameralAgent

class ConsciousChatbot:
    """
    A chatbot powered by the Bicameral Agent architecture.
    It maintains an internal 'mental state' that evolves with the conversation.
    """
    def __init__(self):
        # State dim 128, Action dim 3 (0: Chit-chat, 1: Fact, 2: Question)
        self.agent = BicameralAgent(state_dim=128, action_dim=3)
        
        # Simple vocabulary for demo purposes
        self.vocab = {
            "hello": 0, "hi": 0, "hey": 0,
            "what": 1, "who": 1, "why": 1, "how": 1,
            "weather": 2, "time": 2, "name": 2
        }
        
        self.responses = {
            0: ["Hello there!", "Hi!", "Greetings."], # Chit-chat
            1: ["That is an interesting fact.", "I believe so.", "Let me check."], # Factual
            2: ["Could you clarify?", "Why do you ask?", "Tell me more."] # Questioning
        }

    def text_to_embedding(self, text: str) -> mx.array:
        """
        Convert text to a sensory vector for the Global Workspace.
        In a real system, this would be an LLM embedding (e.g., BERT/Llama).
        Here, we use a simple hashed vector for demonstration.
        """
        # Create a deterministic random vector based on the input text hash
        # This ensures the same text always gives the same "sensation"
        seed = abs(hash(text)) % (2**32)
        key = mx.random.key(seed)
        vector = mx.random.normal((128,), key=key)
        return vector

    def chat(self, user_input: str):
        """
        Process a user message through the cognitive architecture.
        """
        # 1. Sensation
        sensory_input = self.text_to_embedding(user_input)
        
        # 2. Cognition (The "Brain" Step)
        # We don't have a reward signal in this simple chat loop yet
        decision = self.agent.step(sensory_input, reward=0.0)
        
        # 3. Action / Response Generation
        action_idx = decision['action']
        confidence = decision['confidence'].item()
        memory_retrieved = decision['memory_retrieved']
        
        # Select response based on action category
        import random
        base_response = random.choice(self.responses.get(action_idx, ["..."]))
        
        # 4. Meta-cognitive modulation (The "Conscious" Layer)
        final_response = base_response
        thought_process = ""
        
        if memory_retrieved:
            thought_process = "(Thinking: I am unsure... recalling past conversations...)"
            final_response = f"Hmm, let me think... {base_response.lower()}"
        elif confidence > 0.9:
            thought_process = "(Thinking: I am very confident!)"
            final_response = f"Absolutely! {base_response}"
            
        return {
            "response": final_response,
            "thought_process": thought_process,
            "internal_state": decision
        }

if __name__ == "__main__":
    bot = ConsciousChatbot()
    print("🤖 Conscious Chatbot Initialized. Type 'quit' to exit.")
    
    while True:
        user_in = input("\nYou: ")
        if user_in.lower() in ['quit', 'exit']:
            break
            
        result = bot.chat(user_in)
        
        conf = result['internal_state']['confidence'].item()
        ent = result['internal_state']['entropy'].item()
        
        print(f"🧠 Internal: Conf={conf:.2f} | Entropy={ent:.2f}")
        if result['thought_process']:
            print(f"💭 {result['thought_process']}")
        print(f"🤖 Bot: {result['response']}")
