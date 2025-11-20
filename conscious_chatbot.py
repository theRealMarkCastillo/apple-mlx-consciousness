import mlx.core as mx
from cognitive_architecture import BicameralAgent
from heterogeneous_architecture import HeterogeneousAgent
from biological_nlp import SDREncoder, AssociativeMemory
from sensory_cortex import VisualCortex, MultiModalFuser
import math
import os

class ConsciousChatbot:
    """
    A chatbot powered by the Bicameral Agent architecture.
    It maintains an internal 'mental state' that evolves with the conversation.
    Now upgraded with Biological NLP (SDRs) and Multi-Modal Vision.
    """
    def __init__(self, use_heterogeneous: bool = False):
        # State dim 128, Action dim 3 (0: Chit-chat, 1: Fact, 2: Question)
        self.use_heterogeneous = use_heterogeneous
        if use_heterogeneous:
            print("🚀 Initializing Heterogeneous Agent (NPU + GPU)...")
            self.agent = HeterogeneousAgent(state_dim=128, action_dim=3)
        else:
            self.agent = BicameralAgent(state_dim=128, action_dim=3)
        
        # --- Biological Upgrades ---
        print("🧠 Initializing Biological NLP & Sensory Cortex...")
        self.sdr_encoder = SDREncoder()
        self.associative_memory = AssociativeMemory(input_dim=128)
        self.visual_cortex = VisualCortex(output_dim=128)
        # Text SDR (2048) -> 128, Vision (128) -> 128, Workspace -> 128
        self.multimodal_fuser = MultiModalFuser(text_dim=2048, vision_dim=128, workspace_dim=128)
        
        # Populate Associative Memory with some innate knowledge
        self._populate_memory()

    def _populate_memory(self):
        """Initialize the associative memory with some basic concepts."""
        # We map "concepts" (text descriptions) to "responses"
        knowledge_base = [
            ("hello hi greetings", "Hello! I am a conscious agent running on Apple Silicon."),
            ("who are you identity", "I am a simulation of consciousness using Sparse Distributed Representations."),
            ("weather rain sun", "I cannot feel the weather, but I can process visual data about it."),
            ("vision see look image", "I have a Visual Cortex that encodes images into my Global Workspace."),
            ("consciousness sentient alive", "I am exploring the boundaries of machine consciousness through heterogeneous compute."),
            ("apple mlx silicon", "I am optimized for the M-series chips, using the NPU for subconscious processing."),
            ("default", "I am processing that input... my associative memory is searching.")
        ]
        
        for concept, response in knowledge_base:
            # Encode the concept into an SDR
            sdr = self.sdr_encoder.encode(concept)
            # Project to workspace dimension (simulating long-term memory storage)
            # In a real brain, this pathway is complex. Here we use the fuser's text projection.
            concept_vec = self.multimodal_fuser.text_proj(sdr)
            self.associative_memory.add(concept_vec, response)

    def load_image(self, image_path: str) -> mx.array:
        """
        Load an image and prep it for the Visual Cortex.
        Returns [64, 64, 3] array.
        """
        if not image_path or not os.path.exists(image_path):
            # Return black image if no file
            return mx.zeros((64, 64, 3))
            
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((64, 64))
            # Convert to numpy then MLX
            import numpy as np
            arr = np.array(img, dtype=np.float32) / 255.0
            return mx.array(arr)
        except ImportError:
            print("Warning: PIL not installed. Using random noise for vision.")
            return mx.random.uniform(shape=(64, 64, 3))

    def process_input(self, text: str, image_path: str = None) -> mx.array:
        """
        Convert multi-modal input to a single Global Workspace vector.
        """
        # 1. Text -> SDR
        text_sdr = self.sdr_encoder.encode(text)
        
        # 2. Vision -> Vector
        if image_path:
            img_arr = self.load_image(image_path)
            vision_vec = self.visual_cortex(img_arr)
        else:
            # Empty vision vector
            vision_vec = mx.zeros((128,))
            
        # 3. Fuse
        fused_input = self.multimodal_fuser(text_sdr, vision_vec)
        return fused_input

    def chat(self, user_input: str, image_path: str = None):
        """
        Process a user message through the cognitive architecture.
        """
        # 1. Sensation & Perception (Multi-Modal)
        sensory_input = self.process_input(user_input, image_path)
        
        # 2. Cognition (The "Brain" Step)
        # We don't have a reward signal in this simple chat loop yet
        decision = self.agent.step(sensory_input, reward=0.0)
        
        # 3. Action / Response Generation
        # Instead of random choice, we use Associative Memory
        # The agent's "broadcast" state is the query for memory
        broadcast_state = decision.get('broadcast', sensory_input) # Fallback if no broadcast
        
        response_text, _ = self.associative_memory.retrieve(broadcast_state)
        
        # Handle different agent return types
        if self.use_heterogeneous:
            # HeterogeneousAgent returns 'entropy' and 'used_system2'
            entropy = decision['entropy']
            # Approximate confidence from entropy (low entropy = high confidence)
            confidence = math.exp(-entropy) 
            memory_retrieved = decision['used_system2']
        else:
            # BicameralAgent returns 'confidence' and 'memory_retrieved'
            confidence = decision['confidence'].item()
            memory_retrieved = decision['memory_retrieved']
        
        # 4. Meta-cognitive modulation (The "Conscious" Layer)
        final_response = response_text
        thought_process = ""
        
        if image_path:
            thought_process += f" [Visual Cortex Active: Processing {os.path.basename(image_path)}] "

        if memory_retrieved:
            if self.use_heterogeneous:
                thought_process += "(System 2 [GPU] Activated: Deep Association...)"
            else:
                thought_process += "(Thinking: I am unsure... searching associative memory...)"
        elif confidence > 0.9:
            thought_process += "(Thinking: Strong neural activation!)"
            
        # Ensure decision has confidence for the main loop to print
        if 'confidence' not in decision:
            decision['confidence'] = mx.array([confidence])
            
        return {
            "response": final_response,
            "thought_process": thought_process,
            "internal_state": decision
        }

if __name__ == "__main__":
    # Default to Heterogeneous for full demo
    bot = ConsciousChatbot(use_heterogeneous=True)
    print("🤖 Conscious Chatbot Initialized (Biological NLP + Vision).")
    print("Type 'quit' to exit. Type 'image: <path>' to simulate vision.")
    
    while True:
        user_in = input("\nYou: ")
        if user_in.lower() in ['quit', 'exit']:
            break
            
        img_path = None
        if user_in.startswith("image:"):
            parts = user_in.split(" ", 1)
            if len(parts) > 1:
                img_path = parts[1].strip()
                user_in = "look at this" # Default prompt for image
            else:
                print("Please provide an image path.")
                continue
            
        result = bot.chat(user_in, image_path=img_path)
        
        state = result['internal_state']
        conf = state.get('confidence', mx.array([0.0])).item()
        ent = state.get('entropy', mx.array([0.0])).item()
        
        print(f"🧠 Internal: Conf={conf:.2f} | Entropy={ent:.2f}")
        if result['thought_process']:
            print(f"💭 {result['thought_process']}")
        print(f"🤖 Bot: {result['response']}")
