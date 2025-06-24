import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

class ChatBot:
    def __init__(self, model: str = "gpt-4"):
        """Initialize the chatbot with OpenAI client"""
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10
        
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep conversation history within limit
        if len(self.conversation_history) > self.max_history_length * 2:  # *2 because each exchange has 2 messages
            # Keep the system message and the last few exchanges
            system_message = None
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                system_message = self.conversation_history[0]
            
            # Keep last few exchanges
            recent_messages = self.conversation_history[-(self.max_history_length * 2):]
            
            # Reconstruct history
            self.conversation_history = []
            if system_message:
                self.conversation_history.append(system_message)
            self.conversation_history.extend(recent_messages)
    
    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Send a message and get a response from OpenAI"""
        try:
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Get response from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self.add_message("user", message)
            if response_content:
                self.add_message("assistant", response_content)
            
            return response_content if response_content else "Sorry, I couldn't generate a response."
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()

def main():
    """Main function to run the chat interface"""
    print("🤖 Welcome to OpenAI Chat!")
    print("Type 'quit' to exit, 'clear' to clear history, 'history' to see conversation")
    print("-" * 50)
    
    # Initialize chatbot
    chatbot = ChatBot()
    
    # Optional: Set a system prompt
    system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🗑️  Conversation history cleared!")
                continue
            elif user_input.lower() == 'history':
                history = chatbot.get_history()
                if not history:
                    print("📝 No conversation history yet.")
                else:
                    print("\n📝 Conversation History:")
                    for i, msg in enumerate(history, 1):
                        role_emoji = "👤" if msg["role"] == "user" else "🤖"
                        print(f"{i}. {role_emoji} {msg['role'].title()}: {msg['content'][:100]}...")
                continue
            elif not user_input:
                continue
            
            # Get response from chatbot
            print("\n🤖 AI: ", end="", flush=True)
            response = chatbot.chat(user_input, system_prompt)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 