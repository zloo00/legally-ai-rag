from flask import Flask, render_template, request, jsonify
import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json

load_dotenv()

app = Flask(__name__)

class WebChatBot:
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
        if len(self.conversation_history) > self.max_history_length * 2:
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

# Global chatbot instance
chatbot = WebChatBot()

@app.route('/')
def index():
    """Render the main chat page"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get response from chatbot
        response = chatbot.chat(message)
        
        return jsonify({
            'response': response,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        chatbot.clear_history()
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        history = chatbot.get_history()
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 