from flask import Flask, render_template, request, jsonify
import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from rag_factory import get_rag_engine
import json
from datetime import datetime

load_dotenv()

app = Flask(__name__)

class WebLegalChatBot:
    def __init__(self, model: str = "gpt-4"):
        """Initialize the legal chatbot with RAG system"""
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.rag_system = get_rag_engine()
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10
        
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep conversation history within limit
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
    
    def get_legal_answer(self, question: str) -> Dict[str, Any]:
        """Get legal answer using RAG system"""
        try:
            # Use RAG system to get answer
            result = self.rag_system.query(question)
            return result
        except Exception as e:
            print(f"Error in RAG query: {e}")
            return {
                "answer": "Извините, произошла ошибка при поиске юридической информации.",
                "sources": [],
                "search_results": []
            }
    
    def get_general_answer(self, question: str) -> str:
        """Get general answer using OpenAI (without RAG)"""
        try:
            # Convert conversation history to proper format
            messages = []
            for msg in self.conversation_history:
                if msg["role"] in ["user", "assistant", "system"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            messages.append({"role": "user", "content": question})
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            return answer if answer else "Извините, не удалось получить ответ."
            
        except Exception as e:
            print(f"Error getting general answer: {e}")
            return "Извините, произошла ошибка при генерации ответа."
    
    def chat(self, message: str, use_rag: bool = True) -> Dict[str, Any]:
        """Main chat method"""
        # Add user message to history
        self.add_message("user", message)
        
        if use_rag:
            # Use RAG for legal questions
            result = self.get_legal_answer(message)
            answer = result["answer"]
            
            # Add assistant response to history
            self.add_message("assistant", answer)
            
            return {
                "answer": answer,
                "sources": result.get("sources", []),
                "search_results": result.get("search_results", []),
                "context_length": result.get("context_length", 0),
                "results_count": result.get("results_count", 0),
                "mode": "legal_rag"
            }
        else:
            # Use general OpenAI for non-legal questions
            answer = self.get_general_answer(message)
            
            # Add assistant response to history
            self.add_message("assistant", answer)
            
            return {
                "answer": answer,
                "sources": [],
                "search_results": [],
                "mode": "general"
            }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.rag_system.clear_conversation_history()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.rag_system.get_system_stats()

# Global chatbot instance
chatbot = WebLegalChatBot()

@app.route('/')
def index():
    """Main page"""
    stats = chatbot.get_system_stats()
    return render_template('legal_chat.html', stats=stats)

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        mode = data.get('mode', 'auto')  # legal | general | auto
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Determine if this is a legal question
        legal_keywords = [
            'закон', 'право', 'статья', 'кодекс', 'договор', 'суд', 'иск',
            'ответственность', 'обязательство', 'собственность', 'наследство',
            'брак', 'развод', 'алименты', 'трудовой', 'налог', 'административный',
            'уголовный', 'гражданский', 'конституция', 'постановление', 'приказ'
        ]
        
        is_legal_question = any(keyword in message.lower() for keyword in legal_keywords)

        if mode == 'legal':
            use_rag = True
        elif mode == 'general':
            use_rag = False
        else:
            use_rag = is_legal_question
        
        # Get response
        result = chatbot.chat(message, use_rag=use_rag)
        
        return jsonify({
            'answer': result['answer'],
            'sources': result.get('sources', []),
            'search_results': result.get('search_results', []),
            'mode': result.get('mode', 'general'),
            'requested_mode': mode,
            'detected_mode': 'legal' if is_legal_question else 'general',
            'results_count': result.get('results_count', 0),
            'context_length': result.get('context_length', 0)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        chatbot.clear_history()
        return jsonify({'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = chatbot.get_system_stats()
        return jsonify(stats)
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
    print("🚀 Запуск веб-сервера для юридического чата...")
    print("📊 Система готова к работе!")
    app.run(debug=True, host='0.0.0.0', port=5000) 
