import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from rag_factory import get_rag_engine
import json
from datetime import datetime

load_dotenv()

class LegalChatBot:
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

def main():
    """Main chat interface"""
    print("🤖 Юридический AI-ассистент")
    print("=" * 50)
    print("💡 Этот бот может помочь с вопросами по казахстанскому законодательству")
    print("📚 Использует RAG систему для поиска в юридических документах")
    print("🔍 Команды: 'quit' - выход, 'clear' - очистить историю, 'stats' - статистика")
    print("=" * 50)
    
    chatbot = LegalChatBot()
    
    # Show system stats
    stats = chatbot.get_system_stats()
    print(f"📊 Система готова! Векторов в индексе: {stats.get('total_vectors', 0)}")
    print()
    
    while True:
        try:
            user_input = input("👤 Вы: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 До свидания!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🧹 История очищена!")
                continue
            elif user_input.lower() == 'stats':
                stats = chatbot.get_system_stats()
                print(f"📊 Статистика системы:")
                print(f"   Векторов в индексе: {stats.get('total_vectors', 0)}")
                print(f"   Размерность: {stats.get('index_dimension', 0)}")
                print(f"   История разговора: {stats.get('conversation_history_length', 0)}")
                continue
            elif user_input.lower() == 'help':
                print("💡 Доступные команды:")
                print("   'quit' - выход из чата")
                print("   'clear' - очистить историю разговора")
                print("   'stats' - показать статистику системы")
                print("   'help' - показать эту справку")
                continue
            
            if not user_input:
                continue
            
            print("🤖 AI: ", end="", flush=True)
            
            # Determine if this is a legal question
            legal_keywords = [
                'закон', 'право', 'статья', 'кодекс', 'договор', 'суд', 'иск',
                'ответственность', 'обязательство', 'собственность', 'наследство',
                'брак', 'развод', 'алименты', 'трудовой', 'налог', 'административный',
                'уголовный', 'гражданский', 'конституция', 'постановление', 'приказ'
            ]
            
            is_legal_question = any(keyword in user_input.lower() for keyword in legal_keywords)
            
            # Get response
            if is_legal_question:
                result = chatbot.chat(user_input, use_rag=True)
                print(result["answer"])
                
                if result["sources"]:
                    print(f"\n📚 Источники: {', '.join(result['sources'])}")
                if result["results_count"] > 0:
                    print(f"🔍 Найдено релевантных документов: {result['results_count']}")
            else:
                result = chatbot.chat(user_input, use_rag=False)
                print(result["answer"])
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main() 