# OpenAI Chat Interface

This project provides two different ways to chat with OpenAI's GPT models:

## 1. Command Line Chat (`chat.py`)

A simple command-line interface for chatting with OpenAI.

### Features:
- Real-time conversation with GPT-4
- Conversation history management
- Special commands for clearing history and viewing history
- Easy to use terminal interface

### Usage:
```bash
python chat.py
```

### Commands:
- Type your message and press Enter to send
- Type `quit` to exit
- Type `clear` to clear conversation history
- Type `history` to view conversation history

## 2. Web Chat Interface (`web_chat.py`)

A beautiful web-based chat interface with a modern UI.

### Features:
- Modern, responsive web interface
- Real-time chat with typing indicators
- Export chat functionality
- Clear history option
- Mobile-friendly design

### Usage:
```bash
# Install Flask if not already installed
pip install flask

# Run the web server
python web_chat.py
```

Then open your browser and go to: `http://localhost:5000`

## Setup

1. Make sure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables in a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. Choose your preferred interface and run it!

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages (see requirements.txt)

## Features Comparison

| Feature | Command Line | Web Interface |
|---------|-------------|---------------|
| Ease of use | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Visual appeal | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Mobile friendly | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Export chat | ❌ | ✅ |
| Typing indicators | ❌ | ✅ |
| Conversation history | ✅ | ✅ |
| Clear history | ✅ | ✅ |

## Customization

### Changing the AI Model
You can change the AI model by modifying the `model` parameter in the ChatBot class:

```python
# For GPT-3.5-turbo (faster, cheaper)
chatbot = ChatBot(model="gpt-3.5-turbo")

# For GPT-4 (more capable, more expensive)
chatbot = ChatBot(model="gpt-4")
```

### Custom System Prompts
You can customize the AI's behavior by changing the system prompt:

```python
system_prompt = "You are a helpful coding assistant. Provide clear, concise code examples."
```

### Adjusting Parameters
You can modify various parameters like:
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = very random)
- `max_tokens`: Maximum length of response
- `max_history_length`: Number of conversation turns to remember

## Troubleshooting

### Common Issues:

1. **"Error: No module named 'flask'"**
   - Solution: `pip install flask`

2. **"Error: Invalid API key"**
   - Solution: Check your `.env` file and make sure your OpenAI API key is correct

3. **"Error: Rate limit exceeded"**
   - Solution: Wait a moment and try again, or upgrade your OpenAI plan

4. **Web interface not loading**
   - Solution: Make sure the Flask server is running and you're accessing the correct URL

## Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure
- The web interface is for development use - add authentication for production

## License

This project is open source and available under the MIT License. 