# voice

# 🎤 Real-Time Voice-to-Text Chatbot

A real-time voice chatbot system that uses OpenAI's **Whisper-1** for speech-to-text conversion and **GPT-4o-mini** for intelligent responses.

## 📋 Features

- **Real-time voice input** using your microphone
- **Speech-to-text conversion** via OpenAI Whisper-1
- **Intelligent responses** using GPT-4o-mini
- **Conversation memory** maintains context throughout the session
- **Simple controls** - just press ENTER to record
- **Comprehensive testing** with automated test suite

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create the project directory
mkdir voice-chatbot && cd voice-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
STT_MODEL=whisper-1
CHAT_MODEL=gpt-4o-mini
```

### 3. Test the Setup (Streamlit Dashboard)

```bash
# Run the interactive test dashboard
streamlit run test.py
```

This will open a web-based test dashboard in your browser where you can:
- Run comprehensive system tests
- View detailed test results  
- Test chat functionality interactively
- Monitor system status

### 4. Start the Voice Chatbot

```bash
# Launch the voice chatbot
python voice.py
```

## 🎯 How to Use

1. **Launch the app**: `python voice.py`
2. **Start recording**: Press ENTER when prompted
3. **Speak**: Talk normally into your microphone
4. **Stop recording**: Press ENTER again
5. **Get response**: The bot will transcribe your speech and respond
6. **Continue conversation**: Repeat the process
7. **Exit**: Type 'quit' and press ENTER

## 🗂️ File Structure

```
project/
├── .env                 # API keys and configuration
├── voice.py            # Main voice chatbot application  
├── test.py             # Comprehensive test suite
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 System Requirements

- **Python 3.7+**
- **Microphone access**
- **OpenAI API key**
- **Internet connection**

### Supported Platforms
- Windows 10/11
- macOS 10.14+
- Linux (most distributions)

## 🧪 Testing

The included `test.py` is a **Streamlit web dashboard** that provides:

- 🌐 **Interactive web interface** for testing
- 📊 **Visual test results** with progress tracking
- 🔄 **Real-time status updates**
- 🎯 **Individual or comprehensive testing**
- 💬 **Interactive chat testing** without voice

### Launch Test Dashboard:
```bash
streamlit run test.py
```

The dashboard checks:
- ✅ Environment variables setup
- ✅ Required dependencies installation  
- ✅ Audio device availability
- ✅ OpenAI API connectivity
- ✅ Voice chatbot functionality

## ⚙️ Configuration Options

You can customize behavior in the `.env` file:

```bash
# Required
OPENAI_API_KEY=your_key_here

# Model Selection  
STT_MODEL=whisper-1              # Speech-to-text model
CHAT_MODEL=gpt-4o-mini           # Chat response model

# Optional Audio Settings (defaults in code)
SAMPLE_RATE=44100                # Audio sample rate
CHANNELS=1                       # Mono audio
MIN_RECORDING_DURATION=2.0       # Minimum recording length
```

## 🔊 Audio Requirements

- **Microphone**: Any standard microphone or headset
- **Sample Rate**: 44.1kHz (default)
- **Channels**: Mono (1 channel)
- **Format**: 16-bit WAV (internal conversion)
- **Minimum Duration**: 2 seconds per recording

## 💡 Tips for Best Results

1. **Speak clearly** and at normal pace
2. **Minimize background noise** when possible  
3. **Record at least 2-3 seconds** of speech
4. **Wait for processing** between recordings
5. **Use good microphone** for better transcription

## 🐛 Troubleshooting

### Common Issues

**"No audio devices found"**
- Check microphone is connected and working
- Grant microphone permissions to terminal/Python

**"API key error"**  
- Verify your OpenAI API key in `.env` file
- Ensure key starts with `sk-`
- Check API key has sufficient credits

**"Module not found"**
- Install requirements: `pip install -r requirements.txt`
- Use virtual environment if needed

**"Recording too short"**
- Speak for at least 2 seconds
- Check microphone is working properly

### Audio Issues on Different OS

**Windows**: May need to install Microsoft C++ Build Tools for pyaudio
**macOS**: Grant microphone permissions in System Preferences
**Linux**: Install ALSA development packages: `sudo apt-get install libasound2-dev`

## 🔒 Privacy & Security

- **Voice data**: Sent to OpenAI for processing only
- **Conversation history**: Stored locally during session only  
- **API key**: Keep your `.env` file private
- **No persistent storage**: Data cleared when app closes

## 📊 Models Used

- **Speech-to-Text**: `whisper-1`
  - High accuracy transcription
  - Supports multiple languages
  - Optimized for real-time use

- **Chat Response**: `gpt-4o-mini`  
  - Fast response times
  - Context-aware conversations
  - Cost-effective for frequent use

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the voice chatbot system.

## 📄 License

This project is open source. Please ensure you comply with OpenAI's usage policies when using their APIs.