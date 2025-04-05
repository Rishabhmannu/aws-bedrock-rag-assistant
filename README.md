# 🤖 BedrockPDF Assistant – Chat with Documents using AWS AI

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful RAG application leveraging AWS Bedrock's generative AI capabilities to enable natural language interactions with PDF documents. Built with ❤️ using Claude 3.5 Sonnet and Streamlit.


## 🚀 Features

- **PDF Document Intelligence**: Ingest and process multiple PDFs simultaneously
- **AWS Bedrock Integration**: Leverage state-of-the-art Claude 3.5 Sonnet model
- **Smart Vector Storage**: FAISS-based efficient document embedding system
- **Throttling Protection**: Built-in error handling & request retry mechanisms
- **Multi-Model Ready**: Architecture designed for easy LLM switching (Claude 3/Llama2)
- **Security First**: Safe deserialization handling for vector store operations
- **Responsive UI**: Streamlit-powered intuitive web interface

## 📦 Dependencies

### Core Components
- Python 3.8+
- AWS Account with Bedrock Access
- `streamlit` (Web Interface)
- `boto3` (AWS SDK)
- `langchain-community` (LLM Orchestration)
- `langchain-aws` (Bedrock Integrations)
- `pypdf` (PDF Processing)
- `faiss-cpu` (Vector Storage)

### Optional
- `watchdog` (File system monitoring - for better Streamlit performance)

## 🛠️ Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/bedrock-pdf-chat.git
cd bedrock-pdf-chat
```

2. **Install Requirements**
```bash
pip install -r requirements.txt
```

3. **AWS Configuration**  
   Set up your AWS credentials either through:
   - `~/.aws/credentials` file
   - Environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=<YOUR_KEY>
   export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET>
   export AWS_DEFAULT_REGION=us-east-1
   ```

4. **Prepare Documents**
```bash
mkdir data  # Place your PDF files here
```

5. **Launch Application**
```bash
streamlit run main.py
```

## 🖥️ Usage

1. **Upload PDFs**  
   Place documents in the `data` directory
2. **Initialize Vector Store**  
   Click "Vectors Update" in sidebar
3. **Start Chatting**  
   Enter questions in the main interface
4. **Switch Models**  
   Use different LLM buttons (Claude 3/Llama2)

## 🔧 Configuration

### Environment Variables
```env
AWS_PROFILE=default  # Optional: If using named profiles
FAISS_INDEX_PATH=faiss_index  # Vector store location
DATA_DIR=data  # PDF storage directory
```

### Advanced Options
- Modify `main.py` to:
  - Adjust chunking strategy (lines 62-65)
  - Change LLM parameters (temperature, max_tokens)
  - Enable additional models (Llama2 support)

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- AWS Bedrock Team
- Anthropic for Claude 3 models
- Streamlit for amazing UI framework
- LangChain community for RAG patterns
