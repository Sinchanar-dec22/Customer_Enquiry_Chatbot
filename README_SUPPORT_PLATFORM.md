# 🤖 Customer Support Platform

A comprehensive AI-powered customer service solution built with Streamlit, LangChain, and Groq LLM.

## 🎯 Features

### 1. **💬 Chatbot with Knowledge Base**
- Upload company documents (TXT, PDF)
- RAG-based retrieval for accurate answers
- Contextual responses from your documentation
- Perfect for product manuals, FAQs, policies

### 2. **🎫 Support Ticket Classifier**
- Automated ticket categorization
- Smart routing to appropriate teams (Billing, Technical, General, Product)
- ML-powered classification with confidence scores
- Reduce response time with auto-routing

### 3. **❓ FAQ Generator**
- Auto-generate FAQs from support tickets
- Learn from past resolutions
- Create professional FAQ documentation
- Reduce repetitive support tickets

### 4. **📊 Sentiment Analysis Dashboard**
- Analyze customer feedback in real-time
- Track positive, neutral, and negative sentiment
- Visual dashboards with sentiment distribution
- Export sentiment reports for analysis

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (get free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone/Download the project**
```bash
cd customer-support-platform
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run customer_support_platform.py
```

4. **Enter your Groq API Key** in the sidebar when prompted

## 📊 Sample Data

We've included sample CSV files to test each feature:

- **sample_training_tickets.csv** - Training data for ticket classifier
- **sample_faq_tickets.csv** - Tickets with resolutions for FAQ generation
- **sample_feedback.csv** - Customer feedback for sentiment analysis

## 📋 Usage Guide

### Tab 1: Chatbot with Knowledge Base
1. Upload company documents (TXT or PDF)
2. Ask questions about the documents
3. Get contextual answers from your knowledge base

**Best for:** Product docs, company policies, service guides

### Tab 2: Ticket Classifier
1. Upload training data (CSV: ticket, category)
2. Click "Train Classifier"
3. Paste new ticket descriptions to auto-classify
4. See classification confidence and routing recommendation

**Categories:** Billing, Technical, General, Product

### Tab 3: FAQ Generator
1. Upload support tickets (CSV: ticket, resolution)
2. Select number of FAQs to generate
3. Click "Generate FAQ"
4. Download generated FAQ as TXT

**Use case:** Create customer-facing FAQ documentation

### Tab 4: Sentiment Dashboard
1. Upload customer feedback (CSV: feedback, optional: date)
2. Analyzes sentiment automatically
3. View distribution charts
4. See sample positive/negative feedback
5. Export sentiment report

## 🔧 Customization

### Add More Ticket Categories
Edit the `routing_map` in Tab 2:
```python
routing_map = {
    "Billing": "📞 Billing Team",
    "Technical": "🔧 Tech Support",
    "General": "💬 General Support",
    "Product": "📦 Product Team",
    "Custom": "🎯 Custom Team"  # Add more
}
```

### Change LLM Model
In the sidebar setup, change:
```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"  # Change model
)
```

## 🔑 Environment Variables (Optional)

Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

Then load in the app:
```python
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
```

## 📦 Requirements

- **streamlit** - Web app framework
- **langchain** - LLM orchestration
- **langchain-groq** - Groq integration
- **faiss-cpu** - Vector database for RAG
- **scikit-learn** - Machine learning for classifier
- **plotly** - Interactive visualizations
- **pandas** - Data processing
- **numpy** - Numerical computing

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Groq Console](https://console.groq.com/)
- [FAISS Vector Store](https://github.com/facebookresearch/faiss)

## 🤝 Future Enhancements

- [ ] Multi-language support
- [ ] Integration with Slack/Teams
- [ ] Email ticket ingestion
- [ ] Advanced NLP preprocessing
- [ ] Database persistence for chat history
- [ ] Analytics and reporting dashboard
- [ ] Custom model training
- [ ] API endpoint deployment

## 📝 License

Free to use and modify

## 🆘 Troubleshooting

**Q: "Module not found" error**
```bash
pip install -r requirements.txt --upgrade
```

**Q: Slow responses**
- Reduce sample size in sentiment analysis
- Use a faster model: `mixtral-8x7b-32768`

**Q: API rate limits**
- Groq has generous free tier
- Check [console.groq.com](https://console.groq.com/) for limits

**Q: FAISS issues on Windows**
```bash
pip install faiss-cpu --no-binary :all:
```

---

**Built with ❤️ using Streamlit, LangChain, and Groq**
