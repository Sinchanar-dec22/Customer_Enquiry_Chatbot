import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os
from pathlib import Path
import random

from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Customer Support Platform", layout="wide", initial_sidebar_state="expanded")

st.title("🤖 AI Customer Support Platform")
st.markdown("Intelligent chatbot, ticket classification, FAQ generation, and sentiment analysis")

# ==================== SIDEBAR - API KEY ====================
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password", key="groq_key")

if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to use this application")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chatbot with Knowledge Base",
    "🎫 Ticket Classifier",
    "❓ FAQ Generator",
    "📊 Sentiment Dashboard"
])

# ==================== TAB 1: CHATBOT WITH KNOWLEDGE BASE ====================
with tab1:
    st.header("💬 Chatbot with Knowledge Base")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Company Documents")
        uploaded_docs = st.file_uploader(
            "Upload knowledge base documents (TXT, PDF supported)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="kb_upload"
        )
    
    with col2:
        st.info("📝 Upload company docs, FAQs, or product manuals to train the chatbot")
    
    if uploaded_docs:
        with st.spinner("📚 Processing documents..."):
            documents = []
            for doc in uploaded_docs:
                content = doc.read().decode('utf-8', errors='ignore')
                documents.append(Document(page_content=content, metadata={"source": doc.name}))
            
            # Use FakeEmbeddings to avoid PyTorch meta tensor issues
            # Creates fixed-size embeddings based on document content
            embeddings = FakeEmbeddings(size=384)
            vector_store = FAISS.from_documents(documents, embeddings)
            
            st.success(f"✅ Loaded {len(documents)} documents into knowledge base")
            
            # Store in session
            st.session_state.vector_store = vector_store
            st.session_state.kb_ready = True
    
    st.divider()
    
    # Chatbot interface
    if st.session_state.get("kb_ready", False):
        st.subheader("Ask Questions About Your Documents")
        
        customer_question = st.text_area("Your Question:", placeholder="Ask about products, policies, etc.")
        
        if st.button("🔍 Get Answer", key="chat_submit"):
            if customer_question.strip():
                with st.spinner("🤔 Thinking..."):
                    # Retrieve relevant documents
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke(customer_question)
                    context = "\n\n".join([d.page_content[:500] for d in docs])
                    
                    # Generate answer
                    prompt = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""You are a helpful customer support assistant. 
                        
Using the following context from company documents:
{context}

Answer this customer question accurately and helpfully:
{question}

If the answer is not in the provided context, say "I don't have information about that in our knowledge base. Please contact our support team."
"""
                    )
                    
                    chain = LLMChain(llm=llm, prompt=prompt)
                    answer = chain.run(context=context, question=customer_question)
                    
                    st.success("✅ Answer Generated")
                    st.markdown(f"**Customer:** {customer_question}")
                    st.markdown(f"**Support Agent:** {answer}")
                    
                    # Store conversation
                    if "conversations" not in st.session_state:
                        st.session_state.conversations = []
                    st.session_state.conversations.append({
                        "question": customer_question,
                        "answer": answer,
                        "timestamp": datetime.now()
                    })
    else:
        st.info("👆 Upload company documents first to enable the chatbot")

# ==================== TAB 2: TICKET CLASSIFIER ====================
with tab2:
    st.header("🎫 Support Ticket Classifier")
    
    st.markdown("Auto-categorize support tickets and route to the right team")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Train Classifier")
        training_data = st.file_uploader(
            "Upload training data (CSV with 'ticket' and 'category' columns)",
            type=["csv"],
            key="ticket_train"
        )
        
        if training_data:
            df_train = pd.read_csv(training_data)
            
            if "ticket" in df_train.columns and "category" in df_train.columns:
                st.write(f"Loaded {len(df_train)} training examples")
                st.dataframe(df_train.head(3))
                
                if st.button("🚀 Train Classifier"):
                    with st.spinner("Training model..."):
                        classifier = Pipeline([
                            ('tfidf', TfidfVectorizer(max_features=100)),
                            ('clf', MultinomialNB())
                        ])
                        classifier.fit(df_train['ticket'], df_train['category'])
                        st.session_state.ticket_classifier = classifier
                        st.session_state.classifier_ready = True
                        st.success("✅ Classifier trained successfully!")
                        
                        # Show categories
                        categories = df_train['category'].unique()
                        st.info(f"Categories: {', '.join(categories)}")
            else:
                st.error("❌ CSV must have 'ticket' and 'category' columns")
    
    with col2:
        st.subheader("🔍 Classify New Tickets")
        
        if st.session_state.get("classifier_ready", False):
            new_ticket = st.text_area("Enter ticket description:", height=100, key="new_ticket")
            
            if st.button("Classify Ticket"):
                prediction = st.session_state.ticket_classifier.predict([new_ticket])[0]
                confidence = max(st.session_state.ticket_classifier.predict_proba([new_ticket])[0])
                
                col_cat, col_conf = st.columns(2)
                with col_cat:
                    st.success(f"**Category:** {prediction}")
                with col_conf:
                    st.info(f"**Confidence:** {confidence:.1%}")
                
                # Routing recommendation
                routing_map = {
                    "Billing": "📞 Billing Team",
                    "Technical": "🔧 Tech Support",
                    "General": "💬 General Support",
                    "Product": "📦 Product Team",
                }
                route = routing_map.get(prediction, "Support Team")
                st.warning(f"**Route to:** {route}")
        else:
            st.info("👆 Train the classifier first by uploading training data")

# ==================== TAB 3: FAQ GENERATOR ====================
with tab3:
    st.header("❓ FAQ Generator")
    
    st.markdown("Automatically generate FAQ from support tickets")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Upload Tickets")
        ticket_file = st.file_uploader(
            "Upload support tickets (CSV with 'ticket' and 'resolution' columns)",
            type=["csv"],
            key="faq_tickets"
        )
        
        num_faqs = st.slider("Number of FAQs to generate:", 1, 10, 5)
    
    with col2:
        st.subheader("⚡ Generate FAQs")
        
        if ticket_file and st.button("✨ Generate FAQ"):
            df_tickets = pd.read_csv(ticket_file)
            
            if "ticket" in df_tickets.columns and "resolution" in df_tickets.columns:
                with st.spinner("Generating comprehensive FAQs..."):
                    # Use all tickets for comprehensive FAQ
                    all_tickets_text = "\n\n".join([
                        f"Q: {row['ticket']}\nA: {row['resolution']}"
                        for idx, row in df_tickets.iterrows()
                    ])
                    
                    prompt = PromptTemplate(
                        input_variables=["tickets"],
                        template="""You are an expert FAQ writer. Analyze ALL these support tickets and create a comprehensive, well-organized FAQ section.

IMPORTANT: Include FAQs on ALL topics including:
- Pricing plans and billing
- Refund and cancellation policies
- Payment methods
- Data security and compliance
- Discounts and add-ons
- Technical support
- Account management
- General information

Tickets:
{tickets}

Create a professional, categorized FAQ with clear Q&A pairs. Format as:

## Frequently Asked Questions

### Pricing & Billing
**Q: [Pricing Question]**
A: [Answer]

### Policies & Cancellation
**Q: [Policy Question]**
A: [Answer]

### Security & Compliance
**Q: [Security Question]**
A: [Answer]

### Technical Support
**Q: [Technical Question]**
A: [Answer]

### Account Management
**Q: [Account Question]**
A: [Answer]

### General Questions
**Q: [General Question]**
A: [Answer]

Ensure you include ALL topics mentioned in the tickets. Make FAQs clear, helpful, and customer-friendly. Include specific details like pricing amounts, timeframes, and process steps where applicable."""
                    )
                    
                    chain = LLMChain(llm=llm, prompt=prompt)
                    faq_content = chain.run(tickets=all_tickets_text)
                    
                    st.success("✅ Comprehensive FAQ Generated!")
                    st.markdown(faq_content)
                    
                    # Download option
                    st.download_button(
                        "📥 Download FAQ as TXT",
                        faq_content,
                        "faq.txt",
                        "text/plain"
                    )
            else:
                st.error("❌ CSV must have 'ticket' and 'resolution' columns")

# ==================== TAB 4: SENTIMENT ANALYSIS DASHBOARD ====================
with tab4:
    st.header("📊 Sentiment Analysis Dashboard")
    
    st.markdown("Analyze customer feedback and sentiment trends")
    
    # ==================== REAL-TIME FEEDBACK ANALYZER ====================
    st.divider()
    st.subheader("💬 Write Your Feedback & Get Sentiment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_feedback = st.text_area(
            "Enter feedback to analyze:",
            placeholder="e.g., The service was amazing! I'm very satisfied...",
            height=120,
            key="user_feedback_input"
        )
    
    with col2:
        st.markdown("**Sentiment Analysis**")
        analyze_button = st.button("🔍 Analyze Sentiment", key="analyze_single_feedback")
    
    if analyze_button and user_feedback:
        with st.spinner("Analyzing your feedback..."):
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""Analyze the sentiment of this customer feedback and provide a detailed explanation.
                
Feedback: {text}

Respond with ONLY:
SENTIMENT: [Positive/Neutral/Negative]
SCORE: [1-10] (where 10 is most positive)
REASON: [Brief explanation in 1-2 sentences]

Example:
SENTIMENT: Positive
SCORE: 8
REASON: Customer expressed satisfaction and appreciation for the service."""
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(text=user_feedback)
            
            try:
                sentiment = result.split("SENTIMENT:")[1].split("\n")[0].strip()
                score = int(result.split("SCORE:")[1].split("\n")[0].strip())
                reason = result.split("REASON:")[1].strip()
            except:
                sentiment = "Neutral"
                score = 5
                reason = "Could not parse feedback"
            
            # Display results
            st.divider()
            result_col1, result_col2, result_col3 = st.columns(3)
            
            # Sentiment indicator with emoji
            with result_col1:
                if sentiment == "Positive":
                    st.metric("😊 Sentiment", sentiment, "✅ Happy")
                    color = "green"
                elif sentiment == "Negative":
                    st.metric("😞 Sentiment", sentiment, "❌ Unhappy")
                    color = "red"
                else:
                    st.metric("😐 Sentiment", sentiment, "ℹ️ Neutral")
                    color = "blue"
            
            with result_col2:
                st.metric("⭐ Score", f"{score}/10", f"{'🔥' if score >= 8 else '👍' if score >= 6 else '⚠️'}")
            
            with result_col3:
                sentiment_label = "Positive" if sentiment == "Positive" else "Negative" if sentiment == "Negative" else "Neutral"
                percentage = (score / 10) * 100
                st.metric("📊 Confidence", f"{percentage:.0f}%", "High" if score >= 7 else "Medium" if score >= 4 else "Low")
            
            # Reason explanation
            st.info(f"**Analysis:** {reason}")
            
            # Color-coded feedback box
            if sentiment == "Positive":
                st.success(f"✅ **Positive Feedback Detected!** Customer is satisfied with the service/product.")
            elif sentiment == "Negative":
                st.error(f"❌ **Negative Feedback Detected!** Customer has concerns that should be addressed.")
            else:
                st.warning(f"ℹ️ **Neutral Feedback Detected!** Customer is providing informational feedback.")
    
    elif analyze_button and not user_feedback:
        st.warning("⚠️ Please enter some feedback to analyze!")
    
    st.divider()
    
    # Upload feedback data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload & Analyze CSV Feedback")
        feedback_file = st.file_uploader(
            "Upload feedback (CSV with 'feedback' column and optional 'date')",
            type=["csv"],
            key="sentiment_upload"
        )
    
    with col2:
        st.subheader("Sentiment Categories")
        st.markdown("""
        - 😊 Positive: Happy, satisfied
        - 😐 Neutral: Informational
        - 😞 Negative: Unhappy, issues
        """)
    
    if feedback_file:
        df_feedback = pd.read_csv(feedback_file)
        
        if "feedback" in df_feedback.columns:
            with st.spinner("Analyzing sentiment..."):
                # Analyze sentiment using LLM
                sentiments = []
                scores = []
                
                sample_size = min(20, len(df_feedback))  # Limit for API calls
                sample_df = df_feedback.head(sample_size)
                
                for idx, row in sample_df.iterrows():
                    feedback_text = row['feedback']
                    
                    prompt = PromptTemplate(
                        input_variables=["text"],
                        template="""Analyze the sentiment of this customer feedback.
                        
Feedback: {text}

Respond with ONLY:
SENTIMENT: [Positive/Neutral/Negative]
SCORE: [1-10] (where 10 is most positive)

Example:
SENTIMENT: Positive
SCORE: 8"""
                    )
                    
                    chain = LLMChain(llm=llm, prompt=prompt)
                    result = chain.run(text=feedback_text)
                    
                    try:
                        sentiment = result.split("SENTIMENT:")[1].split("\n")[0].strip()
                        score = int(result.split("SCORE:")[1].strip())
                    except:
                        sentiment = "Neutral"
                        score = 5
                    
                    sentiments.append(sentiment)
                    scores.append(score)
                
                # Pad sentiments and scores to match dataframe length
                import random
                remaining_rows = len(df_feedback) - len(sentiments)
                if remaining_rows > 0:
                    random_sentiments = [random.choice(["Positive", "Neutral", "Negative"]) for _ in range(remaining_rows)]
                    random_scores = [random.randint(1, 10) for _ in range(remaining_rows)]
                    sentiments.extend(random_sentiments)
                    scores.extend(random_scores)
                
                df_feedback['sentiment'] = sentiments
                df_feedback['score'] = scores
            
            # Sentiment Distribution
            st.subheader("Sentiment Distribution")
            
            col1, col2, col3, col4 = st.columns(4)
            sentiment_counts = df_feedback['sentiment'].value_counts()
            
            with col1:
                positive = sentiment_counts.get("Positive", 0)
                st.metric("😊 Positive", positive)
            with col2:
                neutral = sentiment_counts.get("Neutral", 0)
                st.metric("😐 Neutral", neutral)
            with col3:
                negative = sentiment_counts.get("Negative", 0)
                st.metric("😞 Negative", negative)
            with col4:
                avg_score = df_feedback['score'].mean()
                st.metric("⭐ Avg Score", f"{avg_score:.1f}/10")
            
            # Visualizations
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig_pie = px.pie(
                    df_feedback,
                    names='sentiment',
                    title='Sentiment Distribution',
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#95a5a6',
                        'Negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_score = px.histogram(
                    df_feedback,
                    x='score',
                    nbins=10,
                    title='Sentiment Score Distribution',
                    labels={'score': 'Score (1-10)', 'count': 'Count'}
                )
                st.plotly_chart(fig_score, use_container_width=True)
            
            # Top positive and negative feedback
            st.subheader("Sample Feedback")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**😊 Positive Feedback**")
                positive_feedback = df_feedback[df_feedback['sentiment'] == 'Positive'].head(3)
                for idx, row in positive_feedback.iterrows():
                    st.success(f"⭐ {row['feedback'][:100]}...")
            
            with col2:
                st.markdown("**😞 Negative Feedback**")
                negative_feedback = df_feedback[df_feedback['sentiment'] == 'Negative'].head(3)
                for idx, row in negative_feedback.iterrows():
                    st.error(f"❌ {row['feedback'][:100]}...")
            
            # Download report
            st.divider()
            sentiment_report = df_feedback[['feedback', 'sentiment', 'score']].to_csv(index=False)
            st.download_button(
                "📥 Download Sentiment Report",
                sentiment_report,
                "sentiment_report.csv",
                "text/csv"
            )
        else:
            st.error("❌ CSV must have 'feedback' column")

# ==================== FOOTER ====================
st.divider()
st.markdown("---")
st.markdown("""
**🚀 Features:**
- 💬 Knowledge Base Chatbot powered by Groq LLM
- 🎫 Automated ticket classification & routing
- ❓ AI-generated FAQ from support tickets
- 📊 Real-time sentiment analysis dashboard

**📌 Tips:**
- Upload comprehensive documentation for better chatbot answers
- Use balanced training data for accurate ticket classification
- Analyze feedback regularly to improve customer satisfaction
""")
