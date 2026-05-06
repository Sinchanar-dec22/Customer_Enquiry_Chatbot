import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI SQL Data Analyst", layout="wide")

st.title("🤖 AI SQL Data Analyst Agent")
st.markdown("Upload CSV → Ask Questions → Get Insights + Charts")

# ------------------- SIDEBAR -------------------
st.sidebar.header("⚙️ Settings")

groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:

    # ------------------- LOAD DATA -------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ------------------- CREATE SQLITE DB -------------------
    conn = sqlite3.connect("data.db")
    df.to_sql("data_table", conn, if_exists="replace", index=False)

    db = SQLDatabase.from_uri("sqlite:///data.db")

    st.success("✅ Data successfully loaded into SQL database")

    # ------------------- QUESTION INPUT -------------------
    question = st.text_input("💬 Ask a question about your data")

    if question and groq_api_key:

        try:
            # ------------------- LLM SETUP -------------------
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name = "llama-3.1-8b-instant"
            )

            # ------------------- SQL AGENT -------------------
            agent = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="openai-tools",
                verbose=True
            )

            # ------------------- RUN QUERY -------------------
            response = agent.run(question)

            # ------------------- OUTPUT -------------------
            st.subheader("✅ Answer")
            st.write(response)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    elif question and not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in sidebar")

    # ------------------- VISUALIZATION -------------------
    st.subheader("📈 Quick Visualization")

    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) >= 1:
        col1 = st.selectbox("Select X-axis", df.columns)
        col2 = st.selectbox("Select Y-axis", numeric_cols)

        chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Scatter"])

        if st.button("Generate Chart"):

            if chart_type == "Line":
                fig = px.line(df, x=col1, y=col2)

            elif chart_type == "Bar":
                fig = px.bar(df, x=col1, y=col2)

            else:
                fig = px.scatter(df, x=col1, y=col2)

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No numeric columns available for visualization")

else:
    st.info("👆 Upload a CSV file to get started")