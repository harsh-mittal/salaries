# from langchain_community.document_loaders import CSVLoader
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

@st.cache_data
def load_data(file_path = 'salaries.csv'):
    try:
        return pd.read_csv(file_path)
    except:
        return None

@st.cache_data
def aggregate_yearly_data(df):
    return df.astype({'work_year': str}).groupby('work_year').agg(job_count=('work_year', 'size'), average_salary=('salary_in_usd', 'mean')).reset_index()

@st.cache_data
def aggregate_job_data(df):
    return df.astype({'work_year': str}).groupby(['work_year', 'job_title']).agg(job_count=('work_year', 'size')).reset_index()

load_dotenv()

raw_df = load_data()
yearly_df = aggregate_yearly_data(raw_df)
job_df = aggregate_job_data(raw_df)

st.title('Salary Analysis')

event = st.dataframe(yearly_df, selection_mode="single-row", on_select='rerun', hide_index=True)

selected_row_idx = event.selection.rows
selected_row = yearly_df.iloc[selected_row_idx]

if selected_row_idx:
    st.dataframe(job_df[job_df['work_year'].isin(selected_row['work_year'])], hide_index=True)

st.line_chart(yearly_df, x='work_year', y='average_salary')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    assistant_messages_container = st.chat_message("assistant")
    user_messages_container = st.chat_message("user")

    user_messages_container.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    assistant_messages_container.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# loader = CSVLoader(file_path="salaries.csv")
# documents = loader.load()

# embeddings = OpenAIEmbeddings()

# print(type(documents[0]))
# print(documents[0])