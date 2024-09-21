# from langchain_community.document_loaders import CSVLoader
import os
import yaml
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities import LoginError
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

with open('./credentials.yml') as file:
    credentials_config = yaml.load(file, Loader=SafeLoader)

@st.cache_data
def load_data(file_path = 'salaries.csv'):
    try:
        return pd.read_csv(file_path)
    except:
        return None

def handle_error(error):
    print("Got Error:")
    print(error)
    return "Missing context, unable to understand question."

@st.cache_data
def aggregate_yearly_data(df):
    return df.astype({'work_year': str}) \
            .groupby('work_year') \
            .agg(job_count=('work_year', 'size'), average_salary=('salary_in_usd', 'mean')) \
            .round({'average_salary': 0}) \
            .reset_index()

@st.cache_data
def aggregate_job_data(df):
    return df.astype({'work_year': str}).groupby(['work_year', 'job_title']).agg(job_count=('work_year', 'size')).reset_index()

def show_main_view(name):
    st.write(f'Welcome *{name}*')
    st.session_state["llm_model"] = "llama3-8b-8192"

    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.secrets("GROQ_API_KEY")

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

    if api_key:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        llm = ChatGroq(model=st.session_state.llm_model, api_key=api_key)
        agent_executor = create_pandas_dataframe_agent(llm, raw_df, verbose=False, agent_executor_kwargs={"handle_parsing_errors": handle_error}, allow_dangerous_code=True)

        if prompt := st.chat_input("Enter your question here..."):
            assistant_messages_container = st.chat_message("assistant")
            user_messages_container = st.chat_message("user")

            user_messages_container.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                response = agent_executor.invoke(prompt)
                assistant_messages_container.markdown(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            except Exception as e:
                st.error(e)

authenticator = stauth.Authenticate(
    credentials_config['credentials'],
    credentials_config['cookie']['name'],
    credentials_config['cookie']['key'],
    credentials_config['cookie']['expiry_days'],
    credentials_config['preauthorized'],
    None,
    False
)

try:
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    authenticator.logout()
    show_main_view(st.session_state["name"])
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
