import streamlit as st

#Added Packages
from vn_querySystem4 import VN_QuerySystem
from dotenv import load_dotenv
from qdrant_client import QdrantClient


#Added for setup_vanna()
load_dotenv('.env')

def get_qdrantClient():
    client = QdrantClient(
        url=st.secrets.get('QDRANT_URL'),
        api_key=st.secrets.get('QDRANT_API_KEY')
    )
    return client

def masked_n_results(n_results:int) -> int:
    if n_results == 0: return 11
    else: return n_results

def get_config():
    config ={
        "client":get_qdrantClient(),
        "fastembed_model":'BAAI/bge-base-en-v1.5',
        "n_results": masked_n_results(5),
        "sql_collection_name" : 'bge-base-en-v1.5_sql_w_sqlSkeleton',
        "ddl_collection_name" : 'experiment_ddl',
        "api_key": st.secrets.get('OpenAI_API_KEY_DEV'),
        "model": 'gpt-4o-mini'
        }
    return config

#VannaStreamlit
@st.cache_resource(ttl=3600)
def setup_vanna():
    vn = VN_QuerySystem(config=get_config())
    vn.connect_to_sqlite(st.secrets.get('dbLoc') + '/Chinook/Chinook.sqlite')
    return vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    sql, message = vn.generate_and_correct_sql(question=question, db_id='Chinook')
    if message == "Not a text-to-sql-question":
        return
    else:
        db_path= st.secrets.get('dbLoc') + '/Chinook/Chinook.sqlite'
        executable, revised_sql, _ = vn.check_sql_for_release(sql, db_path)
        if executable:
            return revised_sql
        else:
            return

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    try:
        df = vn.run_sql(sql=sql)
    except:
        df = None
    return df

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code

@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)

@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    try:
        summary = vn.generate_summary(question=question, df=df)
    except:
        summary = None
    return summary