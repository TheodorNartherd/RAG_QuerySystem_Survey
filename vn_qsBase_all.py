from vanna.qdrant import Qdrant_VectorStore
from vanna.openai import OpenAI_Chat

from vanna.types import TrainingPlan
from dotenv import load_dotenv
import os

import sqlite3
import pandas as pd

import openai_cookbook as oc

import re
import ast

load_dotenv('.env')

class VN_QsBase(Qdrant_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

        self.model = config.get("model")

    # Prompt-Modul
    ## Table Representation
    def convert_ddlToSchema(self,ddl:str) -> str:
        chrList = ['"',"'","`", '\t', '\\t']
        for chr in chrList:
            ddl = ddl.replace(chr,' ')

        code_lines = ddl.split('\n')
        schema = code_lines[0].strip().removeprefix('CREATE TABLE')
        code_lines = code_lines[1:]
        for cl in code_lines:
            if cl.startswith(','):
                cl = cl.replace(',', '')
            cl = cl.strip()
            if cl.upper().startswith('PRIMARY KEY'):
                continue
            if ('FOREIGN KEY' in cl.upper()  or len(cl)==1): 
                schema +=cl.replace('( ', '(').replace(' )', ')') + ' '
            else:
                firstWord = cl.split(' ')[0]
                schema += firstWord + ', '

        if not(schema.strip().endswith(')')):
            schema = schema + ')'
        result = ')'.join(schema.rsplit(', )', 1)).strip()
    
        return result
    
    def get_exampleValues(self, tbl_name, db_name) -> str:
        self.connect_to_sqlite(os.getenv('dbLoc') + '/' + db_name + '/' + db_name + '.sqlite')
        sql = 'SELECT * FROM ' + tbl_name + ' ORDER BY RANDOM() LIMIT 2'
        example_dict = self.run_sql(sql).to_dict('list')
        return 'Value-Examples: ' + str(example_dict).replace('{', '').replace('}', '')

    def add_ddl(self, ddl: str, **kwargs) -> str:
        schema = self.convert_ddlToSchema(ddl)
        exampleValues = self.get_exampleValues(kwargs.get('tbl_name'), kwargs.get('db_name'))
        schema += ' \n' + exampleValues
        return super().add_ddl(schema)
    
    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
        **kwargs
    ) -> str:
        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl,**kwargs)
        else: 
            super().train(question,sql,documentation,plan)

    
    ## Prompt Representation
    #   Override of VannaBase. Newly implemented according to (Gao et al., 2023)
    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):

        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. Complete {self.dialect} SQL query only and with no explanation. \n" #Instead of '*' always name all the columns. If there are duplicate column names, use aliases by using the table-name as a prefix with an '_' as a seperator. \n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]

        if len(question_sql_list) > 0:
            message_log.append(self.system_message('Some example questions and corresponding SQL queries are provided based on similar problems:'))

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.system_message("Answer the following:"))
        message_log.append(self.user_message(question))

        return message_log
    
    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += '\n' + f"{self.dialect} SQL tables, with their properties:\n\n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n"

        return initial_prompt
    """
    # Schema-Linking-Modul
    def get_related_ddl(self, question: str, **kwargs) -> list:
        keywordList = self.get_keywords(question)
        ddl_list = []
        for keyword in keywordList:
            results = self._client.query_points(
            self.ddl_collection_name,
            query=self.generate_embedding(keyword),
            limit=2,
            with_payload=True,
            ).points
            result_list = [result.payload["ddl"] for result in results]
            ddl_list.extend(result_list)

        return list(set(ddl_list))

    def get_keywords(self,question):
        keywords = self.kw_model.extract_keywords(question)
        res_list = [x[0] for x in keywords]
        return res_list
    """
    
    # Correction-Modul
    def generate_and_correct_sql(self, question: str, **kwargs) -> str:
        sql = self.generate_sql(question, **kwargs)
        if not self.is_sql_valid(sql):
            return "", "Not a text-to-sql-question"

        db_id = kwargs.get('db_id')
        db_path = os.getenv('dbLoc') + '/' + db_id + '/' + db_id + '.sqlite'
        executable, message = self.check_sql(sql,db_path)

        if executable:
            return sql, message
        else:
            self.log(title="SQL Correction needed: 1. Attempt", message=message)
            return self.correct_sql(question, sql, message, db_path, 1)
        

    def check_sql(self,predicted_sql,db_path):
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes
        cursor = conn.cursor()
        try:
            cursor.execute(predicted_sql)
        except Exception as e:
            return False, str(e)
    
        predicted_res = cursor.fetchall()
        if len(predicted_res) > 0:
            return True, None
        else:
            return False, "sql returns no value"
        
    def correct_sql(self, question, sql, message, db_path, attempt, **kwargs) -> str:
        correction_prompt = self.get_correction_prompt(question, sql, message, **kwargs)
        self.log(title="Correction Prompt", message=correction_prompt)
        corrected_llm_response = self.submit_prompt(correction_prompt)
        corrected_sql = self.extract_dict_value(corrected_llm_response, "corrected_SQL")
        executable, new_message = self.check_sql(corrected_sql, db_path)
        if executable or attempt == 3:
            return self.extract_sql(corrected_sql), new_message
        else:
            self.log(title="SQL Correction needed: " + str(attempt + 1) + '. Attempt', message=new_message)
            return self.correct_sql(question, corrected_sql, new_message, db_path, attempt + 1, **kwargs) 

    def get_correction_prompt(self, question, sql, message, **kwargs) -> str:
        initial_prompt = f"You are a {self.dialect} expert. There is a SQL-Query generated based on the following Database Schema to respond to the Question. Executing this SQL-Query has resulted in an error and you need to fix it based on the Error-Message. \n"
        
        ddl_list = self.get_all_ddl()
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt += '\n' + f"Question:\n{question} \n"
        initial_prompt += '\n' + f"Executed SQL:\n{sql} \n"
        initial_prompt += '\n' + f"Error-Message:\n{message} \n"

        initial_prompt += '\n Please respond with a Python-Dictionary storing the two keys "chain_of_thought_reasoning" and "corrected_SQL" written as a one-liner. \n'

        message_log = [self.system_message(initial_prompt)]

        return message_log
    
    def extract_dict_value(self, llm_response: str, key: str):

        pattern = r'\{.*?\}'

        matches = re.findall(pattern, llm_response, re.DOTALL)
        for match in matches:
            try:
                result = ast.literal_eval(match)
                if isinstance(result, dict):
                    return result[key]
            except (ValueError, SyntaxError):
                continue

        print("No valid dictionary found.")
        return None


    # Added to stay within the context-window
    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        summary_prompt = self.get_summary_prompt(question, df, **kwargs)
        summary = self.submit_prompt(summary_prompt, **kwargs)
        return summary

    def get_summary_prompt(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df[:100].to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]
        num_tokens = oc.num_tokens_from_messages(message_log, self.model)
        context_window = self.get_context_window(self.model)
        if num_tokens < (context_window / 2):
            return message_log
        else: 
            row = len(df)
            reduce = round(row * 0.75)
            return self.get_summary_prompt(question, df[:reduce])

    def get_context_window(self, model_name):
        context_windows = {
            'gpt-3.5-turbo': 4096,
            'gpt-3.5-turbo-16k': 16384,
            'gpt-4': 8192,
            'gpt-4-32k': 32768,
            'gpt-4-turbo': 128000,
            'gpt-4o': 128000,
            'gpt-4o-mini': 128000,
            'gpt-4.1': 1000000,
            'gpt-4.1-mini': 1000000,
        }
        return context_windows.get(model_name)
    

    # Revision-Modul
    def check_sql_for_release(self, predicted_sql, db_path):
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes
        cursor = conn.cursor()
        try:
            cursor.execute(predicted_sql)
        except Exception as e:
            return False, predicted_sql, str(e)
    
        return True, predicted_sql, None
    
    #For Correction-Prompt
    def get_all_ddl(self):
        trainingData = self.get_training_data()
        ddlData = trainingData[trainingData.training_data_type=='ddl']
        return ddlData['content'].tolist()