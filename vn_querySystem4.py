from vanna.qdrant import Qdrant_VectorStore
from vanna.openai import OpenAI_Chat

from vanna.types import TrainingPlan
from vanna.utils import deterministic_uuid
from qdrant_client import models
import sql_skeleton as sk
import json
from dotenv import load_dotenv
import os
import pandas as pd
from vn_qsBase_all import VN_QsBase

import sqlite3
import openai_cookbook as oc

from collections import defaultdict, deque
import re
import ast

import sqlglot
from sqlglot.errors import ParseError

load_dotenv('.env')

class VN_QuerySystem(Qdrant_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

        self.graph = self.get_join_graph('Chinook')

        # For Example-Select-Module
        config_zero ={
            "client": config.get("client"),
            "fastembed_model": config.get("fastembed_model"),
            "n_results": 11,
            "documentation_collection_name": "zero_documentation",
            "ddl_collection_name": config.get("ddl_collection_name"),
            "sql_collection_name": "zero_sql",
            "api_key": config.get("api_key"),
            "model": config.get("model")
        }

        self.vn_zero = VN_QsBase(config=config_zero)

        config_preSelection ={
            "client": config.get("client"),
            "fastembed_model": config.get("fastembed_model"),
            "n_results": 500,
            "documentation_collection_name": config.get("documentation_collection_name"),
            "ddl_collection_name": config.get("ddl_collection_name"),
            "sql_collection_name": config.get("sql_collection_name")
        }

        self.vn_preSelection = VN_QsBase(config=config_preSelection)
        self.n_results = config.get("n_results", 10)   
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

    ## Join-Graph
    def set_join_graph(self,db_name):
        self.graph = self.get_join_graph(db_name)

    def get_join_graph(self, db_name):
        self.connect_to_sqlite(os.getenv('dbLoc') + '/' + db_name + '/' + db_name + '.sqlite')
        df_sqliteMaster = self.run_sql("SELECT tbl_name, sql FROM sqlite_master WHERE sql is not null and type='table'")
        
        joins = pd.Series(df_sqliteMaster.apply(lambda row: self.extract_joins(self.convert_ddlToSchema(row['sql']),row['tbl_name']), axis=1).dropna()).explode().tolist()
        #print(joins)

        graph = defaultdict(list)
        for a, b in joins:
            graph[a].append(b)
            graph[b].append(a)
        return graph
    
    def extract_joins(self, ddl, tbl_name) -> list:
        foreign_keys = self.extract_foreign_keys(ddl)
        foreign_keys_clean = list(map(lambda x: x.replace('(', ' ').replace(')', ' '), foreign_keys))
        fk_component = pd.Series(map(lambda x: x.split(' '), foreign_keys_clean)).explode().tolist()
        fk_component_clean = list(filter(lambda x: x !='', fk_component))
        #print(fk_component_clean)
        ref_indexes = [i for i, val in enumerate(fk_component_clean) if val.upper() == 'REFERENCES']
        referenced_table = list(map(lambda x: fk_component_clean[x + 1], ref_indexes))
        joins = list(map(lambda x: (tbl_name, x), referenced_table))
        #print(joins)

        if len(joins) > 0:
            return joins
        
    def extract_foreign_keys(self,ddl) -> list:
        column_list = ddl.split(',')
        foreign_keys = list(filter(lambda x: 'FOREIGN KEY' in x.upper(), column_list))
        return foreign_keys
    
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
            initial_prompt = f"You are a {self.dialect} expert. Complete {self.dialect} SQL query only and with no explanation. Instead of '*' always name all the columns. If there are duplicate column names, use aliases by using the table-name as a prefix with an '_' as a seperator. \n"

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
    def get_related_ddl(self, question: str, **kwargs) -> list:
        results = self._client.query_points(
            self.ddl_collection_name,
            query=self.generate_embedding(question),
            #limit=self.n_results,
            with_payload=True,
        ).points

        return [result.payload["ddl"] for result in results]
    """    
    # Schema-Linking-Modul
    def get_related_ddl(self, question: str, **kwargs) -> list:
        preliminary_sql = kwargs.get('preliminary_sql')
        preliminary_tables = self.extract_table_names_from_sql(preliminary_sql)
        ddl_list = self.get_intermediate_ddl(preliminary_tables)

        key_ddl_list = list(set(ddl_list))
        tables = self.extract_table_name(key_ddl_list)
        intermediates = self.get_intermediates(tables)
        intermediate_ddl_list = self.get_intermediate_ddl(intermediates)
        return key_ddl_list + intermediate_ddl_list

    def extract_table_names_from_sql(self, sql: str):
        try:
            expression = sqlglot.parse_one(sql)
            tables = {table.name for table in expression.find_all(sqlglot.exp.Table)}
            return list(tables)
        
        except ParseError as e:

            # Attempt simple fallback extraction (basic FROM/JOIN regex)
            pattern = r'(?:FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][\w.]*)'
            tables = re.findall(pattern, sql, re.IGNORECASE)
            return list(set(tables))
    
    def extract_table_name(self, ddl_list:list) -> list:
        tables = []
        for ddl in ddl_list:
            ddl_clean = ddl.replace('(',' ')
            table = ddl_clean.split(' ')[0]
            tables.append(table)
        return tables
    
    def get_intermediate_ddl(self,intermediates):
        training_data = self.get_training_data()
        ddls = list(training_data[training_data['training_data_type']=='ddl']['content'])
        return [ddl for ddl in ddls if ddl.startswith(tuple(intermediates))]
    
    def bfs_connected(self, start, allowed_nodes, graph):
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor in allowed_nodes and neighbor not in visited:
                    queue.append(neighbor)
        return visited
    
    def get_intermediates(self, tables):
        table_set = set(tables)
        components = []
        seen = set()

        for table in tables:
            if table not in seen:
                component = self.bfs_connected(table, table_set,self.graph)
                components.append(component)
                seen.update(component)

        if len(components) == 1:
            return []
        else:
            required_intermediates = set()
            base_component = components[0]
            remaining_components = components[1:]

            for comp in remaining_components:
                target_found = False
                for target in comp:
                    queue = deque([(target, [])])
                    visited = set()
                    while queue:
                        current, path = queue.popleft()
                        if current in base_component:
                            intermediates = [node for node in path if node not in table_set]
                            required_intermediates.update(intermediates)
                            target_found = True
                            break
                        visited.add(current)
                        for neighbor in self.graph[current]:
                            if neighbor not in visited:
                                queue.append((neighbor, path + [neighbor]))
                    if target_found:
                        break

            return list(required_intermediates)
    
    
    # Example-Select-Module
    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
        db_id: str = None,
        **kwargs
    ) -> str:
        
        if db_id:
            sql_skeleton = sk.extract_skeleton(sk.normalization(sql),sk.get_db_schemas(json.load(open(os.getenv('tabLoc'))))[db_id])
            return self.add_question_sql(question=question, sql=sql, sql_skeleton=sql_skeleton)
        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl,**kwargs)
        else: 
            super().train(question,sql,documentation,plan)

        
    def get_jaccard_similarity(self, set1, set2):
            # intersection of two sets
        intersection = len(set1.intersection(set2))
            # Unions of two sets
        union = len(set1.union(set2))
     
        return intersection / union

    def remove_prefix_before_dot(self, sql):
        return re.sub(r'\b\w+\.', '', sql)
    
    def remove_column_aliases(self, sql):
    # Replace "AS alias" (case-insensitive) with nothing
        return re.sub(r'\s+AS\s+\w+', '', sql, flags=re.IGNORECASE)
    
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        sql = kwargs.get('preliminary_sql')
        
        try:
            sql = self.remove_prefix_before_dot(self.remove_column_aliases(sql))
            print(sql)

            sql_normalized = sk.normalization(sql)
            all_db_infos = json.load(open(os.getenv('tabLoc')))
            db_schemas = sk.get_db_schemas(all_db_infos)
            sql_skeleton = sk.extract_skeleton(sql_normalized,db_schemas[kwargs.get('db_id')])

            sql_skeleton = sql_skeleton.replace('join', '').replace('left', '').replace('right', '').replace('inner', '').replace('outer', '').replace('full', '').replace('cross', '')
            print(sql_skeleton)
        except:
            sql_skeleton = sql

        zpred = set(sql_skeleton.split(' '))

        q_candidates = self.vn_preSelection.get_similar_question_sql(question)
        for q in q_candidates:
            candi = set(q['sql_skeleton'].split(' '))
            q['jaccard_similarity'] = self.get_jaccard_similarity(zpred,candi)


        df = pd.DataFrame.from_dict(q_candidates)
        df['initialOrder'] = df.index
        df = df.sort_values(by=['jaccard_similarity','initialOrder'], ascending=[False,True], axis=0)
        df = df.head(self.n_results)
        df = df[['question','sql']]

        return df.to_dict(orient='records')
    
    def add_question_sql(self, question: str, sql: str, sql_skeleton: str = None, **kwargs) -> str:
        if sql_skeleton:
        
            question_answer = "Question: {0}\n\nSQL: {1}".format(question, sql)
            id = deterministic_uuid(question_answer)

            self._client.upsert(
                self.sql_collection_name,
                points=[
                    models.PointStruct(
                        id=id,
                        vector=self.generate_embedding(question_answer),
                        payload={
                            "question": question,
                            "sql": sql,
                            "sql_skeleton": sql_skeleton
                        },
                    )
                ],
            )

    # Correction-Modul
    def generate_and_correct_sql(self, question: str, **kwargs) -> str:
        preliminary_sql = self.vn_zero.generate_sql(question, **kwargs)
        if not self.is_sql_valid(preliminary_sql):
            return "", "Not a text-to-sql-question"

        sql = self.generate_sql(question, preliminary_sql=preliminary_sql, **kwargs)
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
        self.connect_to_sqlite(db_path)
        try:
            df = self.run_sql(predicted_sql)
        except Exception as e:
            return False, str(e)
    
        if len(df) > 0:
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
        summary_prompt = self.get_summary_prompt(question, df)
        summary = self.submit_prompt(summary_prompt, **kwargs)
        return summary

    def get_summary_prompt(self, question: str, df: pd.DataFrame) -> str:
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