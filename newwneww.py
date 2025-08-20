import os
import re
import pandas as pd
from pandas_gbq import read_gbq
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
import yaml
import sqlparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st

# --- Streamlit Page Configuration (Always at the very top) ---
st.set_page_config(page_title="Dilytics Procurement Insights Chatbot", layout="wide")

# --- ABSOLUTE TOP: ALL SESSION STATE INITIALIZATIONS ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_db_df" not in st.session_state:
    st.session_state.last_db_df = None
if "last_db_query" not in st.session_state:
    st.session_state.last_db_query = None
if "last_db_plot_question" not in st.session_state:
    st.session_state.last_db_plot_question = None

if "plot_x_col" not in st.session_state:
    st.session_state.plot_x_col = None
if "plot_y_col" not in st.session_state:
    st.session_state.plot_y_col = None
if "plot_chart_type" not in st.session_state:
    st.session_state.plot_chart_type = None


# --- 🔐 Global Configurations ---
# IMPORTANT: Ensure this path is correct for your system.
SERVICE_ACCOUNT_KEY_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", r"C:\Users\UtkarshSrivastava\OneDrive - DILYTICS TECHNOLOGIES PVT LTD\Documents\Google AI ChatBot\vertex-ai-462816-c5f33c6dc69a.json") # <<<--- YAHAN APNA PATH CHECK KAREIN AUR UPDATE KAREIN
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "vertex-ai-462816")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1") # Ensure this matches your Vertex AI model location
DATASET_ID = os.environ.get("BQ_DATASET_ID", "PROCUREMENT_DATA") # For BigQuery

# --- Document Chatbot Specific Paths ---
DOCUMENT_PATHS = [
    r"C:\Users\UtkarshSrivastava\OneDrive - DILYTICS TECHNOLOGIES PVT LTD\Documents\Google AI ChatBot\DiLytics Procurement Insight Solution Overview v1.0 1.pdf", # <<<--- YAHAN APNA PATH CHECK KAREIN AUR UPDATE KAREIN
    r"C:\Users\UtkarshSrivastava\OneDrive - DILYTICS TECHNOLOGIES PVT LTD\Documents\Google AI ChatBot\Dilytics Procuremnt Insights Mertics and Data Logic Draft A.pdf" # <<<--- YAHAN APNA PATH CHECK KAREIN AUR UPDATE KAREIN
]
PERSIST_DIRECTORY = "./chroma_db"
schema_file_path = r"C:\Users\UtkarshSrivastava\OneDrive - DILYTICS TECHNOLOGIES PVT LTD\Documents\Google AI ChatBot\Full_Procurement_Schema.yaml" # <<<--- YAHAN APNA PATH CHECK KAREIN AUR UPDATE KAREIN

# Set environment variable for authentication for Google Cloud Libraries
if os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH
else:
    st.error(f"❌ Service account key file not found at: {SERVICE_ACCOUNT_KEY_PATH}. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable or correct the path.")
    st.stop() # Stop execution if credentials are not found


# --- Cached Resources for performance ---
@st.cache_resource(show_spinner="⏳ Initializing AI Models and Databases...")
def initialize_resources():
    """Initializes LLM, Embeddings, Document Vector Store, and Schema."""
    try:
        llm = ChatVertexAI(
            model="gemini-2.0-flash-001",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.1
        )
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=PROJECT_ID,
            location=LOCATION
        )

        vectorstore = None
        if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            st.info("📦 Loading existing document knowledge base...")
            vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        else:
            st.info("Creating new document knowledge base from scratch...")
            all_documents = []
            for doc_path in DOCUMENT_PATHS:
                try:
                    if doc_path.lower().endswith(".pdf"):
                        loader = PyPDFLoader(doc_path)
                    elif doc_path.lower().endswith(".docx"):
                        loader = Docx2txtLoader(doc_path)
                    elif doc_path.lower().endswith(".txt"):
                        loader = TextLoader(doc_path)
                    else:
                        st.warning(f"⏭️ Skipping unsupported file: {os.path.basename(doc_path)}")
                        continue
                    pages = loader.load()
                    all_documents.extend(pages)
                except Exception as e:
                    st.error(f"❌ Error loading {os.path.basename(doc_path)}: {e}")

            if not all_documents:
                st.warning("⚠️ No documents loaded for RAG. Document chatbot might not work.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(all_documents)
                try:
                    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIRECTORY)
                    vectorstore.persist()
                    st.success(f"📁 Document knowledge base saved to: {os.path.abspath(PERSIST_DIRECTORY)}")
                except Exception as e:
                    st.error(f"❌ Vector store creation failed: {e}")

        SCHEMA_GUIDE = ""
        try:
            with open(schema_file_path, 'r', encoding='utf-8') as f:
                schema_yaml = yaml.safe_load(f)
            for table in schema_yaml.get("tables", []):
                table_name = table.get("name")
                dimensions = table.get("dimensions", [])
                facts = table.get("facts", [])
                time_dimensions = table.get("time_dimensions", [])
                all_fields = dimensions + facts + time_dimensions
                column_names = [field.get("name") for field in all_fields if field.get("name")]
                synonyms = []
                for field in all_fields:
                    synonyms.extend(field.get("synonyms", []))
                all_keywords = sorted(set(column_names + synonyms))
                if table_name and all_keywords:
                    SCHEMA_GUIDE += f"\n- {table_name} (Columns: {', '.join(all_keywords)})"
        except FileNotFoundError:
            st.error(f"❌ YAML schema file not found at: {schema_file_path}. Please check path.")
            SCHEMA_GUIDE = "No schema loaded. Please check the 'schema_file_path'."
        except Exception as e:
            st.error(f"❌ Error loading YAML schema: {e}. Check YAML format.")
            SCHEMA_GUIDE = "Error loading schema. Check YAML format."

        st.success("✅ All resources loaded successfully!")
        return llm, embeddings, vectorstore, SCHEMA_GUIDE

    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()
        return None, None, None, None

llm, embeddings, vectorstore, SCHEMA_GUIDE = initialize_resources()

alias_map = {
    "DIL_SUPPLIERS_D": "sup",
    "DIL_PURCH_SCHEDULE_LINE_F": "psl",
    "DIL_PURCH_COST_F": "pc",
    "DIL_PURCHASE_ORDER_D": "po",
    "DIL_SUPPLIER_SITE_D": "site",
    "DIL_ITEMS_D": "itm",
    "DIL_CURRENCY_D": "cur",
    "DIL_ORG_D": "org"
}

reserved_keywords = {
    "select", "from", "where", "group", "order", "limit",
    "join", "union", "on", "inner", "outer", "as", "and", "or",
    "left", "right", "full", "by", "asc", "desc", "count", "sum", "avg", "min", "max"
}

# --- auto_cast_fix function ---
def auto_cast_fix(sql_query, error_msg):
    if "aggregate function sum" in error_msg and "string" in error_msg:
        fixed_query = re.sub(r"SUM\((.*?)\)", r"SUM(CAST(\1 AS FLOAT64))", sql_query, flags=re.IGNORECASE)
        return fixed_query
    if "operator =" in error_msg and "float64, string" in error_msg:
        fixed_query = re.sub(
            r"(\w+)\s*=\s*'(\d+\.?\d*)\b'(\s*AND|\s*OR|\s*GROUP BY|\s*ORDER BY|\s*LIMIT|\s*|$)",
            r"\1 = CAST('\2' AS FLOAT64)\3",
            sql_query,
            flags=re.IGNORECASE
        )
        fixed_query = re.sub(
            r"'(\d+\.?\d*)\b'\s*=\s*(\w+)(\s*AND|\s*OR|\s*GROUP BY|\s*ORDER BY|\s*LIMIT|\s*|$)",
            r"CAST('\1' AS FLOAT64) = \2\3",
            fixed_query,
            flags=re.IGNORECASE
        )
        return fixed_query
    if "operator =" in error_msg and "int64, string" in error_msg:
        fixed_query = re.sub(
            r"(\w+)\s*=\s*'(\d+)\b'(\s*AND|\s*OR|\s*GROUP BY|\s*ORDER BY|\s*LIMIT|\s*|$)",
            r"\1 = CAST('\2' AS INT64)\3",
            sql_query,
            flags=re.IGNORECASE
        )
        fixed_query = re.sub(
            r"'(\d+)\b'\s*=\s*(\w+)(\s*AND|\s*OR|\s*GROUP BY|\s*ORDER BY|\s*LIMIT|\s*|$)",
            r"CAST('\1' AS INT64) = \2\3",
            fixed_query,
            flags=re.IGNORECASE
        )
        return fixed_query
    return sql_query

# --- Helper to save plots to bytes ---
def _save_plot_to_bytes(fig: plt.Figure) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1, dpi=300) # Increased DPI for better quality
    buf.seek(0)
    plt.close(fig)
    return buf

# --- generate_plot_from_df (Revised for readability) ---
def generate_plot_from_df(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str, title: str = "Visualization") -> BytesIO | None:
    # Configurable limits for plot categories
    MAX_PIE_SLICES = 10  # Max slices before grouping into 'Others' for pie charts
    MAX_BAR_CATEGORIES = 15 # Max categories to display on bar chart before considering truncation (less aggressive than pie)

    if df.empty or x_col not in df.columns:
        return None
    if y_col and y_col not in df.columns: # Y_col is optional for some chart types
        y_col = None # Ensure y_col is None if it doesn't exist or is not needed

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_generated = False

    try:
        # Convert date-like columns for better plotting if needed
        cols_to_check = [c for c in [x_col, y_col] if c and c in df.columns]
        for col in cols_to_check:
            if any(k in col.lower() for k in ['date', 'time', 'month', 'year', 'dt']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass # Ignore if conversion fails

        if chart_type == 'bar':
            if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                df_grouped = df.groupby(x_col)[y_col].sum().reset_index()
                df_sorted = df_grouped.sort_values(by=y_col, ascending=False)

                # Handle too many categories for bar chart
                if len(df_sorted) > MAX_BAR_CATEGORIES:
                    df_display = df_sorted.head(MAX_BAR_CATEGORIES)
                    other_sum = df_sorted.iloc[MAX_BAR_CATEGORIES:][y_col].sum()
                    if other_sum > 0:
                        df_display = pd.concat([df_display, pd.DataFrame({x_col: ['Others'], y_col: [other_sum]})])
                else:
                    df_display = df_sorted

                sns.barplot(x=df_display[x_col], y=df_display[y_col], ax=ax)
                ax.set_ylabel(y_col.replace('_', ' ').title())
                plot_generated = True
            elif pd.api.types.is_numeric_dtype(df[x_col]) and not y_col:
                 # If only one numeric column, assume frequency distribution (like histogram but with bars for categories)
                df_counts = df[x_col].value_counts().reset_index()
                df_counts.columns = [x_col, 'Count']
                df_counts_sorted = df_counts.sort_values(by='Count', ascending=False)

                if len(df_counts_sorted) > MAX_BAR_CATEGORIES:
                    df_display = df_counts_sorted.head(MAX_BAR_CATEGORIES)
                    other_sum = df_counts_sorted.iloc[MAX_BAR_CATEGORIES:]['Count'].sum()
                    if other_sum > 0:
                        df_display = pd.concat([df_display, pd.DataFrame({x_col: ['Others'], 'Count': [other_sum]})])
                else:
                    df_display = df_counts_sorted

                sns.barplot(x=df_display[x_col], y=df_display['Count'], ax=ax)
                ax.set_ylabel('Count')
                plot_generated = True

        elif chart_type == 'line':
            if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                df_sorted = df.sort_values(by=x_col)
                sns.lineplot(x=df_sorted[x_col], y=df_sorted[y_col], ax=ax)
                ax.set_ylabel(y_col.replace('_', ' ').title())
                plot_generated = True
        elif chart_type == 'scatter':
            if y_col and pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                ax.set_ylabel(y_col.replace('_', ' ').title())
                plot_generated = True
        elif chart_type == 'pie' and y_col: # Pie chart uses x_col for labels, y_col for values
            if pd.api.types.is_numeric_dtype(df[y_col]) and (df[y_col] >= 0).all():
                pie_data = df.groupby(x_col)[y_col].sum().reset_index()
                pie_data = pie_data.sort_values(by=y_col, ascending=False) # Sort for consistent "Others" grouping

                # Implement "Top N + Others" for pie charts
                if len(pie_data) > MAX_PIE_SLICES:
                    top_n_data = pie_data.head(MAX_PIE_SLICES - 1) # One slice reserved for 'Others'
                    other_sum = pie_data.iloc[MAX_PIE_SLICES - 1:][y_col].sum()
                    
                    if other_sum > 0:
                        labels = top_n_data[x_col].tolist() + ['Others']
                        sizes = top_n_data[y_col].tolist() + [other_sum]
                    else: # If sum of others is 0, just use top_n_data
                        labels = top_n_data[x_col].tolist()
                        sizes = top_n_data[y_col].tolist()
                else:
                    labels = pie_data[x_col]
                    sizes = pie_data[y_col]
                
                # Check if all sizes are zero to prevent error
                if sum(sizes) == 0:
                    plt.close(fig)
                    return None # Cannot plot a pie chart with all zero values

                # Autopct format for labels on slices
                def autopct_format(pct):
                    return ('%1.1f%%' % pct) if pct > 0 else '' # Only show percentage if greater than 0

                # Wedgeprops to add some spacing between slices
                ax.pie(sizes, labels=labels, autopct=autopct_format, startangle=90, textprops={'fontsize': 9}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
                ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                plot_generated = True
        elif chart_type == 'histogram': # Histograms only need one column (x_col)
            if pd.api.types.is_numeric_dtype(df[x_col]):
                sns.histplot(df[x_col], kde=True, ax=ax)
                ax.set_ylabel('Frequency')
                plot_generated = True
        elif chart_type == 'countplot': # Count plots only need one categorical column (x_col)
            if pd.api.types.is_object_dtype(df[x_col]) or pd.api.types.is_categorical_dtype(df[x_col]):
                df_counts = df[x_col].value_counts().reset_index()
                df_counts.columns = [x_col, 'Count']
                df_counts_sorted = df_counts.sort_values(by='Count', ascending=False)

                if len(df_counts_sorted) > MAX_BAR_CATEGORIES:
                    df_display = df_counts_sorted.head(MAX_BAR_CATEGORIES)
                    other_sum = df_counts_sorted.iloc[MAX_BAR_CATEGORIES:]['Count'].sum()
                    if other_sum > 0:
                        df_display = pd.concat([df_display, pd.DataFrame({x_col: ['Others'], 'Count': [other_sum]})])
                else:
                    df_display = df_counts_sorted

                sns.barplot(y=df_display[x_col], x=df_display['Count'], order=df_display[x_col], ax=ax)
                ax.set_ylabel(x_col.replace('_', ' ').title())
                ax.set_xlabel('Count')
                plot_generated = True

        if plot_generated:
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
            # Rotate X-axis labels for better readability, but only if they are not dates or numbers already handled by seaborn
            if chart_type in ['bar', 'countplot'] and (pd.api.types.is_object_dtype(df[x_col]) or pd.api.types.is_categorical_dtype(df[x_col])):
                plt.xticks(rotation=45, ha='right', fontsize=10)
            else:
                 plt.xticks(fontsize=10) # Keep default rotation for numeric/datetime if not bar/countplot
            plt.yticks(fontsize=10) # Set fontsize for y-ticks too
            plt.tight_layout()
            return _save_plot_to_bytes(fig)
        else:
            plt.close(fig)
            return None

    except Exception as e:
        plt.close(fig)
        st.error(f"Error generating plot: {e}") # Log error to Streamlit
        return None

# --- get_document_answer function ---
def get_document_answer(question: str, vectorstore: Chroma, llm: ChatVertexAI) -> str:
    if not vectorstore:
        return "Sorry, the document knowledge base is not set up correctly."

    qa_prompt_template = """
    You are a helpful assistant. Use only the following context to answer the question.
    and you have two documents so 2nd one is related to matrics Procuremnt Insights Mertics so if user is asking question related matrics then
    follow 2nd document
    otherwise follow 1st document which is Procurement Insight Solution Overview in this there are so many table which have columns like
    S. No., Report Name, Description, Key Question Answered so if user question is related or matched with keyquestion answered then in answer
    you should show the description of that row
    If the answer isn't found in the context, say:
    "Sorry, I don't have enough information in the documents to answer that."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_prompt_template)

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        document_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        result = document_qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        return f"Error from document chatbot: {e}"

# --- get_database_answer function (Modified to return data, plot_bytes, and query string) ---
def get_database_answer(question: str, llm: ChatVertexAI, schema_guide: str) -> tuple[str, pd.DataFrame | None, BytesIO | None, str | None]:
    sql_query = ""
    df = None
    plot_bytes = None
    database_response_text = ""
    generated_sql_display = None

    sql_prompt = PromptTemplate(
        input_variables=["question"],
        template=f"""
You are an expert BigQuery SQL generator specifically for procurement data.
Your goal is to translate user questions into valid and efficient BigQuery SQL queries.

Here are the available tables and their relevant columns from the procurement domain:
{schema_guide}

⚠️ Strict Rules for SQL Generation:
- **Always use fully qualified table names in the `FROM` and `JOIN` clauses, e.g., `project.dataset.table`.** The system will handle aliasing automatically.
- **Do NOT assign aliases yourself (e.g., `FROM table AS alias`).** The system will handle aliases.
- **Do NOT use markdown code blocks (e.g., ```sql).** Return ONLY the raw SQL query.
- Use `column_name` directly in `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`. The system will prepend aliases.
- If a column is used in `ORDER BY` or `GROUP BY`, it MUST also be in the `SELECT` clause (unless it's part of an aggregate function like `COUNT(*)`, `SUM(column)`).
- If the requested data is not available or cannot be answered with the provided schema, respond EXACTLY with the string: "Sorry, this data is not available".
- Ensure your query is syntactically correct and logical. Pay close attention to aggregations and grouping.
- When calculating differences (e.g., 'month over month change'), try to use `LAG()` or self-joins if appropriate, but keep it simple if possible.
- if question includes "Top 5" then create bar chart.
- When generating SQL queries with text comparisons in the WHERE clause or inside a CASE statement (e.g., status = 'Approved'),
please always make the string comparison case-insensitive using UPPER(column) = 'value'.
- Generate a BigQuery SQL that compares text values in a case-insensitive way when using the WHERE clause.
- dil_approval_status_d.DESCRIPTION column has data like - "Document has been Approved" and "Document has been Cancelled" dont use upper case for it
User Question: {{question}}
"""
    )

    retry_prompt = PromptTemplate(
        input_variables=["sql", "error_message"],
        template="""
You are an expert in BigQuery SQL debugging and correction.

The following SQL query failed with the error message below.
Your task is to fix the query based on the error.

Instructions for fixing:
- Focus ONLY on resolving the specific error mentioned.
- Do NOT change the query logic or structure unnecessarily.
- Preserve all original `SELECT` columns, `WHERE` conditions, `JOIN`s, `GROUP BY`s, and `ORDER BY` clauses.
- **Crucially: Ensure any column used in `ORDER BY` or `GROUP BY` is also present in the `SELECT` clause (unless it's an aggregate function).**
- If a column is a `STRING` type and needs to be used in an aggregate function (like `SUM`, `AVG`, `COUNT` on numbers) or a numerical comparison, apply **explicit type casting using `CAST(column AS FLOAT64)` or `CAST(column AS INT64)`.**
- If a numerical column is being compared to a string literal (e.g., `WHERE amount = '100'`), `CAST` the string literal to the correct numeric type (e.g., `CAST('100' AS FLOAT64)`).
- In the table `DIL_PURCH_COST_F`, the column `AMOUNT_ORDERED` is of type STRING.
- I am using it in a SUM() aggregation, so this throws a type mismatch error in BigQuery: "No matching signature for aggregate function SUM for argument type: STRING".
- Return ONLY the corrected SQL query. Do not include any explanations or markdown formatting.
- ensure at the time of fixing project id and dataset id is written correct
- If the data type of any numeric-looking column (like amounts, prices, or quantities) is STRING,
and it's used in SUM(), AVG(), or other numeric functions, then cast it using CAST(column AS FLOAT64).
- Always avoid division by zero in SQL.
- In the DIL_APPROVAL_STATUS_D table, the description column should only be considered to contain one of the following six specific values:

'Document has been Approved'

'Document has been Cancelled'

'Document is Approved but not yet Accepted'

'Document is still undergoing Approval'

'Document is not yet Complete'

'Document has been Rejected'

Do not assume any other possible values for this column while generating the SQL query.
- in the DIL_PURCH_REQ_LINES_F fact table REQUISITION_NUMBER as purch_rqstn_num.
Whenever dividing two columns (e.g., col1 / col2), first check if the denominator is zero.
Use a CASE WHEN condition to safely handle the division:

❌ Do not write: col1 / col2

✅ Instead, write:

sql
Copy
Edit
CASE WHEN col2 != 0 THEN col1 / col2 ELSE NULL END
This logic must be used inside AVG(), SUM(), or any other aggregation if the expression is a ratio or percentage.
Never perform direct division without this condition, or BigQuery will throw an error.
- While generating SQL queries, always use table aliases consistently throughout the query. Do not refer to full table names after assigning aliases. For example, if the DIL_SUPPLIERS_D table is aliased as sup, then use sup.VENDOR_NAME instead of DIL_SUPPLIERS_D.VENDOR_NAME in SELECT, JOIN, GROUP BY, and ORDER BY clauses.
- ❌ [DB Chatbot] All attempts to fix the query failed: Reason: 400 POST [https://bigquery.googleapis.com/bigquery/v2/projects/vertex-ai-462816/queries?prettyPrint=false](https://bigquery.googleapis.com/bigquery/v2/projects/vertex-ai-462816/queries?prettyPrint=false): No matching signature for aggregate function SUM
  Argument types: STRING
  If you got such type of error then change string to float64 with help of cast function and try again to execute

Ensure all queries run without type mismatch errors.
Broken SQL:
{sql}

Error Message:
{error_message}
"""
    )

    try:
        prompt_text = sql_prompt.format(question=question)
        sql_query = llm.invoke(prompt_text).content.strip()

        # Clean up unwanted markdown code blocks
        if "```" in sql_query:
            sql_query = sql_query.split("```")[1].replace("sql", "").strip()

        if "sorry" in sql_query.lower() or "not available" in sql_query.lower():
            database_response_text = "Sorry, I don't have enough data in the database to answer that specifically."
            return database_response_text, df, plot_bytes, None

        # --- SQL Aliasing and Formatting ---
        # 1. Remove any aliases LLM might have added initially (e.g., table AS alias)
        sql_query = re.sub(
            r'\b((?:\w+\.)?\w+)\s+AS\s+(\w+)(?=\s|,|$)',
            lambda m: m.group(1), # Keep only the table name
            sql_query,
            flags=re.IGNORECASE
        )

        final_table_aliases = {}
        # Pattern to find FROM/JOIN clauses and extract table name and potential alias
        table_alias_extraction_pattern = re.compile(
            r'(FROM|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN)\s+(`?[\w.]+`?)\s*(?:AS\s+)?(\w+)?(?=\s|$)',
            re.IGNORECASE
        )

        # First pass: Identify base table names and assign aliases
        for match in table_alias_extraction_pattern.finditer(sql_query):
            raw_table_name = match.group(2).strip('`')
            base_table_name = raw_table_name.split('.')[-1]
            default_alias = alias_map.get(base_table_name, base_table_name.lower())
            if default_alias in reserved_keywords: # Prevent alias being a reserved keyword
                default_alias += "_t"
            final_table_aliases[base_table_name] = default_alias
        
        # Second pass: Replace full table paths with aliases
        def replace_table_with_alias(match):
            raw_table_name = match.group(2).strip('`')
            base_table_name = raw_table_name.split('.')[-1]
            full_table_path = f"{PROJECT_ID}.{DATASET_ID}.{base_table_name}"
            alias = final_table_aliases[base_table_name]
            return f"{match.group(1)} `{full_table_path}` {alias}"

        sql_query = table_alias_extraction_pattern.sub(replace_table_with_alias, sql_query)

        # Third pass: Prepend aliases to column names
        # This pattern looks for column names (word characters) optionally preceded by a dot
        # that are NOT already preceded by an alias or full table path
        for base_table_name, alias in final_table_aliases.items():
            # This regex needs to be more careful to only prepend aliases where necessary
            # It targets column names that are not already aliased (e.g., `column_name` or `table.column_name`)
            # and are present in the table
            column_ref_pattern = re.compile(
                rf"(?<![\w.`'])(?!\b{re.escape(alias)}\.)(?:\b{re.escape(base_table_name)}\.|\b)(?P<column>\w+)\b",
                re.IGNORECASE
            )
            
            # Use schema to check if column exists in table (more robust)
            # For simplicity here, we assume if column name appears without alias, it belongs to this table
            # A more robust solution would cross-reference with schema_yaml
            
            # Simple replacement for now, assumes no ambiguity unless already aliased
            sql_query = re.sub(column_ref_pattern, lambda m: f"{alias}.{m.group('column')}" if not m.group(0).startswith(f"{alias}.") else m.group(0), sql_query)

        # Ensure "FROM project.dataset.table" is correctly aliased, even if not matched by group above
        for base_table, alias in final_table_aliases.items():
            full_path = f"{PROJECT_ID}.{DATASET_ID}.{base_table}"
            # This covers cases where `project.dataset.table` might appear without an AS clause
            sql_query = re.sub(rf'`{re.escape(full_path)}`(?!\s+{re.escape(alias)}\b)', f'`{full_path}` {alias}', sql_query, flags=re.IGNORECASE)


        generated_sql_display = sqlparse.format(sql_query, reindent=True, keyword_case='upper')

        df = read_gbq(sql_query, project_id=PROJECT_ID)

        if df.empty:
            database_response_text += "No relevant data found for your query in the database."
            return database_response_text, df, plot_bytes, generated_sql_display

        # Default plot generation logic (used for chat history)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include='object').columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()

        default_x = None
        default_y = None
        default_chart_type = None

        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            default_x = datetime_cols[0]
            default_y = numeric_cols[0]
            default_chart_type = 'line'
        elif len(object_cols) > 0 and len(numeric_cols) > 0:
            default_x = object_cols[0]
            default_y = numeric_cols[0]
            default_chart_type = 'bar'
        elif len(numeric_cols) > 0:
            default_x = numeric_cols[0]
            default_chart_type = 'histogram'
        elif len(object_cols) > 0:
            default_x = object_cols[0]
            default_chart_type = 'countplot'

        if default_x and default_chart_type:
            plot_bytes = generate_plot_from_df(df.copy(), default_x, default_y, default_chart_type, title=question)

        if len(df.columns) == 1 and len(df) == 1:
            database_response_text = f"The {question.lower().replace('what is the ', '').replace('show me the ', '').replace('tell me the ', '')} is: **{df.iloc[0, 0]}**"
        else:
            database_response_text = f"Here are the key insights from the database:\n"

    except Exception as e:
        error_msg = str(e)
        database_response_text = f"An error occurred fetching data from the database: {error_msg}\n"

        needs_auto_cast_handling = (
            ("no matching signature for operator =" in error_msg.lower() and ("float64, string" in error_msg.lower() or "int64, string" in error_msg.lower()))
            or ("no matching signature for aggregate function sum" in error_msg.lower() and "string" in error_msg.lower())
        )

        # Attempt auto-cast fix first
        if needs_auto_cast_handling:
            try:
                cast_sql = auto_cast_fix(sql_query, error_msg)
                generated_sql_display = sqlparse.format(cast_sql, reindent=True, keyword_case='upper')

                df = read_gbq(cast_sql, project_id=PROJECT_ID)

                if df.empty:
                    database_response_text += "No data found for your query in the database after auto-fix attempt."
                    return database_response_text, df, plot_bytes, generated_sql_display

                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                object_cols = df.select_dtypes(include='object').columns.tolist()
                datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
                default_x, default_y, default_chart_type = None, None, None
                if len(datetime_cols) > 0 and len(numeric_cols) > 0: default_x, default_y, default_chart_type = datetime_cols[0], numeric_cols[0], 'line'
                elif len(object_cols) > 0 and len(numeric_cols) > 0: default_x, default_y, default_chart_type = object_cols[0], numeric_cols[0], 'bar'
                elif len(numeric_cols) > 0: default_x, default_chart_type = numeric_cols[0], 'histogram'
                elif len(object_cols) > 0: default_x, default_chart_type = object_cols[0], 'countplot'

                if default_x and default_chart_type:
                    plot_bytes = generate_plot_from_df(df.copy(), default_x, default_y, default_chart_type, title=question + " (Auto-Fixed)")

                if len(df.columns) == 1 and len(df) == 1:
                    database_response_text = f"The {question.lower().replace('what is the ', '').replace('show me the ', '').replace('tell me the ', '')} is: **{df.iloc[0, 0]}** (auto-fixed)"
                else:
                    database_response_text = f"Here are the results from the database (auto-fixed):\n"
                return database_response_text, df, plot_bytes, generated_sql_display

            except Exception as e3:
                database_response_text += f"Auto-cast retry failed: {e3}\n"
        
        # Then attempt LLM-based retry
        retry_sql_prompt = retry_prompt.format(sql=sql_query, error_message=error_msg)
        try:
            fixed_sql = llm.invoke(retry_sql_prompt).content.strip()
            if "```" in fixed_sql:
                fixed_sql = fixed_sql.split("```")[1].replace("sql", "").strip()

            generated_sql_display = sqlparse.format(fixed_sql, reindent=True, keyword_case='upper')

            df = read_gbq(fixed_sql, project_id=PROJECT_ID)

            if df.empty:
                database_response_text += "No data found for your query in the database after Gemini-fix attempt."
                return database_response_text, df, plot_bytes, generated_sql_display

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            object_cols = df.select_dtypes(include='object').columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
            default_x, default_y, default_chart_type = None, None, None
            if len(datetime_cols) > 0 and len(numeric_cols) > 0: default_x, default_y, default_chart_type = datetime_cols[0], numeric_cols[0], 'line'
            elif len(object_cols) > 0 and len(numeric_cols) > 0: default_x, default_y, default_chart_type = object_cols[0], numeric_cols[0], 'bar'
            elif len(numeric_cols) > 0: default_x, default_chart_type = numeric_cols[0], 'histogram'
            elif len(object_cols) > 0: default_x, default_chart_type = object_cols[0], 'countplot'

            if default_x and default_chart_type:
                plot_bytes = generate_plot_from_df(df.copy(), default_x, default_y, default_chart_type, title=question + " (Gemini-Fixed)")

            if len(df.columns) == 1 and len(df) == 1:
                database_response_text = f"The {question.lower().replace('what is the ', '').replace('show me the ', '').replace('tell me the ', '')} is: **{df.iloc[0, 0]}** (Gemini-fixed)"
            else:
                database_response_text = f"Here are the results from the database (Gemini-fixed):\n"
            return database_response_text, df, plot_bytes, generated_sql_display

        except Exception as e2:
            database_response_text += f"All attempts to fix the query failed: {e2}\n"
            database_response_text += "Please try rephrasing your question or check the schema definition."
            return database_response_text, df, plot_bytes, generated_sql_display
            
    return database_response_text, df, plot_bytes, generated_sql_display


# --- Main Chatbot Orchestration Function ---
def process_user_question(question: str, doc_vectorstore: Chroma, llm_model: ChatVertexAI, schema_guide: str) -> dict:
    response_elements = {
        "text": "",
        "query_display": None,
        "dataframe_display": None, # Kept for internal storage, but not displayed in chat bubbles
        "plot_data_bytes": None,
        "source_info": [] # To store details about where answers came from
    }

    db_response_text, db_dataframe, db_plot_bytes, generated_sql = get_database_answer(question, llm_model, schema_guide)
    doc_answer_text = get_document_answer(question, doc_vectorstore, llm_model)

    final_response_text = ""
    
    # --- Combine Database and Document Answers ---
    if db_dataframe is not None and not db_dataframe.empty:
        final_response_text += f"**Database Insights:**\n{db_response_text}\n\n"
        response_elements["plot_data_bytes"] = db_plot_bytes
        response_elements["query_display"] = generated_sql
        response_elements["source_info"].append("database")
        
        # Store for dynamic visualization section (for the current query)
        st.session_state.last_db_df = db_dataframe
        st.session_state.last_db_query = generated_sql
        st.session_state.last_db_plot_question = question # Store question for plot title
        
        # Reset plot controls to default for new query
        st.session_state.plot_x_col = None
        st.session_state.plot_y_col = None
        st.session_state.plot_chart_type = None
    elif db_response_text and "Sorry, I don't have enough data" not in db_response_text:
        # If DB query ran but returned no data or general error without specific "sorry"
        final_response_text += f"**Database Attempt:**\n{db_response_text}\n\n"
        response_elements["query_display"] = generated_sql
        response_elements["source_info"].append("database_attempt")
        st.session_state.last_db_df = None # Ensure no stale data
        st.session_state.last_db_query = None
        st.session_state.last_db_plot_question = None
    else:
        final_response_text += "**Database Attempt:** No direct data found or an error occurred.\n\n"
        response_elements["query_display"] = generated_sql if generated_sql else "N/A"
        st.session_state.last_db_df = None # Ensure no stale data
        st.session_state.last_db_query = None
        st.session_state.last_db_plot_question = None


    if "Sorry, I don't have enough information" not in doc_answer_text:
        final_response_text += f"**Document-Based Information:**\n{doc_answer_text}\n\n"
        response_elements["source_info"].append("document")
    else:
        final_response_text += f"**Document-Based Information:** {doc_answer_text}\n\n" # Even if no info, state it.

    # Fallback if neither provides a substantial answer
    if not response_elements["source_info"]:
        final_response_text += "I couldn't find a direct answer from either the database or documents. Please try rephrasing your question or ask a question related to procurement insights."
        response_elements["source_info"].append("fallback")


    response_elements["text"] = final_response_text.strip()
    # Store the actual dataframe in session state for dynamic plotting below the chat
    # response_elements["dataframe_display"] is kept for consistency but won't be displayed in chat bubbles
    response_elements["dataframe_display"] = db_dataframe
    
    return response_elements

# --- Authentication Logic ---
def authenticate():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "DILPYTHONPRO" and password == "UTechBot":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")

# --- Streamlit App UI ---
st.title("📊 PROCUREMENT INSIGHTS CHATBOT")
st.markdown("---")


# Main application logic for authenticated users
if not st.session_state.authenticated:
    authenticate()
else:
    st.sidebar.success("Logged in as DILPYTHONPRO")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.messages = [] # Clear chat on logout
        st.session_state.last_db_df = None # Clear current data on logout
        st.session_state.last_db_query = None
        st.session_state.plot_x_col = None
        st.session_state.plot_y_col = None
        st.session_state.plot_chart_type = None
        st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                # Removed: st.dataframe(message["dataframe_display"].head(5))
                
                # Display Plot for history (default generated plot)
                if message["plot_data_bytes"] is not None:
                    st.markdown("Associated Plot:")
                    st.image(message["plot_data_bytes"], use_column_width=True)
                # Display Query/Details in an expander for history
                if message["query_display"] is not None:
                    with st.expander("View Query/Details"):
                        # Heuristic to guess if it's SQL for syntax highlighting
                        lang = "sql" if "SELECT" in message["query_display"].upper() and "FROM" in message["query_display"].upper() else "markdown"
                        st.code(message["query_display"], language=lang)


    # React to user input
    if prompt := st.chat_input("Ask your question about procurement (e.g., 'What is the total value of approved purchase orders last month?' or 'What is the overview of the solution?'):"):
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_elements = process_user_question(prompt, vectorstore, llm, SCHEMA_GUIDE)
                
                # Display the main text response in the current chat bubble
                st.markdown(response_elements["text"])

                # Display Plot in current chat bubble (if available)
                if response_elements["plot_data_bytes"] is not None:
                    st.markdown("Associated Plot:")
                    st.image(response_elements["plot_data_bytes"], use_column_width=True)

                # Display Query in current chat bubble (if available)
                if response_elements["query_display"] is not None:
                    with st.expander("View Query/Details"):
                        lang = "sql" if "SELECT" in response_elements["query_display"].upper() and "FROM" in response_elements["query_display"].upper() else "markdown"
                        st.code(response_elements["query_display"], language=lang)

            # Store the full response elements to chat history for redraw on rerun
            # dataframe_display is stored, but not explicitly rendered in chat bubbles
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_elements["text"],
                "dataframe_display": response_elements["dataframe_display"], # Still store for dynamic section below
                "plot_data_bytes": response_elements["plot_data_bytes"],
                "query_display": response_elements["query_display"]
            })
        # Important: Rerun the script to ensure the "Current Query Results" section updates
        # and new selectbox states are picked up correctly for dynamic plotting.
        st.rerun()


    # --- Dedicated Section for Current Query Results and Dynamic Visualization ---
    # This section appears below the chat messages and updates dynamically with the last query's data
    if st.session_state.last_db_df is not None and not st.session_state.last_db_df.empty:
        st.markdown("---")
        st.markdown("### 📊 Current Query Results & Dynamic Visualization")

        # Display the full DataFrame
        st.markdown("#### Raw Query Results:")
        st.dataframe(st.session_state.last_db_df, use_container_width=True)

        # Show SQL query in expander
        if st.session_state.last_db_query:
            with st.expander("View Executed SQL Query"):
                st.code(st.session_state.last_db_query, language="sql")
            
        st.markdown("#### Dynamic Visualization Options:")

        df_for_plot = st.session_state.last_db_df
        available_columns = [""] + df_for_plot.columns.tolist() # Add empty option for selectbox
        chart_types = ["", "bar", "line", "pie", "scatter", "histogram", "countplot"]

        # Layout select boxes in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            default_x_index = 0
            if st.session_state.plot_x_col and st.session_state.plot_x_col in available_columns:
                default_x_index = available_columns.index(st.session_state.plot_x_col)
            st.session_state.plot_x_col = st.selectbox(
                "Select X-axis", 
                options=available_columns,
                index=default_x_index,
                key="x_axis_select"
            )
        with col2:
            default_y_index = 0
            if st.session_state.plot_y_col and st.session_state.plot_y_col in available_columns:
                default_y_index = available_columns.index(st.session_state.plot_y_col)
            st.session_state.plot_y_col = st.selectbox(
                "Select Y-axis", 
                options=available_columns,
                index=default_y_index,
                key="y_axis_select"
            )
        with col3:
            default_chart_index = 0
            if st.session_state.plot_chart_type and st.session_state.plot_chart_type in chart_types:
                default_chart_index = chart_types.index(st.session_state.plot_chart_type)
            st.session_state.plot_chart_type = st.selectbox(
                "Select Chart Type", 
                options=chart_types,
                index=default_chart_index,
                key="chart_type_select"
            )

        # Generate and display the dynamic plot based on user selections
        if st.session_state.plot_chart_type and st.session_state.plot_x_col:
            plot_title = st.session_state.last_db_plot_question if st.session_state.last_db_plot_question else "Dynamic Plot"
            current_plot_bytes = generate_plot_from_df(
                df_for_plot, 
                st.session_state.plot_x_col, 
                st.session_state.plot_y_col, # Pass Y-axis even if not used by all chart types
                st.session_state.plot_chart_type,
                title=plot_title
            )
            if current_plot_bytes:
                st.image(current_plot_bytes, use_column_width=True)
            else:
                st.warning(f"Could not generate {st.session_state.plot_chart_type} chart with selected columns. Please check column types and chart compatibility.")
        else:
            st.info("Select X-axis, Y-axis (if applicable), and Chart Type to generate a custom plot.")

    st.markdown("---")
    st.caption("Powered by Google Vertex AI & Streamlit")