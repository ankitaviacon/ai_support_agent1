import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langfuse import Langfuse, observe
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent


load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


TABLES = ["trial_masters", "delivery_schedules", "delivery_partner", "store"]
engine = create_engine(
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER', 'root')}:{os.getenv('MYSQL_PASSWORD', '')}"
    f"@{os.getenv('MYSQL_HOST', 'localhost')}:{os.getenv('MYSQL_PORT', '3306')}/{os.getenv('MYSQL_DB', 'test')}"
)
db = SQLDatabase(engine, include_tables=TABLES)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=False,
    agent_type="openai-tools",
)


def classify_intent(user_message: str) -> str:
    """Classify user intent into Greeting, DatabaseQuery, or APIQuery (future-proof)."""
    classification_prompt = f"""
    You are an intent classifier for a chatbot.

    Classify the user's message into exactly one of these categories:
    - Greeting (hello, good morning, how are you, chit-chat)
    - DatabaseQuery (anything asking for structured data from SQL DB)
    - APIQuery (requests that would need an external API, e.g. weather, booking) 

    User message: "{user_message}"

    Respond ONLY with one of: Greeting, DatabaseQuery, APIQuery
    """
    resp = llm.invoke([{"role": "system", "content": classification_prompt}])
    return resp.content.strip()


@observe(name="sql_agent_query")
def answer_user_query(user_message: str) -> str:
    try:
        response = agent_executor.invoke({"input": user_message})
        sql_query = response.get("intermediate_steps", "")
        if sql_query:
            sql_text = str(sql_query).lower()
            if not any(tbl.lower() in sql_text for tbl in TABLES):
                return " Query attempted to access unauthorized tables. Request blocked."

        return response["output"]

    except Exception as e:
        return f"Sorry, I could not process your request. Error: {str(e)}"



def front_agent(user_message: str) -> str:
    intent = classify_intent(user_message)

    if intent == "Greeting":
        reply_prompt = f"""
        Respond naturally, politely, and conversationally to the user message below.
        User: "{user_message}"
        """
        reply = llm.invoke([{"role": "system", "content": reply_prompt}])
        return reply.content.strip()

    elif intent == "DatabaseQuery":
        return answer_user_query(user_message)

    elif intent == "APIQuery":
        return "This request looks like it needs an external API. (Future feature 🚀)"

    else:
        return "Sorry, I couldn’t understand your request."



if __name__ == "__main__":
    print("AI Support Agent (LangChain Router + SQL)\n")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you! Have a great day ahead.")
            break

        answer = front_agent(user_input)
        print(f"Assistant: {answer}\n")

        langfuse.flush()
