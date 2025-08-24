from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.utilities import SerpAPIWrapper
from langchain_community.utilities import SearchApiAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["SEARCHAPI_API_KEY"] = os.getenv("SEARCHAPI_API_KEY")
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    verbose=True,
    temperature=0.2,
    max_tokens=8192,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


@tool
def search_web_tool(query:str):
    """This tool helps in getting any information from the internet and accepts any string as input"""

    #search_tool=SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
    search_tool=SearchApiAPIWrapper(searchapi_api_key=os.getenv("SEARCHAPI_API_KEY"))
    return search_tool.run(query)
