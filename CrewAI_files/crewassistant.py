from researchAgents import marketing_manger, digital_marketing_expert, research_analyst
from researchtask import (
    marketing_manger_task,
    digital_marketing_task,
    research_analyst_task,
)
from researchtools import search_web_tool
from crewai import Crew, Process
from crewai import LLM
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm_agent = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    verbose=True,
    temperature=0.2,
    max_tokens=8192,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

def crewexecute(product_name, product_purpose, budget_amount, timelines):
    manager_task = marketing_manger_task(
        marketing_manger,
        product_name,
        product_purpose,
        "Rs. "+str(budget_amount),
        str(timelines)+" days",
    )
    research_expert = research_analyst_task(
        research_analyst, product_name
    )


    digital_expert = digital_marketing_task([manager_task,research_expert],
        digital_marketing_expert, product_name
    )

    crew = Crew(
        agents=[marketing_manger, research_analyst, digital_marketing_expert],
        tasks=[manager_task, research_expert, digital_expert],
        process=Process.sequential,
        full_output=True,
        verbose=True,
    )
    result = crew.kickoff()
    print(result)
    return str("Your files are ready in this location : 'digital_strategy.md'")
