from crewai import Task
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


def marketing_manger_task(
    agent, product_name, product_purpose, budget_amount, timelines
):
    return Task(
        description=f"""
                    Evaluate the {product_name} and define an initial marketing strategy to achieve its purpose of {product_purpose}, considering the allocated budget of {budget_amount} and the campaign timelines: {timelines}.
                    Identify target market segments and develop key messaging that resonates with the audience.

                    Your Task:
                        1) Outline campaign ideas and allocate the budget across various channels based on the given timelines.
                        2) Integrate insights from the Marketing Analyst to finalize the strategy.
                        3) Set clear performance indicators (KPIs) for the campaigns.
                        4) Review overall campaign performance and refine future strategies based on data.
                    """,
        expected_output=f"A comprehensive marketing strategy document for {product_name} that includes a detailed budget allocation plan of {budget_amount}, a clearly defined timeline of {timelines}, a target audience and messaging framework, detailed campaign outlines with KPIs, a finalized data-driven marketing plan, and actionable recommendations for future efforts.",
        agent=agent,
        output_file="Marketing_manager_task.md",
    )


def digital_marketing_task(context,agent, product_name):
    return Task(
        description=f"""
                            1) Combine information which you have got into a well-structured and Develop creative assets and content for digital campaigns promoting {product_name}.
                            2) Plan and execute digital marketing campaigns (SEO, PPC, social media, email) to support the product's purpose.
                            3) Manage digital advertising platforms and monitor real-time performance.
                            4) Adjust campaign tactics based on performance data and analyst feedback.
                            5) Ensure consistent digital branding and messaging across all channels.
                    """,
        expected_output=f"A well structured set of creative assets and campaign content tailored for digital platforms that successfully launches campaigns across targeted channels, provides real-time performance dashboards, and maintains a consistent digital brand presence and messaging.",
        agent=agent,
        context=context,
        output_file="digital_strategy.md",
    )


def research_analyst_task(agent, product_name):
    return Task(
        description=f"""
                    1) Conduct market research and competitive analysis for {product_name}.
                    2) Analyze consumer behavior and trends relevant to the product.
                    3) Collect data from market research and ongoing campaigns.
                    4) Generate actionable insights to refine the marketing strategy.
                    5) Monitor key performance metrics during campaign execution.
                    6) Prepare regular performance and analytics reports.
                    """,
        expected_output=f"A detailed market research report and competitive analysis for {product_name} with data-driven insights on consumer behavior, actionable recommendations for optimizing the marketing strategy, and regular performance reports featuring key metrics and trends.",
        agent=agent,
        output_file="research.md",
    )
