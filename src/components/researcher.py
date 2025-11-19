from typing import Type
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from exa_py import Exa
import requests
import streamlit as st
import os

#--------------------------------#
#         EXA Answer Tool        #
#--------------------------------#
class EXAAnswerToolSchema(BaseModel):
    query: str = Field(..., description="The question you want to ask Exa.")

class EXAAnswerTool(BaseTool):
    name: str = "Ask Exa a question"
    description: str = "A tool that asks Exa a question and returns the answer."
    args_schema: Type[BaseModel] = EXAAnswerToolSchema
    answer_url: str = "https://api.exa.ai/answer"

    def _run(self, query: str):
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": st.secrets["EXA_API_KEY"]
        }

        try:
            response = requests.post(
                self.answer_url,
                json={"query": query, "text": True},
                headers=headers,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # Log the HTTP error
            print(f"Response content: {response.content}")  # Log the response content for more details
            raise
        except Exception as err:
            print(f"Other error occurred: {err}")  # Log any other errors
            raise

        response_data = response.json()
        answer = response_data["answer"]
        citations = response_data.get("citations", [])
        output = f"Answer: {answer}\n\n"
        if citations:
            output += "Citations:\n"
            for citation in citations:
                output += f"- {citation['title']} ({citation['url']})\n"

        return output

#--------------------------------#
#         LLM & Research Agent   #
#--------------------------------#
def create_crew(selection):
    """Create a multi-agent crew with the specified LLM configuration.

    Args:
        selection (dict): Contains provider and model information
            - provider (str): The LLM provider ("OpenAI", "GROQ", or "Ollama")
            - model (str): The model identifier or name

    Returns:
        Crew: A configured CrewAI crew with multiple agents

    Note:
        Ollama models have limited function-calling capabilities. When using Ollama,
        the agents will rely more on their base knowledge and may not effectively use
        external tools like web search.
    """
    provider = selection["provider"]
    model = selection["model"]

    if provider == "GROQ":
        llm = LLM(
            api_key=st.secrets["GROQ_API_KEY"],
            model=f"groq/{model}"
        )
    elif provider == "Ollama":
        llm = LLM(
            base_url="http://localhost:11434",
            model=f"ollama/{model}",
        )
    else:
        # Map friendly names to concrete model names for OpenAI
        if model == "GPT-3.5":
            model = "gpt-3.5-turbo"
        elif model == "GPT-4":
            model = "gpt-4"
        elif model == "o1":
            model = "o1"
        elif model == "o1-mini":
            model = "o1-mini"
        elif model == "o1-preview":
            model = "o1-preview"
        # If model is custom but empty, fallback
        if not model:
            model = "o1"
        llm = LLM(
            api_key=st.secrets["OPENAI_API_KEY"],
            model=f"openai/{model}"
        )

    # Researcher Agent
    researcher = Agent(
        role='Research Analyst',
        goal='Conduct thorough research on given topics for the current year 2025',
        backstory='Expert at analyzing and summarizing complex information',
        tools=[EXAAnswerTool()],
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    # Analyst Agent
    analyst = Agent(
        role='Data Analyst',
        goal='Analyze research data and extract key insights and trends',
        backstory='Specialist in data analysis and visualization',
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    # Writer Agent
    writer = Agent(
        role='Content Writer',
        goal='Compile research findings into a comprehensive, well-structured report',
        backstory='Proficient in writing clear, engaging reports',
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    crew = Crew(
        agents=[researcher, analyst, writer],
        verbose=True,
        process=Process.sequential
    )
    return crew

#--------------------------------#
#         Research Task          #
#--------------------------------#
def create_research_tasks(crew, task_description):
    """Create research tasks for the multi-agent crew to execute.

    Args:
        crew (Crew): The multi-agent crew
        task_description (str): The research query or topic to investigate

    Returns:
        list: List of configured CrewAI tasks
    """
    research_task = Task(
        description=f"Research the topic: {task_description}. Gather comprehensive information, data, and insights from reliable sources.",
        expected_output="Raw research data, key findings, and sources on the topic.",
        agent=crew.agents[0],  # Researcher agent
    )

    analysis_task = Task(
        description="Analyze the gathered research data. Extract key insights, trends, and metrics.",
        expected_output="Analyzed data with insights, trends, and statistical information.",
        agent=crew.agents[1],  # Analyst agent
        context=[research_task]
    )

    writing_task = Task(
        description="""Compile the research and analysis into a comprehensive report for the year 2025.
        The report must be detailed yet concise, focusing on the most significant and impactful findings.

        Format the output in clean markdown (without code block markers or backticks) using the following structure:

        # Executive Summary
        - Brief overview of the research topic (2-3 sentences)
        - Key highlights and main conclusions
        - Significance of the findings

        # Key Findings
        - Major discoveries and developments
        - Market trends and industry impacts
        - Statistical data and metrics (when available)
        - Technological advancements
        - Challenges and opportunities

        # Analysis
        - Detailed examination of each key finding
        - Comparative analysis with previous developments
        - Industry expert opinions and insights
        - Market implications and business impact

        # Future Implications
        - Short-term impacts (next 6-12 months)
        - Long-term projections
        - Potential disruptions and innovations
        - Emerging trends to watch

        # Recommendations
        - Strategic suggestions for stakeholders
        - Action items and next steps
        - Risk mitigation strategies
        - Investment or focus areas

        # Citations
        - List all sources with titles and URLs
        - Include publication dates when available
        - Prioritize recent and authoritative sources
        - Format as: "[Title] (URL) - [Publication Date if available]"

        Note: Ensure all information is current and relevant to 2025. Include specific dates,
        numbers, and metrics whenever possible to support findings. All claims should be properly
        cited using the sources discovered during research.
        """,
        expected_output="A comprehensive research report in markdown format.",
        agent=crew.agents[2],  # Writer agent
        context=[research_task, analysis_task],
        output_file="output/research_report.md"
    )

    return [research_task, analysis_task, writing_task]

#--------------------------------#
#         Research Crew          #
#--------------------------------#
def run_research(crew, tasks):
    """Execute the research tasks using the configured multi-agent crew.

    Args:
        crew (Crew): The multi-agent crew
        tasks (list): List of research tasks to execute

    Returns:
        str: The research results in markdown format
    """
    crew.tasks = tasks
    result = crew.kickoff()

    # If result is empty, try to read from the output file
    if not result or str(result).strip() == "":
        try:
            with open("output/research_report.md", "r", encoding="utf-8") as f:
                result = f.read()
        except FileNotFoundError:
            result = "Research completed but no output was generated."

    return result
