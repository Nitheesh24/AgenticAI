from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os 
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#web search agent 
web_search_agent = Agent(
    name = "web search agent",
    role = "Search the web for the information",
    model = Groq(id = "llama3-groq-8b-8192-tool-use-preview"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True,

)
#Financial agent
Finance_agent = Agent(
    name = "Finance AI agent",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions = ["Use tables to display the data"],
    show_tool_calls = True,
    markdown = True,

)
multi_ai_agent = Agent(
    team = [web_search_agent,Finance_agent],
    model = Groq(id = "llama-3.3-70b-versatile"),
    instructions = ["Always include sources","Use tables to display the data"],
    show_tool_calls = True,
    markdown = True,

)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA",stream=True)