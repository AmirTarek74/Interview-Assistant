import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState

load_dotenv()  
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Interview Assistant"
llm_name = "gemini-1.5-flash-8b"

llm = ChatGoogleGenerativeAI(model=llm_name, api_key=google_api_key, verbose=False)
tavily_tool = TavilySearchResults(max_results=5)


class AgentState(MessagesState):
    user_query: str
    search_results: str
    advice: str

def search_agent(state: AgentState) -> AgentState:
    """
    Executes a search using the user's query, generates a detailed report, and returns an updated AgentState.

    Args:
        state (AgentState): The current state containing the user's query.

    Returns:
        AgentState: The updated state including the search results and advice report.

    This function processes the user's query to retrieve search results via the Tavily tool. It then constructs a detailed report that includes advice and interview questions with answers relevant to the query context. The report is generated using a language model that ensures accuracy and relevance.
    """

    print("\nSearch Agent Working...")
    user_query = state["user_query"]
    if not user_query:
        return state

    # Here we tell the summary+buffer memory about the user input.
    search_results = tavily_tool.run(user_query)

    advice_template = """Combine these search results into a detailed report:    
    {results}
    The report should include Advice for the user. Interview questions should be relevant to the advice in Q & A format.
    The report should be at least 30 questions with their answers. 
    Context: {query}    
    Double Check every answer in the report."""

    search_chain = (
        ChatPromptTemplate.from_template(advice_template)
        | llm
        | StrOutputParser()
    )
    advice = search_chain.invoke({"results": search_results, "query": user_query})

    return AgentState(user_query=user_query, search_results=search_results, advice=advice)

def advice_save(state: AgentState) -> AgentState:  
    """
    Saves the advice report to a file named advice.txt in the current directory.
    
    Args:
        state (AgentState): The current state containing the advice report.
    
    Returns:
        AgentState: The state unchanged.
    """
    print("\nSaving Advice...")
    advice = state["advice"]
    file_name= "advice.txt"
    with open(file_name, "w") as f:
        f.write(advice)
    print(f"Advice saved to {file_name}")
    return state

def bulid_graph():
    """
    Builds a graph with two nodes: search_agent and advice_save.

    The entry point is the search_agent node, which takes a user query and returns
    an AgentState with the search results and advice report.

    The search results are passed to the advice_save node, which saves the report
    to a file named advice.txt in the current directory.

    The graph is then compiled and returned.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("search_agent", search_agent)
    workflow.add_node("advice_save", advice_save)

    workflow.set_entry_point("search_agent")
    workflow.add_edge("search_agent", "advice_save")
    graph = workflow.compile()
    return graph

def main(query: str):
    """
    Runs the graph with the given user query.

    Args:
        query (str): The user query to run in the graph.

    Returns:
        None
    """
    
    app = bulid_graph()
    output= app.invoke({"user_query": query})
    print("\nGenerated Files:")
    print(f"Advice: {len(output['advice'])} characters")


if __name__ == "__main__":
    query = '''
    Hi my name is Amir,
    I have upcoming Interview for LLM postion.
    '''
    main(query)