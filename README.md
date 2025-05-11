# Interview Assistant

A lightweight workflow that uses LangGraph, Google’s Gemini LLM, and the Tavily search tool to generate tailored interview advice and practice questions based on user queries. The generated advice is saved to a local file for easy review.

---

## Features

- **Contextual Search**  
  Uses TavilySearchResults to fetch up to 5 relevant search results for any interview topic.

- **LLM-Powered Report Generation**  
  Combines search results into a detailed report via Gemini-1.5-Flash-8B, including:
  - Expert advice  
  - 30+ interview questions and model answers

- **Stateful Workflow**  
  Built with LangGraph’s `StateGraph` and `MessagesState` to track:
  1. User query  
  2. Search results  
  3. Generated advice

- **Automatic Saving**  
  Saves the final advice report as `advice.txt` in the project root.


