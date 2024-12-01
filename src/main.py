import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm_model import get_llm
from config.combined import process_query

if __name__ == "__main__":
    # Get user input
    user_input = input('Enter your query: ')

    # Process the query
    final_output = process_query(user_input)

    top_3_texts = [item[0] for item in sorted(final_output, key=lambda x: x[1], reverse=True)[:3]]

    language_model = get_llm()
    
    prompt_template = """
    You are an intelligent assistant who has access to relevant context for better answering questions. 
    Below is some context information that may help you understand the query better:

    Context:
    {context}

    Now, here is the user's query:

    Query:
    {query}

    Please provide a short and simple response based on the context above. Response from the company point of view
    """
    
    context = "\n".join(top_3_texts)

    # Define the user query
    query = user_input
    
    prompt = prompt_template.format(context=context, query=query)

    response = language_model(prompt)
    
    print(response.content)
    

    
    
