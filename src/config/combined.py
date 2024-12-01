import sys
import os

# Ensure module paths are properly added
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from modules.cognitive_self import get_result
from modules.extraction_topk import query_data
from modules.filtering_res import filter_top3
from modules.splitter import split_text_by_words, compute_cosine_similarity
from models.llm_model import get_llm
from models.embedding_model import get_embedding_model

embedding_model = get_embedding_model()

print(embedding_model)

def process_query(user_input: str):
    """
    Processes the input query through a series of functions as specified.
    
    Steps:
    1. Call `get_result`. If it returns 'Not sure', proceed to the next steps.
    2. Call `query_data` with the input.
    3. Call `filter_top3` with the input and output of `query_data`.
    4. Call `split_text_by_words` with the filtered results.
    5. Call `compute_cosine_similarity` with the split text results.
    
    Args:
        user_input (str): The input query from the user.

    Returns:
        str: The final result after processing.
    """
    # Step 1: Call get_result
    result = get_result(user_input)
    result = str(result)
    if result.strip() not in ['Not sure.', 'Not sure']:
        print(len(result))
        print('We terminate here......')
        print(result)
        return result

    # Step 2: Call query_data
    query_result = query_data(user_input)
    
    query_result_text = texts = [match['metadata']['text'] for match in query_result['matches']]
    
    print(query_result_text)

    # Step 3: Call filter_top3
    filtered_results = filter_top3(user_input, query_result_text)
    # Access the first TaskOutput object
    task_output = filtered_results.tasks_output[0]

    # Extract the raw attribute (contains the relevant results)
    raw_results = task_output.raw

    # Split the raw results into separate entries
    separate_entries = raw_results.split("\n")

    print(len(separate_entries))




    # Step 4: Call split_text_by_words
    split_arr = []
    for entry in separate_entries:
        split_text_results = split_text_by_words(entry,12,4)
        for text in split_text_results:
            split_arr.append(text)

    # # Step 5: Call compute_cosine_similarity
    final_result = compute_cosine_similarity(split_arr, user_input, embedding_model)
    
    return final_result