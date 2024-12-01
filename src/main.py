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
    print('*************************Filter Results************************','\n')
    print(dir(filtered_results))


    # Step 4: Call split_text_by_words
    # split_text_results = split_text_by_words(filtered_results)
    # print('*******************Splitted Text******************************')
    # print(split_text_results)

    # # Step 5: Call compute_cosine_similarity
    # final_result = compute_cosine_similarity(user_input, split_text_results)
    # print('****************************Final Result ******************************')
    # print(final_result)

    # return final_result


if __name__ == "__main__":
    # Get user input
    user_input = input('Enter your query: ')

    # Process the query
    final_output = process_query(user_input)

    # Display the final output
    print(f"Final Result: {final_output}")
