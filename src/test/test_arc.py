from sentence_transformers import SentenceTransformer, util
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.llm_model import get_llm
# from test.queries_answer import questions, answers
from config.combined import process_query
from modules.extraction_topk import query_data

# Load the pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [
    "How can I review my bills?",
    "Are you guys offering insurance in case of damage",
    "Do you guys offer preowned phone?",
    "How can I check my device is approved device?",
    "Do you guys have pre-owned phone for sale",
    "I got problem with cards"
    
]

answers = [
    "You can send us the email and we can review the bills for you",
    "we offer insurance incase of damage to your phone",
    "Yes we do offer preowned phone you can visit our website http:/preowned.com",
    "To check whether your phone is approved we recommend you to visiting retail for activation",
    "We do have pre-owned phones and other items for sales",
    "The cards are non refundable. Do you mind sending me your email so I can check"
]

print(len(questions),len(answers))

language_model = get_llm()


def calculate_semantic_similarity(reference: str, generated_output: str) -> float:
    """
    Calculate semantic similarity using Sentence-BERT.

    Args:
    reference (str): The reference or expected sentence.
    generated_output (str): The sentence generated by the model/system.

    Returns:
    float: The semantic similarity score.
    """
    # Compute embeddings
    embedding1 = model.encode(reference, convert_to_tensor=True)
    embedding2 = model.encode(generated_output, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = util.cos_sim(embedding1, embedding2)
    
    return similarity.item()

prompt_template = """
    You are an intelligent assistant who has access to relevant context for better answering questions. 
    Below is some context information that may help you understand the query better:

    Context:
    {context}

    Now, here is the user's query:

    Query:
    {query}

    Please provide a short and simple response based on the context above. Response from the company point of view
    and yeah answer in tone of someone really from the company avoid unecessary phreses
    """
    

def blue_score_special_Arch():
        # Step: Calculate BLEU score for the generated response
    cosine_similarities = 0
    for i, q in enumerate(questions):
        # Reference answer for the current question
        reference_answer = answers[i]
        
        generated_context = process_query(q)   
        if isinstance(generated_context, list) and len(generated_context) > 0 and isinstance(generated_context[0], tuple):
            top_3_texts = [item[0] for item in sorted(generated_context, key=lambda x: x[1], reverse=True)[:3]]
        else:
            top_3_texts = []

        print('************************************Top 3 are*****************************','\n')
        print(top_3_texts)
        
        context = "\n".join(top_3_texts)
        query = q
        
        prompt = prompt_template.format(context=context, query=query)

        generated_answer = language_model(prompt)
        
        # print('Answer',generated_answer.content)
        # print('Refrence Answer',reference_answer)
        
        # Calculate BLEU score
        cosine_similarity = calculate_semantic_similarity(reference_answer, generated_answer.content)
        # Print the BLEU score for the current question
        print(f"BLEU score for Question {i+1}: {cosine_similarity:.4f}")
        
        cosine_similarities = cosine_similarities + cosine_similarity
    
    return cosine_similarities/6
        

print('The average cosine similarity is',blue_score_special_Arch())

def blue_score_rag():
    cosine_similarities = 0
    for i,q in enumerate(questions):
        generated_context = query_data(q)
                
        matches = generated_context.get('matches', [])
        if matches:
            # Sort matches by score in descending order and extract the top 3
            top_3_texts = [match['metadata']['text'] for match in sorted(matches, key=lambda x: x['score'], reverse=True)[:3]]
        else:
            top_3_texts = []

        # Print the results
        print('************************************Top 3 are in blue*****************************', '\n')
        print(top_3_texts)
                    
        context = "\n".join(top_3_texts)
        print(context)
        query = q
        reference_answer = answers[i]
        
        prompt = prompt_template.format(context=context, query=query)
 
        generated_answer = language_model(q)
        
        # print('Answer',generated_answer.content)
        # print('Refrence aswer',reference_answer)
        
        # Calculate BLEU score
        cosine_similarity = calculate_semantic_similarity(reference_answer, generated_answer.content)
        # Print the BLEU score for the current question
        print(f"BLEU score for Question {i+1}: {cosine_similarity:.4f}")
        
        cosine_similarities = cosine_similarities + cosine_similarity
    
    return cosine_similarities/6

print('The averge cosine similarity using pure RAG is',blue_score_rag())