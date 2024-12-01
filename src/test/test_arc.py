import nltk
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nltk.translate.bleu_score import sentence_bleu
from models.llm_model import get_llm
# from test.queries_answer import questions,answers
from config.combined import process_query
from modules.extraction_topk import query_data

questions = [
    "In what zip code is this happening?",
    "Is bill payment available via Twitter?",
    "How much tethering data is available on the unlimited Total Plan?",
    "How can I adjust the back light on the S7 Edge?",
    "How can I order now?",
    "Can I trade in my phone if it's paid off?",
    "How much tethering data is available before my speed is reduced?",
    "What phone are you interested in upgrading to?",
    "How do I know when my phone has shipped?",
    "Could changing your plan or financing your phones cause an increase in the bill?",
    "Have you spoken with Tech Support?",
    "What phone do you have and what zip code is this happening in?",
    "Which store are you referring to?",
    "How much high-speed data is available with the unlimited plan?",
    "Can you look at my bill?",
    "What should I do to make sure I have voice and data roaming enabled?",
    "When will pricing information be available for the pre-sale?",
    "What do I get when upgrading to the Samsung S8?",
    "What is your current zip code?",
    "Have the holiday offers been released?",
    "Have you checked if your software is up to date?",
    "What happened to your phone?",
    "Can you email to check what you've been advised for accuracy?",
    "When can I pre-order the iPhone X?",
    "What device are you interested in and are you already a customer?",
    "What is your zip code to check our service in your area?",
    "Can you help with checking my email request?",
    "Is there an issue with the tower in the area?",
    "How can I activate an old device or get a pre-owned device?",
    "Is the phone activated on the new owner's account?",
    "Has your current phone been paid off, or do you need to pay the early upgrade amount?",
    "What caused you to feel this way about our service?",
    "Can I get more details about the device you're interested in upgrading to?",
    "What issues are you experiencing with your service?"
]

answers = [
    "Please tell me the zip code you’re referencing. Also, do you live in this area or are you just visiting? ^AP",
    "Bill payment is not available via Twitter. You can call 611 from your device or visit a store. ^BR",
    "On an unlimited Total Plan, there is up to 10 GB of tethering data available. ^JD",
    "Try the steps below to turn off or adjust the back light on the S7 Edge. ^BR",
    "You can order now with our Telesales Department at 888-289-8722. ^AH",
    "For existing customers, as long as your phone is paid off and it qualifies, you can trade it in. You don't have to port a number in unless you're adding a line. ^AS",
    "While tethering, you can use up to 10GBs before your data speed is reduced. You can get up to 120GBs on our hotspot plans. ^KJ",
    "Which phone were you interested in upgrading to? ^AW",
    "Once your card has been charged, this means your phone has shipped. ^AW",
    "Were you changing your plan or financing your phones? That could cause an increase in the bill. ^AS",
    "Have you spoken with Tech Support at 888-944-9400? If so, what have you been told? ^AW",
    "Which phone do you have? Also, in what zip code is this happening? ^AW",
    "I'm so sorry about that. Which store are you referring to? I can provide feedback to the leadership team. ^LC",
    "With the unlimited plan, you are allotted 22GB of high-speed data per month. Be sure to close out of any unused apps and stream on standard definition to help with speeds and usage. ^LC",
    "Let me look at your bill. Send me an email to __email__ with my initials in the subject line. ^AH",
    "Make sure you have voice and data roaming enabled on your device. Also, make sure to dial *228 before traveling. ^KJ",
    "Pricing information will be available on the pre-sale date which is 10/27/2017. ^AW",
    "Now when upgrading to the Samsung S8 you get a $100 promo card for accessories. ^BR",
    "Let me look at this for you. What is your current zip code? ^SL",
    "Our holiday offers have not yet been released. There may or may not be one that includes the iPhone X, so please keep checking. ^JD",
    "Have you checked to make sure your software is up to date? ^SL",
    "We have plenty of phones available. What happened to your phone? ^SL",
    "Email me at __email__; I can look into what you have been advised for accuracy. ^BR",
    "The iPhone X can be pre-ordered on Oct. 27th online, in-store, or by phone. Your current phone will need to be paid off to pre-order. ^KJ",
    "I can check into this with you. What device are you wanting and are you already a customer? ^BR",
    "We will be glad to have you. What is your zip code? I can check our service in your area. ^BR",
    "You can send an email to __email__ with 'Allie' in the subject line and I'll check for you. ^AS",
    "We have a tower in that area that is experiencing an issue. Our technicians are aware of the problem and are working on it. ^AS",
    "Try drying it out. If it won't work, you can activate an old device or purchase a pre-owned device with Telesales at 888-289-8722. ^LC",
    "Do you know if the phone has been activated on the new owner's account? ^KJ",
    "As long as your current device is paid off or you pay the early upgrade amount, you can certainly upgrade to the iPhone X. ^AW",
    "What happened to cause you to feel this way, Dillon? We are here to help. ^AS",
    "Exactly which phone did you order? ^AW",
    "What’s going on with your service? What’s the zip code, and I can look into this further. ^AP",
]

print(len(questions),len(answers))

language_model = get_llm()

# Ensure that necessary data is downloaded
nltk.download('punkt')

def calculate_bleu_score(reference: str, generated_output: str) -> float:
    """
    Calculate BLEU score for a given reference and generated output.
    
    Args:
    reference (str): The correct or expected sentence.
    generated_output (str): The sentence generated by the model/system.
    
    Returns:
    float: The BLEU score for the generated output.
    """
    # Tokenize the reference and generated text
    reference_tokens = nltk.word_tokenize(reference)
    generated_tokens = nltk.word_tokenize(generated_output)
    
    # Compute the BLEU score
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    
    return bleu_score

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
    

def blue_score_special_Arch():
        # Step: Calculate BLEU score for the generated response
    blue_scores = 0
    for i, q in enumerate(questions):
        # Reference answer for the current question
        reference_answer = answers[i]
        
        generated_context = process_query(q)   
        if isinstance(generated_context, list) and len(generated_context) > 0 and isinstance(generated_context[0], tuple):
            top_3_texts = [item[0] for item in sorted(generated_context, key=lambda x: x[1], reverse=True)[:3]]
        else:
            top_3_texts = []

        context = "\n".join(top_3_texts)
        query = q
        
        prompt = prompt_template.format(context=context, query=query)

        generated_answer = language_model(prompt)
        
        print('Answer',generated_answer.content)
        
        # Calculate BLEU score
        bleu_score = calculate_bleu_score(reference_answer, generated_answer.content)
        # Print the BLEU score for the current question
        print(f"BLEU score for Question {i+1}: {bleu_score:.4f}")
        
        blue_scores = blue_scores + bleu_score
    
    return blue_scores/34
        

# print('The average blue score is',blue_score_special_Arch())

def blue_score_rag():
    blue_scores = 0
    for i,q in enumerate(questions):
        generated_context = query_data(q)
        
        if isinstance(generated_context, list) and len(generated_context) > 0 and isinstance(generated_context[0], tuple):
            top_10_texts = [item[0] for item in sorted(generated_context, key=lambda x: x[1], reverse=True)[:3]]
        else:
            top_10_texts = []
            
        print(top_10_texts[:3])
        
        context = "\n".join(top_10_texts[:3])
        query = q
        reference_answer = answers[i]
        
        prompt = prompt_template.format(context=context, query=query)

        generated_answer = language_model(prompt)
        
        print('Answer',generated_answer.content)
        
        # Calculate BLEU score
        bleu_score = calculate_bleu_score(reference_answer, generated_answer.content)
        # Print the BLEU score for the current question
        print(f"BLEU score for Question {i+1}: {bleu_score:.4f}")
        
        blue_scores = blue_scores + bleu_score
    
    return blue_scores/34

print('The blue score using pure RAG is',blue_score_rag())