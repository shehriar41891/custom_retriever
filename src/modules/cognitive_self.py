from crewai import Agent, Task, Crew, Process
from model.llm_model import get_llm
from dotenv import load_dotenv
import os 
import litellm
# from test.test_queries import spotifyQueries

litellm.set_verbose=True

load_dotenv()

openai_api = os.getenv('OPENAI_API')
# os.environ['OPENAI_API'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

litellm.api_key = openai_api

print(openai_api)

# llm = get_llm()
# print('The llm from cognitive side is ',llm)

Knowledge_Confidence_Evaluator = Agent(
    role='Query Answer Assistant',
    goal="""
    Evaluate the query {query} to identify the specific sector it refers to. For example,
    if the query asks about tracking a shipment but does not specify which company, respond with
    'Not sure'. Only provide an answer 
    if you are 100% certain about the sector being discussed. Otherwise, respond with 'Not sure'.
    """,
    verbose=True,
    memory=True,
    backstory=(
        """You must answer only if you are 100% confident. If unsure, respond with 'Not sure'.
        Follow this format when providing your response:

        Thought: [Your reasoning or confidence assessment]
        Final Answer: [Your best complete final answer]"""
    ),
    allow_delegation=True
)

ConfidenceEvaluation = Task(
    description=(
        """Analyze the query {query}. Answer only if you are 100% sure. 
        If you are unsure, respond with 'Not sure'. Follow this format:
        
        Thought: I now can give a great answer
        Final Answer: [your final answer here]"""
    ),
    expected_output="A precise and accurate answer or 'Not sure' if you are not 100% confident, using the correct format.",
    agent=Knowledge_Confidence_Evaluator,
    allow_delegation=True
)

crew = Crew(
    agents=[Knowledge_Confidence_Evaluator],
    tasks=[ConfidenceEvaluation],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=2
)

spotifyQueries = [
  "How do I get back into my account if I forgot the password?",
  "Why isn’t my playlist showing up on my other devices?",
  "How can I stop songs from playing automatically after my playlist ends?",
  "Why do I keep hearing ads even though I paid for Premium?",
  "How can I find songs I recently played?",
  "What’s the easiest way to share a playlist with a friend?",
  "How do I make my playlists private so no one can see them?",
  "Why is my music stopping when I switch apps?",
  "How do I save data when using Spotify on my phone?",
  "Why can’t I find certain songs or albums?",
  "What’s the deal with 'Spotify Wrapped'? How do I see mine?",
  "How do I get better sound quality on my Spotify?",
  "Can I play Spotify on my TV? How?",
  "Why is the app asking me to update my payment info?",
  "How do I add a new card for my subscription?",
  "Why can’t I skip songs anymore?",
  "How do I turn on shuffle?",
]

def get_result(user_query):
    result = crew.kickoff(inputs={'query' : user_query})
    return result 

print(get_result('Where is my package? Can I track it in real-time?'))

arr = []
for query in spotifyQueries:
    result = get_result(query)  # Get the result for the current query
    if result and hasattr(result, 'raw'):# Check if result has a 'raw' attribute
        arr.append(result.raw)  # Append only the 'raw' field to the array

print(arr)


# print(arr)