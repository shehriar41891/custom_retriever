from crewai import Agent, Task, Crew, Process
# from models.llm_model import get_llm
from dotenv import load_dotenv
import os 
import litellm

litellm.set_verbose=True

load_dotenv()

openai_api = os.getenv('OPENAI_API')
# os.environ['OPENAI_API'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

litellm.api_key = openai_api

Evaluator = Agent(
    role='Evaluation of the documents',
    goal="""
    Evaluate the given 10 documents {documents} based on the query: {query} and take the top 3 out of 
    them which you think contain the answer to the query: {query}
    """,
    verbose=True,
    memory=True,
    backstory=(
        """You are an document evaluator who looks carefully on the documents and query and take top 3
        most matched documents with query"""
    ),
    allow_delegation=True
)

Evaluation = Task(
    description=(
        """Analyze the documents {documents} based on the query {query}. Take top 3 documents
        from given 10 documents which you think contains the answer to the {query}
        rank them from most relevant to less relevant        
        """
    ),
    expected_output="Top 3 relevant documents to the given query",
    agent=Evaluator,
    allow_delegation=True
)

crew = Crew(
    agents=[Evaluator],
    tasks=[Evaluation],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=2
)


def filter_top3(user_query,documents):
    result = crew.kickoff(inputs={'documents': documents,'query' : user_query})
    return result 

data = {'matches': [{'id': '158',
              'metadata': {'text': '@129336 Would you like me to look at your '
                                   'data usage for you? You can also check it '
                                   'online at https://t.co/QO2Q75BSGA ^KJ'},
              'score': 0.8534884,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '159',
              'metadata': {'text': "@129336 Do you know how much data you've "
                                   'used and how much is left in your plan? '
                                   '^KJ'},
              'score': 0.8447256,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '14',
              'metadata': {'text': '@134780 If you are on a Total Plan, once '
                                   'you have met your allowance, your data '
                                   'speeds are reduced for the month. You can '
                                   'help improve data speeds &amp; data usage '
                                   'by closing out of apps that are not in use '
                                   '&amp; streaming on SD vs HD in addition to '
                                   'updating your device while on Wi-Fi. ^LC'},
              'score': 0.8258266,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '225',
              'metadata': {'text': '@211633 Do you know how much data you have '
                                   'used? After 22GB, data speeds are reduced '
                                   'to 2G for the remainder of the billing '
                                   'cycle. ^LC'},
              'score': 0.8236319,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '880',
              'metadata': {'text': '@643270 You can view the amount online at '
                                   'https://t.co/iGBvQXSyy4. Once there hove '
                                   'over Device &amp; Plans then select Device '
                                   '&amp; Upgrade Info. ^AW'},
              'score': 0.8188178,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '766',
              'metadata': {'text': '@581767 Do you know how much data is '
                                   'included in your plan? ^KJ'},
              'score': 0.8180731,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '281',
              'metadata': {'text': '@261304 Which data plan do you have? Any '
                                   'idea how much data you have used thus far '
                                   'this bill cycle? ^JD'},
              'score': 0.8166827,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '1073',
              'metadata': {'text': "@750251 Thank you. I don't show any issues "
                                   'in that area. Which data plan do you have? '
                                   'Do you know how much data you have used so '
                                   'far this bill cycle? ^JD'},
              'score': 0.81331915,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '173',
              'metadata': {'text': "@170513 Let's see what's going on. How "
                                   'much data is included in your plan? ^KJ'},
              'score': 0.81299454,
              'sparse_values': {'indices': [], 'values': []},
              'values': []},
             {'id': '773',
              'metadata': {'text': "@581768 Let's check the network settings "
                                   'on your phone. To do this go to Settings '
                                   '&gt; Cellular &gt; Cellular Data Options '
                                   '&gt; Voice/Data roaming On. ^KJ'},
              'score': 0.81179416,
              'sparse_values': {'indices': [], 'values': []},
              'values': []}],
 'namespace': 'namespace6',
 'usage': {'read_units': 6}
 }

texts = [match['metadata']['text'] for match in data['matches']]

query = 'How can I check my data usage and remaining balance'

print(filter_top3(query,texts))