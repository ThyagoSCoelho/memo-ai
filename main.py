import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI
from textwrap import dedent

class Tasks():
    def taking_doubts(self, agent, biograpy, doubts):
        return Task(
                description=dedent(f"""\
                        Você devera responder de forma humana a {doubts} referentes a o renomado cientista Stephen Halking, utilize as informações contidas no {biograpy}
                        """),
                expected_output=dedent("""\
                        Responda todas as perguntas em portugues e somente perguntas relacionadas a infomrações sobre o Stephen Hawking
                        """),
                agent=agent
        )

class Agents():  
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
    def agent_hawking(self):
        return Agent(
            name="AgentHawking",
            role="Educador",
            goal="Ensinar cosmologia e responder perguntas sobre a vida de Stephen Hawking.",
            backstory="Inspirado no renomado físico teórico Stephen Hawking, este agente é programado para compartilhar conhecimento detalhado sobre cosmologia e fatos da vida de Hawking. Utiliza um modelo de linguagem avançado para fornecer explicações claras e precisas em português.",
            verbose=True,
            llm=self.OpenAIGPT35,
            allow_delegation=False,
            language="pt"
        )
    
    
def extract_data_biograpy():
    file_path = 'biograpy.txt'

    with open(file_path, 'r') as file:
        return file.read()

def halwking(doubts):
    biograpy = extract_data_biograpy()
    
    agent = Agents()
    agent_hawking = agent.agent_hawking()
    
    tasks = Tasks()
    ask_questions = tasks.taking_doubts(agent_hawking, biograpy, doubts)
    
    crew = Crew(
        agents=[agent_hawking],
        tasks=[ask_questions]
    )
    result = crew.kickoff()
    return result


def main():
    print("## Memoria Biografico de Stephen Hawking")
    print('-------------------------------')
    print("Agente Hawking: Olá meu caro, como vai?! Me pergunte alguma curiosidade sobre o Sthepen Halking")
    
    while True:
        # Solicitando a entrada do usuário
        user_input = input("Digite: (ou 'sair' para encerrar): ")
        if user_input.lower() == 'sair':
            break
        
        # Gerando a resposta do modelo
        response = halwking(user_input)
        
        # Exibindo a resposta
        print("Hawking:", response)

if __name__ == "__main__":
    main()