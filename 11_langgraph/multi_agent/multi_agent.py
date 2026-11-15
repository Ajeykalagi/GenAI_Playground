# multi_agent_authoritarian_selection.py

import functools
import random
from collections import OrderedDict
from typing import Callable, List
import tenacity
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class DialogueAgent:
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        message = self.model.invoke(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f"{name}: {message}")

class DialogueSimulator:
    def __init__(self, agents: List[DialogueAgent], selection_function: Callable[[int, List[DialogueAgent]], int]) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def step(self) -> tuple[str, str]:
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        message = speaker.send()
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        self._step += 1
        return speaker.name, message

class IntegerOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."

class DirectorDialogueAgent(DialogueAgent):
    def __init__(self, name, system_message: SystemMessage, model: ChatOpenAI, speakers: List[DialogueAgent], stopping_probability: float) -> None:
        super().__init__(name, system_message, model)
        self.speakers = speakers
        self.next_speaker = ""
        self.stop = False
        self.stopping_probability = stopping_probability
        self.termination_clause = "Finish the conversation by stating a concluding message and thanking everyone."
        self.continuation_clause = "Do not end the conversation. Keep the conversation going by adding your own ideas."

        self.response_prompt_template = PromptTemplate(
            input_variables=["message_history", "termination_clause"],
            template=f"""{{message_history}}

Follow up with an insightful comment.
{{termination_clause}}
{self.prefix}
            """
        )

        self.choice_parser = IntegerOutputParser(
            regex=r"<(\d+)>", output_keys=["choice"], default_output_key="choice"
        )
        self.choose_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "speaker_names"],
            template=f"""{{message_history}}

Given the above conversation, select the next speaker by choosing index next to their name: 
{{speaker_names}}

{self.choice_parser.get_format_instructions()}

Do nothing else.
            """
        )

        self.prompt_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "next_speaker"],
            template=f"""{{message_history}}

The next speaker is {{next_speaker}}. 
Prompt the next speaker to speak with an insightful question.
{self.prefix}
            """
        )

    def _generate_response(self):
        sample = random.uniform(0, 1)
        self.stop = sample < self.stopping_probability
        response_prompt = self.response_prompt_template.format(
            message_history="\n".join(self.message_history),
            termination_clause=self.termination_clause if self.stop else self.continuation_clause,
        )
        self.response = self.model.invoke(
            [self.system_message, HumanMessage(content=response_prompt)]
        ).content
        return self.response

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
        ),
        retry_error_callback=lambda retry_state: 0,
    )
    def _choose_next_speaker(self) -> int:
        speaker_names = "\n".join([f"{idx}: {name}" for idx, name in enumerate(self.speakers)])
        choice_prompt = self.choose_next_speaker_prompt_template.format(
            message_history="\n".join(self.message_history + [self.prefix] + [self.response]),
            speaker_names=speaker_names,
        )
        choice_string = self.model.invoke(
            [self.system_message, HumanMessage(content=choice_prompt)]
        ).content
        choice = int(self.choice_parser.parse(choice_string)["choice"])
        return choice

    def select_next_speaker(self) -> int:
        return self.chosen_speaker_id

    def send(self) -> str:
        self.response = self._generate_response()
        if self.stop:
            message = self.response
        else:
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            next_prompt = self.prompt_next_speaker_prompt_template.format(
                message_history="\n".join(self.message_history + [self.prefix] + [self.response]),
                next_speaker=self.next_speaker.name,
            )
            message = self.model.invoke(
                [self.system_message, HumanMessage(content=next_prompt)]
            ).content
            message = " ".join([self.response, message])
        return message

topic = "The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze"
director_name = "Jon Stewart"
agent_summaries = OrderedDict(
    {
        "Jon Stewart": ("Host of the Daily Show", "New York"),
        "Samantha Bee": ("Hollywood Correspondent", "Los Angeles"),
        "Aasif Mandvi": ("CIA Correspondent", "Washington D.C."),
        "Ronny Chieng": ("Average American Correspondent", "Cleveland, Ohio"),
    }
)
word_limit = 50

agent_summary_string = "\n- ".join(
    [""] + [f"{name}: {role}, located in {location}" for name, (role, location) in agent_summaries.items()]
)

conversation_description = f"""This is a Daily Show episode discussing the following topic: {topic}.

The episode features {agent_summary_string}."""

agent_descriptor_system_message = SystemMessage(content="You can add detail to the description of each person.")

def generate_agent_description(agent_name, agent_role, agent_location):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a creative description of {agent_name}, who is a {agent_role} in {agent_location}, that emphasizes their particular role and location.
            Speak directly to {agent_name} in {word_limit} words or less.
            Do not add anything else."""
        ),
    ]
    agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
    return agent_description

def generate_agent_header(agent_name, agent_role, agent_location, agent_description):
    return f"""{conversation_description}

Your name is {agent_name}, your role is {agent_role}, and you are located in {agent_location}.

Your description is as follows: {agent_description}

You are discussing the topic: {topic}.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.
"""

def generate_agent_system_message(agent_name, agent_header):
    return SystemMessage(
        content=(
            f"""{agent_header}
You will speak in the style of {agent_name}, and exaggerate your personality.
Do not say the same things over and over again.
Speak in the first person from the perspective of {agent_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {agent_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
"""
        )
    )

agent_descriptions = [
    generate_agent_description(name, role, location)
    for name, (role, location) in agent_summaries.items()
]
agent_headers = [
    generate_agent_header(name, role, location, description)
    for (name, (role, location)), description in zip(agent_summaries.items(), agent_descriptions)
]
agent_system_messages = [
    generate_agent_system_message(name, header)
    for name, header in zip(agent_summaries, agent_headers)
]

for name, description, header, system_message in zip(agent_summaries, agent_descriptions, agent_headers, agent_system_messages):
    print(f"\n\n{name} Description:")
    print(f"\n{description}")
    print(f"\nHeader:\n{header}")
    print(f"\nSystem Message:\n{system_message.content}")

topic_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{conversation_description}
        Please elaborate on the topic. 
        Frame the topic as a single question to be answered.
        Be creative and imaginative.
        Please reply with the specified topic in {word_limit} words or less. 
        Do not add anything else."""
    ),
]
specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")

def select_next_speaker(step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent) -> int:
    if step % 2 == 1:
        idx = 0
    else:
        idx = director.select_next_speaker() + 1
    return idx

director = DirectorDialogueAgent(
    name=director_name,
    system_message=agent_system_messages[0],
    model=ChatOpenAI(temperature=0.2),
    speakers=[DialogueAgent(name, system_message, ChatOpenAI(temperature=0.2)) for name, system_message in zip(agent_summaries.keys(), agent_system_messages) if name != director_name],
    stopping_probability=0.2,
)

agents = [director]
for name, system_message in zip(list(agent_summaries.keys())[1:], agent_system_messages[1:]):
    agents.append(DialogueAgent(name=name, system_message=system_message, model=ChatOpenAI(temperature=0.2)))

simulator = DialogueSimulator(agents=agents, selection_function=functools.partial(select_next_speaker, director=director))
simulator.reset()
simulator.inject("Audience member", specified_topic)
print(f"(Audience member): {specified_topic}")
print("\n")

while True:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print("\n")
    if director.stop:
        break
