import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
# from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.sequential import SequentialChain

st.title('First App based on LangChain')
display_text = st.text_input("Type a topic/person to get information")


# memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
match_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me something about celebrity {name}"
)

# HuggingFaceHub LLMs
llm = HuggingFaceEndpoint(
    model_kwargs={
    "temperature": 0.5,
    "max_length": 64},
    repo_id = "google/flan-t5-xxl"
)

chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key='person',
    memory=person_memory
)

# Prompt Templates
second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="Tell 3 major matches of {person} career"
)

chain2=LLMChain(
    llm=llm, prompt=second_input_prompt,
    verbose=True, output_key='matches',
    memory=match_memory
)

parent_chain = SequentialChain(
    chains=[chain, chain2],
    input_variables = ['name'],
    output_variables=['person', 'matches'],
    verbose=True
)

if display_text:
    st.write(parent_chain({'name': display_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Matches'):
        st.info(match_memory.buffer)