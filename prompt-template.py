import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
# from langchain.chains.llm import LLMChain
# from langchain.chains import LLMChain
# from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
# from langchain.chains.sequential import SequentialChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain.schema import StrOutputParser
from langchain_core.output_parsers import StrOutputParser

st.title('First App based on LangChain')
display_text = st.text_input("Type a person to get information")


# memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
match_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')

# Prompt Templates
# first_input_prompt = PromptTemplate.from_template(
    # input_variables = ['name'],
    # template = "Tell me something about celebrity {name}"
# )
# Prompt Templates
first_input_prompt = PromptTemplate.from_template("Tell me something about celebrity {name}")


# Step 2: Create a prompt template
# prompt = PromptTemplate(
    # input_variables=["chat_history", "user_input"],
    # template="""
    # The following is a conversation:
    # {chat_history}
    # User: {user_input}
    # AI:"""
# )

# HuggingFaceHub LLMs
llm = HuggingFaceEndpoint(
    repo_id = "google/flan-t5-xxl",
    temperature = 0.5,
    max_length = 654
)


# chain = LLMChain(
    # llm=llm,
    # prompt=first_input_prompt,
    # verbose=True,
    # output_key='person',
    # memory=person_memory
# )

# Prompt Templates
# second_input_prompt=PromptTemplate(
    # input_variables=['person'],
    # template="Tell 3 major matches of {person} career"
# )
# Prompt Templates
second_input_prompt=PromptTemplate.from_template("Tell 3 major matches of {person} career")


# chain2=LLMChain(
    # llm=llm, prompt=second_input_prompt,
    # verbose=True, output_key='matches',
    # memory=match_memory
# )

# parent_chain = SequentialChain(
    # chains=[chain, chain2],
    # input_variables = ['name'],
    # output_variables=['person', 'matches'],
    # verbose=True
# )

first_chain = first_input_prompt | llm | StrOutputParser()
second_chain = second_input_prompt | llm | StrOutputParser()
# optimization_chain = optimization_prompt | llm | StrOutputParser()


chain = ({"structure" : first_chain} 
        | RunnablePassthrough.assign(review=second_chain))

if display_text:
    st.write(first_chain({'name': display_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Matches'):
        st.info(match_memory.buffer)


