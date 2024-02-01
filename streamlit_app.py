import traceback
import json
import pandas as pd
from dotenv import load_dotenv
from src.mcq_generator.utils import read_file, get_table_data
from src.mcq_generator.logger import logging
from src.mcq_generator.mcq_gen import final_chain
import streamlit as st
from langchain_community.callbacks import get_openai_callback

with open('/Users/abhinavnagaboyina/Documents/Gen_AI_Practice/mcq_generator/response.json','r') as file:
    Response_json= json.load(file)

st.title("MCQ Creator Application🤖")

with st.form("user_inputs"):
    uploaded_file= st.file_uploader("Upload either a pdf file or text file")
    count= st.number_input("No of mcqs", min_value=3, max_value= 15)
    subject= st.text_input("Insert the subject name", max_chars=20)
    tone= st.text_input("Complexity level of questions", max_chars=20, placeholder='Simple')
    button= st.form_submit_button("Create the MCQs")

    if button and uploaded_file is not None and count and subject and tone:
        with st.spinner("loading..."):
            try:
                text= read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response= final_chain.invoke(
                        {
                            "text": text,
                            "count": count,
                            "subject": subject, 
                            "tone": tone,
                            "response_json": json.dumps(Response_json)
                        }
                    )
                st.sidebar.write(f"Total Tokens:{cb.total_tokens}")
                st.sidebar.write(f"Prompt Tokens:{cb.prompt_tokens}")
                st.sidebar.write(f"Completion Tokens:{cb.completion_tokens}")
                st.sidebar.write(f"Total Cost:{cb.total_cost}")

                if isinstance(response, dict):
                    quiz= response.get("quiz")
                    if quiz:
                        table_data= get_table_data(quiz)
                        print(table_data)
                        if table_data is not None:
                            df= pd.DataFrame(table_data)
                            df.index= df.index+1
                            st.table(df)
                            st.text_area(label="Review", value= response["review"])
                        else:
                                st.error("Error in the table data")
                    else:
                        st.write(response)     
                #st.write(response)
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
           
                
            
                