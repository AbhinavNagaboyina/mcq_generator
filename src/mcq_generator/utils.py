import os
import pypdf
import json
import traceback
from langchain_community.document_loaders import PyPDFLoader


def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            loader = PyPDFLoader(file)
            pages = loader.load_and_split()
            text=""
            for page in pages:
                text= page.extract_text()
            return text
        
        except Exception as e:
            raise Exception("Error in reading the file")
    
    elif file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    
    else:
        raise Exception(
            "Unsupported file format only pdf or text file formats will be supported"
        )

def get_table_data(quiz_str):
    try:
        quiz_dict= json.loads(quiz_str)
        quiz_table_data=[]

        for key,value in quiz_dict.items():
            mcq = value["mcq"]
            options = " || ".join(
                [
                    f"{option}: {option_value}"
                    for option, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
            
        return quiz_table_data

    except Exception as e:
        traceback.print_exception(type(e),e,e.__traceback__)
        return False
