from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json


def semantic_meta(data_list: list, n: int = 5, task: str = 'semantic_match') -> dict:
    """ 
   Performs semantic matching or meta-labeling on a list of data.

    Args:
        data_list (list): A 2D array representing the input data.
        n (int, optional): The number of meta-categories to sort into (used for 'meta_label' task). Defaults to 5.
        task (str, optional): The task to perform: 'semantic_match' or 'meta_label'. Defaults to 'semantic_match'.

    Returns:
        dict: A dictionary containing the results of the semantic matching or meta-labeling.
    
    """
    # Assert only correct tasks 
    assert task in ["semantic_match", "meta_label"], "Task must be either 'semantic_match' or 'meta_label'. Invalid task: {task}"
    
    ## Prompt templates
    
    # Semantic matching system template
    system_message_semantic = """You are a data categorizer. You will be given a 2D array and create a new 1D array. 
    The new array will contain values which best semantically match one value from all the prior arrays.  
    The new values do not necessarily have to be present in any of the arrays. 
    The new values should be the most common way to refer to the values it is trying to represent. 
    Return a json with the new values as the keys and all the array values it maps to as the values."""
     
    # Meta categorization system template
    system_message_meta = """You will be provided a list and will sort them into {number} meta categories based on semantics.
                    Return each category as a json."""

    human_message = "{data}"
    
    # Formate templates into langchain chat template 
    template = ChatPromptTemplate.from_messages([
        ("system", system_message_semantic if task == 'semantic_match' else system_message_meta),     
        ("human", human_message)])
    
    if task == 'semantic_match':
        formatted_template = template.format_messages(data = data_list)
    else:
        formatted_template = template.format_messages(number = n, data = data_list)
    
    # Instantiate LLM class and get response
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0)
    response = llm(formatted_template).content
    
    return json.loads(response) 
