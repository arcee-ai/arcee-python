from datasets import load_dataset
from langchain import PromptTemplate

default_response_template = PromptTemplate(
    input_variables=["response"],
    template="### Response: {response}",
)

default_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="### Question: {question}",
)

class PromptSet():
    """
        A PromptSet is a collection of filled in prompts templates and responses for a given task.
        
        Each entry in the PromptSet has a PromptTemplate that is used for the first part of the training data prompt. 
        And each entry has a ResponseTemplate for the second part of the training data prompt.
        
        At inference time, the PromptTemplate is given to the LM 
        and the ResponseTemplate is used to extract the response from the LM output.
    """
    
    def __init__(self, data_file, prompt_template=None, response_template=None):
        
        if not data_file.endswith(".csv"):
            raise("Datafile must be a .csv file")
        
        self.dataset = load_dataset("csv", data_files=data_file, split="train")
        dataset_fields = list(self.dataset.features.keys())
        self.dataset_fields = dataset_fields
        
        if response_template is None:
            if "response" not in dataset_fields:
                raise("QA datafile must have a column named 'question' and 'answer'")
            else:
                self.response_template = default_response_template
        else:
            if not all(item in dataset_fields for item in response_template.input_variables):
                raise("Response template input variables must be in the dataset")
        
        if prompt_template is None:
            if "question" not in dataset_fields:
                raise("Datafile must have a column named 'question' when using default prompt template")
            else:
                self.prompt_template = default_prompt_template
        else:
            if not all(item in dataset_fields for item in prompt_template.input_variables):
                raise("Prompt template input variables must be in the dataset")
            else:
                self.prompt_template = prompt_template
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def formatting_prompts_func(self, example):
        """
            This function takes a dataset example and returns a formatted prompt string.
            This formatted example will be passed to the LM for training.
        """
        
        example_for_prompt = {key: example[key] for key in self.prompt_template.input_variables if key in example}
        example_for_response = {key: example[key] for key in self.response_template.input_variables if key in example}
        
        formatted_prompt = self.prompt_template.format(**example_for_prompt)
        formatted_response = self.response_template.format(**example_for_response)
        
        print("FORMATTED: ", formatted_prompt+formatted_response)
        
        return formatted_prompt + " " + formatted_response
    
    def generate_examples(self, output_file):
        
        if "OPENAI_API_KEY" not in os.environ:
            raise("OPENAI_API_KEY environment variable must be set to use the generate_examples method")
        
        #logic to iterate through dataset examples and generate prompts for each
        #write generated prompts to new file