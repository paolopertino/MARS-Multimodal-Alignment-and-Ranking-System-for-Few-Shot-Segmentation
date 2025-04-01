import json

from pydantic import BaseModel
from openai import OpenAI

from mars.components.LLM import LLM
from mars.components.helpers.prompts import SYSTEM_PROMPT_OPENAI, EXAMPLES_OPENAI


class SynonymObject(BaseModel):
    name: str
    description: str


class MeronymObject(BaseModel):
    name: str
    description: str


class WordObject(BaseModel):
    name: str
    description: str
    synonyms: list[SynonymObject]
    meronyms: list[MeronymObject]


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model_version: str = "gpt-4o-mini-2024-07-18", db_path: str = None):
        """Initializes the OpenAILLM class.

        :param api_key: The OpenAI API key.
        :type api_key: str
        :param model_version: openai model to use, defaults to "gpt-4o-mini-2024-07-18"
        :type model_version: str, optional
        :param db_path: path to the json file holding prefetched data, defaults to None
        :type db_path: str, optional
        """
        super().__init__(db_path)
        self.client = OpenAI(api_key=api_key)
        self.model_version = model_version

        # Check if the database file exists. If not, create it.
        self.db_content = {}
        if db_path is not None:
            try:
                with open(db_path, "r") as f:
                    self.db_content = json.load(f)
            except Exception as e:
                print(f"Cannot open {db_path}. - {e}")
                # raise ValueError()
                with open(db_path, "w") as f:
                    json.dump({}, f)

    def get_info(self, class_name: str) -> dict:
        """Fetches complete information about a class.

        The complete set of information includes the class name, description, synonyms, and meronyms.
        The method first look up the class in the local database, and if not found, it fetches the information from the llm.

        Here the structure of the response:

        ```python
        result = {
            "word": "<input_word>",
            "description": "<description_of_the_input_word>",
            "synonyms": [{"name" : "name_of_synonym_1", "description" : "description_of_synonym_1"}, {"name" : "name_of_synonym_2", "description" : "description_of_synonym_2"}, {"name" : "name_of_synonym_3", "description" : "description_of_synonym_3"}, ...],
            "meronyms": [{"name" : "name_of_meronym_1", "description" : "description_of_meronym_1"}, {"name" : "name_of_meronym_2", "description" : "description_of_meronym_2"}, {"name" : "name_of_meronym_3", "description" : "description_of_meronym_3"}, ...]
        }
        ```
        :param class_name: The name of the class to fetch information for.
        :type class_name: str
        :return: A dictionary containing the class information
        :rtype: dict
        """
        # Look for the class in the local database. The local database is a
        # JSON file that contains information about objects.
        if class_name in self.db_content:
            return self.db_content[class_name]

        # If the class is not found in the local database, fetch the information
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_OPENAI}
        ] + EXAMPLES_OPENAI
        completion = self.client.beta.chat.completions.parse(
            model=self.model_version,
            messages=messages + [{"role": "user", "content": class_name}],
            response_format=WordObject,
            temperature=0.0,
        )

        parsed_response = completion.choices[0].message.parsed
        result = {
            "word": parsed_response.name,
            "description": parsed_response.description,
            "synonyms": [{"name": e.name, "description": e.description} for e in parsed_response.synonyms],
            "meronyms": [{"name": e.name, "description": e.description} for e in parsed_response.meronyms]
        }
        self.db_content[class_name] = result

        # Update the local database with the new information.
        if self.db_path is not None:
            with open(self.db_path, "r") as f:
                data = json.load(f)
                data[class_name] = result

            with open(self.db_path, "w") as f:
                json.dump(data, f)

        return result
