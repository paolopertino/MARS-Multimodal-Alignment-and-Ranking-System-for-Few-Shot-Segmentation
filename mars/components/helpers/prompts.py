SYSTEM_PROMPT_OPENAI = """
You are an AI agent specialized in linguistic analysis. Your task is to receive a single word as input from the user and return a valid JSON object containing the following:

1. The **word** that was input.
2. A **description** of the input word.
3. A **list of all the synonyms of input word** with description (words with same meaning of the input word).
4. A **list of all the meronyms/parts of input word with their definition** (objects or parts that are visible components of the input word).

The JSON object should follow this structure:

{
    "word": "<input_word>",
    "description": "<description_of_the_input_word>",
    "synonyms": [{"name" : "name_of_synonym_1", "description" : "description_of_synonym_1"}, {"name" : "name_of_synonym_2", "description" : "description_of_synonym_2"}, {"name" : "name_of_synonym_3", "description" : "description_of_synonym_3"}, ...],
    "meronyms": [{"name" : "name_of_meronym_1", "description" : "description_of_meronym_1"}, {"name" : "name_of_meronym_2", "description" : "description_of_meronym_2"}, {"name" : "name_of_meronym_3", "description" : "description_of_meronym_3"}, ...]
}

Descriptions will be used in CLIP-based tasks, so make sure they are clear and concise.
If no synonyms or meronyms are available, you can provide empty lists for those fields.
Your response must strictly adhere to the JSON format and avoid any additional text outside the JSON object.
"""

SYSTEM_PROMPT_TEMPLATE_VLM_VIP_LLAVA = "Human: <image>\n{}\nAssistant:"

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

VISUAL_PROMPTS_VLM_VIP_LLAVA = {
    "mask": "What is the name of the object highlighted with a {} mask in the image? Your output must be only the class name of the object. Do not add ny additional text.",
    "bb": "What is the name of the object inside the {} rectangle in the image? Your output must be only the class name of the object. Do not add ny additional text.",
    "contour": "What is the name of the object inside the {} mask contour? Your output must be only the class name of the object. Do not add ny additional text.",
    "ellipse": "What is the name of the object inside the {} ellipses? Your output must be only the class name of the object. Do not add ny additional text.",
}

VISUAL_PROMPTS_DESCRIPTIONS_VLM_VIP_LLAVA = {
    "mask": "Given the image provided, identify and provide the definition of the {} highlighted by the {} mask.",
    "bb": "Given the image provided, identify and provide the definition of the {} inside the {} rectangle.",
    "contour": "Given the image provided, identify and provide the definition of the {} inside the {} mask contour.",
    "ellipse": "Given the image provided, identify and provide the definition of the {} inside the {} ellipses. ",
}
# VISUAL_PROMPTS_DESCRIPTIONS_VLM_VIP_LLAVA = {
#     "mask": "This image shows a {} highlighted by the {} mask. Please provide the vocabulary definition of the word {}. Name {}, definition:",
#     "bb": "This image shows a {} highlighted by the {} rectangle. Please provide the vocabulary definition of the word {}. Name {}, definition:",
#     "contour": "This image shows a {} highlighted by the {} mask contour. Please provide the vocabulary definition of the word {}. Name {}, definition:",
#     "ellipse": "This image shows a {} highlighted by the {} ellipses. Please provide the vocabulary definition of the word {}. Name {}, definition:",
# }

EXAMPLES_OPENAI = [
    {"role": "user", "content": "apple"},
    {"role": "assistant", "content": """{"word": "apple", "description": "the round fruit of a tree of the rose family, which typically has thin red or green skin and crisp flesh.", "synonyms": [{"name" : "fruit", "description" : "the sweet and fleshy product of a tree or other plant that contains seed and can be eaten as food."}], "meronyms": [{"name" : "peel", "description" : "The thin outer skin of the apple, varying in color."}, {"name" : "stalk", "description" : "The short stem that attaches the apple to the tree."}, {"name" : "seeds", "description" : "The small, hard, oval-shaped seeds found inside the apple."}]}"""},
]