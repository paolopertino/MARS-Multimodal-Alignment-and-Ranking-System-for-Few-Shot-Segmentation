import abc
import os

import numpy as np

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from mars.components.OpenAILLM import OpenAILLM
from mars.components.LlamaLLM import LlamaLLM
from mars.utils.coco_prompts import compile_coco_prompts, coco_class_to_synset_map
from mars.components.LLM import LLM
from mars.components.VLM import VLM


class TextFetcher(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_text_name_only_wordnet') and
                callable(subclass.get_text_name_only_wordnet) and
                hasattr(subclass, 'get_text_name_only_llm') and
                callable(subclass.get_text_name_only_llm) and
                hasattr(subclass, 'get_text_synonyms_wordnet') and
                callable(subclass.get_text_synonyms_wordnet) and
                hasattr(subclass, 'get_text_synonyms_llm') and
                callable(subclass.get_text_synonyms_llm) and
                hasattr(subclass, 'get_text_with_description_wordnet') and
                callable(subclass.get_text_meronyms_wordnet) and
                hasattr(subclass, 'get_text_with_description_llm') and
                callable(subclass.get_text_meronyms_llm) or
                NotImplemented)

    def __init__(self, configs: dict):
        """Initialize the TextFetcher class.

        :param llm: the llm used by the text fetcher, defaults to None if LLM is not used
        :type llm: LLM, optional
        """
        self.validate_configs(configs)

    @abc.abstractmethod
    def get_text_name_only_wordnet(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get prompts for the target object with the class name only.

        If use_description is set to True, the description of the target object is
        fetched from wordnet and used in the prompts.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target object, defaults to False
        :type use_description: bool, optional
        :return: prompts for the target object with the class name only
        :rtype: list[str]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_name_only_llm(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get prompts for the target object with the class name only.

        If use_description is set to True, the description of the target object is
        fetched from a large language model and used in the prompts.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target object, defaults to False
        :type use_description: bool, optional
        :return: prompts for the target object with the class name only
        :rtype: list[str]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_synonyms_wordnet(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get prompts for the target object with the class name and synonyms
        provided by wordnet.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target object, defaults to False
        :type use_description: bool, optional
        :return: prompts for the target object with the class name and synonyms
        :rtype: list[str]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_synonyms_llm(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get prompts for the target object with the class name and synonyms
        provided by a large language model.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target object, defaults to False
        :type use_description: bool, optional
        :return: prompts for the target object with the class name and synonyms
        :rtype: list[str]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_meronyms_wordnet(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get prompts for the target object with the class name and meronyms
        provided by wordnet.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target object, defaults to False
        :type use_description: bool, optional
        :return: prompts for the target object with the class name and meronyms
        :rtype: list[str]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_meronyms_llm(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get prompts for the target object with the class name and meronyms
        provided by a large language model.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target object, defaults to False
        :type use_description: bool, optional
        :return: prompts for the target object with the class name and meronyms
        :rtype: list[str]
        """
        raise NotImplementedError

    def validate_configs(self, configs: dict) -> None:
        """Validate the configs for the TextFetcher class.

        :param configs: configurations for the TextFetcher class
        :type configs: dict
        """
        # Only one of the available llms can be used at a time
        if configs.get('use_openai_llm') and configs.get('use_llama_llm'):
            raise ValueError(
                "Only one of 'use_openai_llm' or 'use_llama_llm' can be True.")

        # At least one of the text sources must be used
        if not configs.get('use_wordnet') and not configs.get('use_openai_llm') and not configs.get('use_llama_llm'):
            raise ValueError(
                "At least one of 'use_wordnet', 'use_openai_llm', or 'use_llama_llm' must be True.")

        if configs.get('use_openai_llm') or configs.get('use_wordnet_openai_fallback'):
            llm_key = configs.get('llm_key') if configs.get(
                'llm_key') is not None else os.getenv("OPENAI_API_KEY")

            if llm_key is None:
                raise ValueError("OpenAI API key is not provided.")

            self.llm: OpenAILLM = OpenAILLM(
                llm_key, db_path=configs.get('description_db_path'))
        elif configs.get('use_llama_llm'):
            llm_key = configs.get('llm_key') if configs.get('llm_key') is not None else os.getenv(
                "HF_TOKEN")

            if llm_key is None:
                raise ValueError("Hugging Face Token is not provided.")

            self.llm: LlamaLLM = LlamaLLM(
                llm_key, db_path=configs.get('description_db_path'))
        else:
            self.llm = None

        self.use_wordnet_gt_mapping = configs.get(
            'use_wordnet_gt_mapping', False)
        self.vlm = None
        self.img = None
        self.mask = None

    def get_synset(self, class_name: str) -> str:
        """Get the synset for the class name.

        :param class_name: class name of the object for which the wordnet synset must be fetched.
        :type class_name: str
        :return: wordnet synset identifier for the class name
        :rtype: str
        """
        lower_class_name = class_name.strip().lower()
        stop_words = set(stopwords.words('english'))

        # If the class name is composed by multiple words,
        # to match the synset most often the synset is matched
        # by changing the spaces with underscores.
        synsets = []
        synsets += wn.synsets(lower_class_name.replace(' ', '_'), pos=wn.NOUN)

        # In other cases the synset is matched by the class name
        # with the spaces removed
        if len(synsets) == 0:
            synsets += wn.synsets(lower_class_name.replace(' ',
                                  ''), pos=wn.NOUN)

        # If no synset is still found, then try to match the subwords of the class name
        if len(synsets) == 0:
            for word in lower_class_name.split():
                synsets += wn.synsets(word.strip(), pos=wn.NOUN)

        # If no synset is found, return None
        if len(synsets) == 0:
            return None

        # If a single synset is found, return the name of the synset
        if len(synsets) == 1:
            return synsets[0].name()

        # If multiple synsets are found, the synset is matched
        # using the description of the object.
        print(
            f"[TextFetcher] - Multiple synsets found for {class_name}. Matching using description.")
        best_synset = None
        max_overlap = 0
        class_description = self.vlm.fetch_definition(
            self.img, self.mask, class_name)
        description_tokens = set(word_tokenize(
            class_description.lower())) - stop_words

        for synset in synsets:
            definition_tokens = set(word_tokenize(
                synset.definition().lower())) - stop_words
            # Count overlapping words
            overlap = len(description_tokens & definition_tokens)

            if overlap > max_overlap:
                max_overlap = overlap
                best_synset = synset

        return best_synset.name() if best_synset else None

    def set_vlm(self, vlm: VLM) -> None:
        """Set the VLM used by the text fetcher.

        :param vlm: the VLM used by the text fetcher
        :type vlm: VLM
        """
        self.vlm = vlm

    def set_img(self, img: np.ndarray) -> None:
        """Set the image used by the text fetcher.

        :param img: the image used by the text fetcher
        :type img: np.ndarray
        """
        self.img = img

    def set_mask(self, mask: np.ndarray) -> None:
        """Set the mask used by the text fetcher.

        :param mask: the mask used by the text fetcher
        :type mask: np.ndarray
        """
        self.mask = mask


class COCOTextFetcher(TextFetcher):
    def __init__(self, configs: dict) -> None:
        super().__init__(configs=configs)

    def get_text_name_only_wordnet(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get COCO prompts for the target object with the class name only.

        If use_description is set to True, the description of the target object is
        fetched from wordnet and used in the prompts.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target objet, defaults to False
        :type use_description: bool, optional
        :return: COCO prompts for the target object with the class name only
        :rtype: list[str]
        """
        if use_description:
            synset = coco_class_to_synset_map.get(
                class_name) if self.use_wordnet_gt_mapping else self.get_synset(class_name)

            # There might be cases in which the synset is not available.
            # In such cases, we will return the COCO prompts with the class name only.
            if synset is None:
                print(f"Synset not found for {class_name}")
                if self.llm is not None:
                    print(
                        f"[Wordnet Fallback Mechanism] - Fetching description from LLM for {class_name}")
                    class_description = self.llm.get_info(
                        class_name).get('description', '')
                    return compile_coco_prompts(class_name, class_description)
                else:
                    return compile_coco_prompts(class_name)

            class_synset = wn.synset(synset)
            class_description = class_synset.definition()

            return compile_coco_prompts(class_name, class_description)

        return compile_coco_prompts(class_name)

    def get_text_name_only_llm(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get COCO prompts for the target object with the class name only.

        If use_description is set to True, the description of the target object is
        fetched from wordnet and used in the prompts.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target objet, defaults to False
        :type use_description: bool, optional
        :return: COCO prompts for the target object with the class name only
        :rtype: list[str]
        """
        if use_description:
            class_info = self.llm.get_info(class_name)

            return compile_coco_prompts(class_name=class_name, description=class_info.get('description', ''))

        return compile_coco_prompts(class_name)

    def get_text_synonyms_wordnet(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get COCO prompts for the target object with the class name and synonyms
        provided by wordnet.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target objet, defaults to False
        :type use_description: bool, optional
        :return: COCO prompts for the target object with the class name and synonyms
        :rtype: list[str]
        """
        synset = coco_class_to_synset_map.get(
            class_name) if self.use_wordnet_gt_mapping else self.get_synset(class_name)

        # There might be cases in which the synset is not available.
        # In such cases, we will return the COCO prompts with the class name only.
        if synset is None:
            return compile_coco_prompts(class_name)

        class_synset = wn.synset(synset)
        class_description = class_synset.definition()
        synonyms = set()

        for synonym in class_synset.lemma_names():
            synonyms.add(synonym)

        # The set contains the class name and the synonyms if any.
        # Compiling the COCO prompts with the class name and synonyms.
        prompts = []

        for synonym in synonyms:
            if use_description:
                prompts.extend(compile_coco_prompts(
                    synonym.replace("_", " "), class_description))
            else:
                prompts.extend(compile_coco_prompts(synonym.replace("_", " ")))

        return prompts

    def get_text_synonyms_llm(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get COCO prompts for the target object with the class name and synonyms
        provided by a large language model.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target objet, defaults to False
        :type use_description: bool, optional
        :return: COCO prompts for the target object with the class name and synonyms
        :rtype: list[str]
        """
        class_info = self.llm.get_info(class_name)
        synonyms = class_info.get('synonyms', [])

        # Initialize the prompts for the class name and possibly the description.
        prompts = compile_coco_prompts(class_name, class_info.get(
            'description', '')) if use_description else compile_coco_prompts(class_name)

        for synonym in synonyms:
            if use_description:
                prompts.extend(compile_coco_prompts(
                    synonym.get('name'), synonym.get('description', '')))
            else:
                prompts.extend(compile_coco_prompts(synonym.get('name')))

        return prompts

    def get_text_meronyms_wordnet(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get COCO prompts for the target object with the class name and meronyms
        provided by wordnet.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target objet, defaults to False
        :type use_description: bool, optional
        :return: COCO prompts for the target object with the class name and description
        :rtype: list[str]
        """
        synset = coco_class_to_synset_map.get(
            class_name) if self.use_wordnet_gt_mapping else self.get_synset(class_name)

        # There might be cases in which the synset is not available.
        # In such cases, we will return the COCO prompts with the class name only.
        if synset is None:
            return compile_coco_prompts(class_name)

        class_synset = wn.synset(synset)
        class_meronyms = class_synset.part_meronyms()
        meronyms_prompts = []

        for meronym in class_meronyms:
            meronym_name = meronym.name().split(".")[0].replace("_", " ")

            if use_description:
                meronym_description = meronym.definition()
            else:
                meronym_description = None

            meronyms_prompts.extend(compile_coco_prompts(
                meronym_name, meronym_description))

        return meronyms_prompts

    def get_text_meronyms_llm(self, class_name: str, use_description: bool = False) -> list[str]:
        """Get COCO prompts for the target object with the class name and meronyms
        provided by a llm.

        :param class_name: class name of the target object
        :type class_name: str
        :param use_description: whether to use the description of the target objet, defaults to False
        :type use_description: bool, optional
        :return: COCO prompts for the target object with the class name and description
        :rtype: list[str]
        """
        raise NotImplementedError(
            "The method get_text_with_description_llm is not implemented yet.")
