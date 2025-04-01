from mars.components.TextFetcher import TextFetcher


class PromptGenerator():
    def __init__(self, text_fetcher: TextFetcher, prompt_usage_cfg: dict):
        self.text_fetcher = text_fetcher
        self.prompt_usage_cfg = prompt_usage_cfg
        self.prompts = []
        self.logger = None

    def prompt_generation(self, class_name: str) -> list[str]:
        """Fetch the prompts for the target object.

        Returns a list of sentences that describe the target object.

        :param class_name: name of the target object
        :type class_name: str
        :return: list of prompts for the target object
        :rtype: list[str]
        """
        self.prompts = []

        if self.prompt_usage_cfg['use_wordnet']:
            # The list of synonyms of a word in wordnet contains also the word itself.
            if self.prompt_usage_cfg['use_class_name_only']:
                self.prompts += self.text_fetcher.get_text_name_only_wordnet(
                    class_name, self.prompt_usage_cfg['use_descriptions'])
            elif self.prompt_usage_cfg['use_synonyms']:
                self.prompts += self.text_fetcher.get_text_synonyms_wordnet(
                    class_name, self.prompt_usage_cfg['use_descriptions'])

            if self.prompt_usage_cfg['use_meronyms']:
                self.prompts += self.text_fetcher.get_text_meronyms_wordnet(
                    class_name, self.prompt_usage_cfg['use_descriptions'])
        else:
            if self.prompt_usage_cfg['use_class_name_only']:
                self.prompts += self.text_fetcher.get_text_name_only_llm(
                    class_name, self.prompt_usage_cfg['use_descriptions'])

            if self.prompt_usage_cfg['use_synonyms']:
                self.prompts += self.text_fetcher.get_text_synonyms_llm(
                    class_name, self.prompt_usage_cfg['use_descriptions'])

            if self.prompt_usage_cfg['use_meronyms']:
                self.prompts += self.text_fetcher.get_text_meronyms_llm(
                    class_name, self.prompt_usage_cfg['use_descriptions'])

        return self.prompts
    
    def set_logger(self, logger):
        """Set the logger object.

        :param logger: logger object
        :type logger: CometLogger
        """
        self.logger = logger

    @property
    def configs(self):
        return self.prompt_usage_cfg

    @classmethod
    def prompt_usage_cfg_sanity_checks(cls, prompt_usage_cfg: dict) -> bool:
        """Checks whether the prompt usage configuration is valid.

        If only the class name should be used for the prompt, then the other options 
        regarding the usage of synonyms should be False and viceversa.

        Synonyms and meronyms and descriptions can be used together.

        :param prompt_usage_cfg: _description_
        :type prompt_usage_cfg: dict
        :return: _description_
        :rtype: bool
        """
        if prompt_usage_cfg['use_class_name_only']:
            if prompt_usage_cfg['use_synonyms']:
                return False

        if not prompt_usage_cfg['use_class_name_only'] and not prompt_usage_cfg['use_synonyms'] and not prompt_usage_cfg['use_meronyms']:
            return False

        return True
