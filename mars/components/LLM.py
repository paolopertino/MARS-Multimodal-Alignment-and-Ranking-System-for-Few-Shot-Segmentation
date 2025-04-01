import abc

class LLM(abc.ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_info') and
                callable(subclass.get_info) or
                NotImplemented)
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path

    @abc.abstractmethod
    def get_info(self, class_name: str) -> dict:
        """Fetches complete information about a class.
        
        The complete set of information includes the class name, description, synonyms, and meronyms.
        The method first look up the class in the local database, and if not found, it fetches the information from the llm.

        :param class_name: The name of the class to fetch information for.
        :type class_name: str
        :raises NotImplementedError: If the method is not implemented by the subclass.
        :return: A dictionary containing the class information
        :rtype: dict
        """
        raise NotImplementedError