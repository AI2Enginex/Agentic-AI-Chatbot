from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser

# creating an output parser class
class PydanticOutputs:
    '''
    class for declaring the output parsers
    1. PydanticOutputParser
    2. CommaSeparatedListOutputParser
    '''
    # the LLM should return a 
    # pydantic as response
    @classmethod
    def DictOutputParser(cls,method: None):

        try:
            return PydanticOutputParser(pydantic_object=method)
        except Exception as e:
            return e
        
    # the LLM should return a 
    # list of items
    # seperated by commas
    @classmethod
    def CommaSeperatedParser(cls):

        try:
            return CommaSeparatedListOutputParser()
        except Exception as e:
            return e
    
    @classmethod
    def OtherOutputParser(cls):

        try:
            pass
        except Exception as e:
            return e