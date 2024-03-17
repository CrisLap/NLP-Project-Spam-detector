import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_trf', disable=['parser'])

def extract_entities(document, ent_type):
    """
    Extract entities of a specific type from a document.

    Args:
        document (str): The text document to extract entities from.
        ent_type (str): The type of entities to extract (e.g., "ORG" for organizations).

    Returns:
        list: A list of strings representing entities of the specified type.

    This function processes the input text document using a natural language processing (NLP) pipeline,
    then iterates through the tokens in the document to identify entities of the specified type.
    It returns a list of strings representing the identified entities.

    """
    organizations = []
    doc = nlp(document)
    for token in doc:
        if str(token.ent_type_) == ent_type:
            organizations.append(str(token))
            
    return organizations

def get_organizations(corpus):
    """
    Extracts organizations from a corpus of documents.

    Args:
    corpus (list): A list of strings representing the documents to be analyzed.

    Returns:
    dict: A dictionary where keys are the documents from the corpus and values are
    lists of organizations extracted from each document.

    Example:
    >>> corpus = ["Apple Inc. is a technology company.", "Google is an Internet company."]
    >>> get_organizations(corpus)
    {'Apple Inc. is a technology company.': ['Apple Inc.'], 'Google is an Internet company.': ['Google']}
    """
    organizations_dict = {}
    
    for document in tqdm(corpus, desc="Extracting organizations", unit="document"):
        organizations_dict[document] = extract_entities(document, "ORG")
    
    return organizations_dict