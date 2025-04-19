"""
Script to figure out the lemmas in Finnish sentences.
"""
import dataclasses
import json
from typing import Any, TypeGuard
from platformdirs import user_cache_dir
from joblib import Memory # type: ignore
from openai import OpenAI
from uralicNLP import uralicApi # type: ignore
import uralicNLP.tokenizer # type: ignore
import uralicNLP # type: ignore

from config import config

if not uralicApi.is_language_installed(config.language_code): # type: ignore
    uralicApi.download(config.language_code) # type: ignore

memory = Memory(user_cache_dir(config.app_name), verbose=0)

@dataclasses.dataclass
class AmbiguousTokenAnalysis:
    """Class to hold the analysis of a token and its possible analyses."""
    token: str
    analyses: list[str]

    def __str__(self):
        return f"{self.token} = {self.analyses})"

@dataclasses.dataclass
class UnambiguosTokenAnalysis:
    """Class to hold the unambiguous analysis of a token."""
    token: str
    analysis: str

    def __str__(self):
        return f"self.token = {self.analysis}"

@dataclasses.dataclass
class AmbiguousSentenceAnalysis:
    """Class to hold the analysis of a sentence and its possible analyses."""
    sentence: str
    __token_analyses: list[AmbiguousTokenAnalysis]
    is_unambiguous: bool
    def __init__(self, sentence: str, analyses: list[AmbiguousTokenAnalysis] | None = None):
        self.sentence = sentence
        self.__token_analyses: list[AmbiguousTokenAnalysis] = []
        self.is_unambiguous = True # The empty list is unambiguous by default
        if analyses is not None:
            for analysis in analyses:
                self.add_analysis(analysis)
    
    def add_analysis(self, token_analysis: AmbiguousTokenAnalysis):
        """Add an analysis to the sentence."""
        self.__token_analyses.append(token_analysis)
        if len(token_analysis.analyses) > 1:
            self.is_unambiguous = False
    
    def token_analyses(self) -> list[AmbiguousTokenAnalysis]:
        """Return the analyses of the sentence."""
        return self.__token_analyses

    def __str__(self):
        return f"self.sentence = {self.__token_analyses})"
    
    def disambiguate(self) -> list[UnambiguosTokenAnalysis]:
        if self.is_unambiguous:
            # return the first analysis for each token
            return [UnambiguosTokenAnalysis(token_analysis.token, token_analysis.analyses[0]) for token_analysis in self.__token_analyses]
        else:
            return disambiguate_sentence_with_gpt(self.sentence, self.__token_analyses)

class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        # For dataclass instances, convert to dict
        if o is dataclasses.dataclass.__str__:
            return dataclasses.asdict(o)
        # Fallback to default behavior
        return super().default(o)

@dataclasses.dataclass
class UnambiguousSentenceAnalysis:
    """Class to hold the final unambiguous analysis of a sentence."""
    sentence: str
    analyses: list[UnambiguosTokenAnalysis]

    def __repr__(self):
        return f"UnambiguousSentenceAnalysis(sentence={self.sentence}, analyses={self.analyses})"

    def __str__(self):
        return f"""{self.sentence}\n{json.dumps(self.analyses, cls=DataclassJSONEncoder, indent=2)}"""

@memory.cache
def openai_response_create(prompt: str):
    '''Wrapper for the OpenAI response API'''
    client = OpenAI()
    response_create = client.responses.create
    response = response_create(
        model="o3-mini",
        reasoning={"effort": "low"},
        input=prompt,
        text={
        "format": {
            "type": "text"
            }
        },
        store=True,
    )
    return response.output_text

def is_list(val: Any) -> TypeGuard[list[Any]]:
    return isinstance(val, list)

def is_list_of_strings(val: Any) -> TypeGuard[list[str]]:
    return is_list(val) and all(isinstance(item, str) for item in val)

def disambiguate_sentence_with_gpt(sentence: str, ambiguous_analysis: list[AmbiguousTokenAnalysis]) -> list[UnambiguosTokenAnalysis]:
    prompt = f"""Here is a {config.language_name} sentence:

"{sentence}"

Here are the morphological analyses of each word:
{"\n".join(str(analysis) for analysis in ambiguous_analysis)}

Select the most contextually correct analysis for each word.
Return an array of the chosen analyses in JSON format without Markdown formatting."""
    json_response = openai_response_create(prompt)
    disambiguated_analysis: Any = json.loads(json_response)
    if not is_list_of_strings(disambiguated_analysis):
        raise ValueError("Invalid response format: expected a list.")
    if len(disambiguated_analysis) != len(ambiguous_analysis):
        raise ValueError("Invalid response format: length mismatch between response and input.")
    return [
        UnambiguosTokenAnalysis(token_analysis.token, analysis)
        for token_analysis, analysis in zip(ambiguous_analysis, disambiguated_analysis)
    ]

@memory.cache
def analyze_token(token: str) -> list[tuple[str, float]]:
    '''Wrapper for the uralicNLP analyze function'''
    return uralicApi.analyze(token, config.language_code) # type: ignore

def uralicnlp_words(token: str) -> list[str]:
    '''Wrapper for the uralicNLP words function'''
    return uralicNLP.tokenizer.words(token) # type: ignore

def analyze_sentence(sentence: str) -> UnambiguousSentenceAnalysis:
    analysis: AmbiguousSentenceAnalysis = AmbiguousSentenceAnalysis(sentence, [])
    for token in uralicnlp_words(sentence):
        analysis.add_analysis(
            AmbiguousTokenAnalysis(token, list(analysis[0] for analysis in analyze_token(token)))
            )
    return UnambiguousSentenceAnalysis(sentence, analysis.disambiguate())

def uralicnlp_sentences(paragraph: str) -> list[str]:
    '''Wrapper for the uralicNLP sentences function'''
    return uralicNLP.tokenizer.sentences(paragraph) # type: ignore

def analyze_paragraph(paragraph: str) -> list[UnambiguousSentenceAnalysis]:
    sentences = uralicnlp_sentences(paragraph)
    analysis: list[UnambiguousSentenceAnalysis] = []
    for sentence in sentences:
        analysis.append(analyze_sentence(sentence))
    return analysis

def main():
    sentences = analyze_paragraph("Auto on punainen; punainen on kaunis väri.")
    for sentence in sentences:
        print(sentence.sentence)
        for analysis in sentence.analyses:
            print(f"  {analysis.token} = {analysis.analysis}")

if __name__ == "__main__":
    main()
