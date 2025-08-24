from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


class geminiModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Gemini Model"


# Replace these with real values
custom_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    verbose=True,
    temperature=0.1,
    max_tokens=8192,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
gemini_model_invoke = geminiModel(model=custom_model)

def get_AnswerRelevancy_score(actual_ouput, actual_input):
    actual_output = actual_ouput

    metric = AnswerRelevancyMetric(
        threshold=0.8, model=gemini_model_invoke, include_reason=True
    )
    test_case = LLMTestCase(input=actual_input, actual_output=actual_output)

    evaluate(test_cases=[test_case], metrics=[metric])
    metric.measure(test_case)
    return metric.score


def get_contextRelevancy(actual_input_res, actual_output_res, context):
    actual_output = actual_output_res
    retrieval_context = [
        context
    ]
    metric = ContextualRelevancyMetric(
        threshold=0.7, model=gemini_model_invoke, include_reason=True
    )
    test_case = LLMTestCase(
        input=actual_input_res,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )
    evaluate(test_cases=[test_case], metrics=[metric])
    metric.measure(test_case)
    print(metric.score)
    return metric.score

def get_full_scores(actual_ouput,actual_input,context):
    score_answerRelevancy=get_AnswerRelevancy_score(actual_ouput,actual_input)
    score_contextrelevancy=get_contextRelevancy(actual_input,actual_ouput,context)
    return [score_answerRelevancy,score_contextrelevancy]

