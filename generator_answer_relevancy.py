from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

rag_input =  "What is boosting in machine learning and how does it work?"

actual_ouput = """Boosting is an ensemble technique that combines multiple weak learners to form a strong learner. 
It trains models sequentially, where each new model focuses on correcting the errors made by the previous ones. 
AdaBoost was one of the first popular boosting algorithms. Random Forests, on the other hand, build trees independently and 
average their results. Gradient Boosting uses gradient descent to minimize a loss function when adding new learners."""

test_case=LLMTestCase(

    input=rag_input,
    actual_output=actual_ouput
)

answer_relevancy=AnswerRelevancyMetric(verbose_mode=True)

answer_relevancy.measure(test_case)