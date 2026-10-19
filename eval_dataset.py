from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
from rag_workflow import graph

ROOT_DIR=Path(__file__).parent

GOLDEN_PATH=ROOT_DIR/"datasets"/"goldens"/"golden_dataset.json"

EVALATION_DATASET_DIR=GOLDEN_PATH.parent.parent/"evaluation_dataset"

#create the directory
EVALATION_DATASET_DIR.mkdir(exist_ok=True,parents=True)

#Generate evaluation dataset
dataset=EvaluationDataset()

#add goldens into evaluation dataset
dataset.add_goldens_from_json_file(

    file_path=GOLDEN_PATH
)

#Invoke our RAG application
for golden in dataset.goldens:
    final_state=graph.invoke({"query":golden.input})
    sleep(3)

    test_case= LLMTestCase(

        input=golden.input,
        actual_output=final_state.get("response"),
        expected_output=golden.expected_output,
        retrieval_context=[doc.page_content for doc in final_state.get("retrieved_doc")]
    )

    dataset.add_test_case(test_case=test_case)

#Save the dataset
    
dataset.save_as(

    file_type="json",
    directory="EVALATION_DATASET_DIR",
    file_name="evaluation_dataset",
    include_test_cases=True
    )
