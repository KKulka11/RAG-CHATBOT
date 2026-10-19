from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig, ContextConstructionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv
from pathlib import Path

#Load openai key

load_dotenv()

#Define paths

ROOT_PATH=Path(__file__).parent

DOC_PATH=ROOT_PATH/"data"/"cleaned"

DATASET_PATH=ROOT_PATH/"datasets"

DATASET_PATH.mkdir(exist_ok=True)
#Filtration config

filtration_config = FiltrationConfig(
  critic_model="gpt-4o-mini",
  synthetic_input_quality_threshold=0.6,
  max_quality_retries=2,
  
)

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.COMPARATIVE: 0.2,
        Evolution.MULTICONTEXT: 0.1,
        Evolution.CONCRETIZING: 0.3,
        Evolution.CONSTRAINED: 0.4
    },
    num_evolutions=2
)

#Create syntheseizer
synthesizer=Synthesizer(
    
    model="gpt-4o-mini",
    filtration_config=filtration_config,
    evolution_config=evolution_config
    
)

context_config=ContextConstructionConfig(

    critic_model="gpt-4o-mini",
    max_contexts_per_document=2,
    context_quality_threshold=0.7,
    max_retries=2


)

def get_doc_path(directory= Path | str)->list[str]:

    directory=Path(directory)

    if directory.exists() and directory.is_dir():
        paths=directory.glob("*.txt")
    
    return [path.as_posix() for path in paths]

#Generate goldens from docs

synthesizer.generate_goldens_from_docs(
  document_paths=get_doc_path(directory=DOC_PATH),
  context_construction_config=ContextConstructionConfig()
)

#save the goldens
GOLDENS_DIR=DATASET_PATH/"goldens"
GOLDENS_DIR.mkdir(exists_ok=True,parens=True)

syntheseizer.save_as(

    file_type=json,
    directory= GOLDENS_DIR.as_posix(),
    file_name="golden_dataset"
)