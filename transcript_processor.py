import re
from pathlib import Path

timestamp_pattern= re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}$")

def transcript_text_cleaner(transcript: str)-> str:
    
    "Remove the timestamp lines from transcript and only focus on text"
    
    lines=transcript.splitlines()
    cleaned_lines=[]
    
    for line in lines:
        stripped_lines=line.strip()
        if not stripped_lines:
            continue
        if timestamp_pattern.match(stripped_lines):
            continue
        cleaned_lines.append(stripped_lines)
        
    return "\n".join(cleaned_lines)

def load_and_clean_transcript(dir_path: Path|str, new_dir: Path|str):
    
    ORIGINAL_DIR=Path(dir_path)
    NEW_DIR=Path(new_dir)
    
    #if not NEW_DIR:
    NEW_DIR.mkdir(parents=True,exist_ok=True)
        
    #Load all transcripts
    transcripts=ORIGINAL_DIR.glob(pattern='*txt')
    
    for transcript_path in transcripts:
        #Read the transcript
        content=transcript_path.read_text(encoding="utf-8")
        
        #Clean the transcripts
        cleaned_content=transcript_text_cleaner(content)
        
        #Get the filename
        filename=transcript_path.name
        
        #Create the new path
        output_path= NEW_DIR/filename
        
        #Write the content
        output_path.write_text(data=cleaned_content, encoding="utf-8")
        
if __name__=="__main__":
    
    load_and_clean_transcript(Path("./data/raw"), Path("./data/cleaned"))