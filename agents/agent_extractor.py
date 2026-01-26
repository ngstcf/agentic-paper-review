import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from core.data_models import Paper, Extraction
from core.config_loader import Config
from core.llm_wrapper import call_llm
from utilities.helpers import get_scale_definition

def build_extraction_prompt(
    paper_content: str,
    criterion: Dict[str, Any],
    domain: str,
    prompt_template: str
) -> str:
    """Builds the user prompt for Agent 1."""
    
    # Truncate paper content if it's too large to fit in the prompt
    max_content_tokens = 100000  # ~100k tokens
    if len(paper_content) > max_content_tokens * 4:
        print(f"[Warning] Truncating paper content for {criterion['id']}")
        paper_content = paper_content[:max_content_tokens * 4]

    return prompt_template.format(
        paper_markdown=paper_content,
        criterion_name=criterion['name'],
        criterion_description=criterion['description'],
        sub_questions="\n".join(f"- {q}" for q in criterion.get('sub_questions', [])),
        scale_definition=get_scale_definition(criterion.get('scale', {})),
        domain=domain
    )

def extract_criterion_evidence(
    paper: Paper,
    criterion: Dict[str, Any],
    config: Config
) -> Optional[Extraction]:
    """
    Agent 1: Extract evidence for a single criterion.
    """
    llm_config = config.get_llm_config()
    prompt_template = config.get_prompt("extractor_user")
    system_prompt = config.get_prompt("extractor_system").format(domain=config.domain)
    
    prompt = build_extraction_prompt(
        paper_content=paper.content_markdown,
        criterion=criterion,
        domain=config.domain,
        prompt_template=prompt_template
    )
    
    response = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        provider=llm_config['extractor_provider'],
        model=llm_config['extractor_model'],
        temperature=llm_config['temperature'],
        max_retries=llm_config['max_retries']
    )
    
    if not response['success']:
        print(f"[Agent 1 Error] LLM call failed for {paper.filename} on {criterion['id']}: {response['error']}")
        return None
        
    json_text = "" # Initialize for error logging
    try:
        raw_content = response['content']
        
        # --- NEW ROBUST JSON PARSING ---
        # Find the first '{' and the last '}'
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}')
        
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise json.JSONDecodeError("Could not find JSON object markers '{}' in response.", raw_content, 0)
            
        json_text = raw_content[start_index : end_index + 1]
        # --- END NEW LOGIC ---
        
        extraction_data = json.loads(json_text)
        
        # Validate and create Extraction object
        extraction = Extraction(
            paper_id=paper.id,
            criterion_id=criterion['id'],
            model_used=f"{llm_config['extractor_provider']}/{llm_config['extractor_model']}",
            cost=response['cost'],
            **extraction_data
        )
        return extraction
        
    except json.JSONDecodeError as e:
        print(f"[Agent 1 Error] Failed to parse JSON for {paper.filename} on {criterion['id']}: {e}")
        print(f"Raw LLM Output (full): {response['content']}")
        if json_text:
             print(f"Attempted to parse: {json_text}")
        return None
    except ValidationError as e:
        print(f"[Agent 1 Error] Pydantic validation failed for {paper.filename} on {criterion['id']}: {e}")
        print(f"Raw LLM Output (full): {response['content']}")
        if json_text:
             print(f"Attempted to parse: {json_text}")
        return None

def process_paper_extractions(
    paper: Paper,
    config: Config
) -> List[Extraction]:
    """
    Run Agent 1 on all criteria for a single paper, in parallel.
    """
    criteria = config.get_criteria()
    llm_config = config.get_llm_config()
    extractions = []
    
    print(f"[Agent 1] Starting parallel extraction for: {paper.filename}")
    
    with ThreadPoolExecutor(max_workers=llm_config['max_parallel']) as executor:
        futures = {
            executor.submit(extract_criterion_evidence, paper, c, config): c
            for c in criteria
        }
        
        for future in as_completed(futures):
            criterion = futures[future]
            try:
                result = future.result()
                if result:
                    extractions.append(result)
                    print(f"  > Completed criterion: {criterion['id']}")
                else:
                    print(f"  > FAILED criterion: {criterion['id']}")
            except Exception as e:
                print(f"[Agent 1 Error] Thread failed for {criterion['id']}: {e}")

    return sorted(extractions, key=lambda e: e.criterion_id)