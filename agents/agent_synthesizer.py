import json
from typing import List, Dict, Any, Tuple, Optional
from pydantic import ValidationError

from core.data_models import Paper, Extraction, Review, WeightedBreakdown
from core.config_loader import Config
from core.llm_wrapper import call_llm
from utilities.helpers import get_weights_table

def calculate_recommendation(
    extractions: List[Extraction],
    config: Config
) -> Tuple[str, str, float, Dict[str, WeightedBreakdown]]:
    """
    Calculate final score and preliminary recommendation.
    """
    weighted_scores = {}
    total_score = 0.0
    
    for ext in extractions:
        criterion = config.get_criterion_by_id(ext.criterion_id)
        if criterion:
            scale = criterion.get('scale', {})
            scale_labels = scale.get('labels', {})
            max_score = len(scale_labels)
            
            if max_score > 0:
                weight = criterion.get('weight', 0)
                # --- FIX: Scale max_score is (range[1] - range[0] + 1) or len(labels)
                # The labels are 1-indexed, but the list is 0-indexed.
                # A 1-5 scale has 5 labels. ext.score is 1, 2, 3, 4, or 5.
                # max_score should be 5.
                # Let's assume the score from LLM is 1-5, matching the labels.
                # Your `criteria.yaml` has labels 1-5, so len(scale_labels) is 5.
                # If ext.score is 5, 5/5 * weight = weight. This seems correct.
                normalized_score = (ext.score / max_score) * weight
                total_score += normalized_score
                
                weighted_scores[ext.criterion_id] = WeightedBreakdown(
                    score=ext.score,
                    weight=weight,
                    weighted_score=normalized_score
                )
            else:
                print(f"[Warning] Criterion {criterion['id']} has no scale labels. Cannot calculate score.")

    total_weight = sum(c.get('weight', 0) for c in config.get_criteria())
    final_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
    
    # --- REPLACED LOGIC ---
    # Get the sorted thresholds from the config
    thresholds = config.get_recommendation_thresholds()
    
    # Default to the last (lowest) threshold's label
    rec = thresholds[-1]['label'] if thresholds else "Reject"
    
    # Iterate from highest to lowest
    for item in thresholds:
        if final_score >= item['threshold']:
            rec = item['label']
            break # Stop at the first match
    # --- END OF REPLACED LOGIC ---
        
    return rec, f"Overall score: {final_score:.1f}", final_score, weighted_scores


def synthesize_review(
    paper: Paper,
    extractions: List[Extraction],
    config: Config
) -> Optional[Review]:
    """
    Agent 2: Synthesize extractions into a final review.
    """
    print(f"[Agent 2] Synthesizing review for: {paper.filename}")
    
    if not extractions:
        print(f"[Agent 2 Error] No extractions provided for {paper.filename}.")
        return None

    llm_config = config.get_llm_config()
    prompt_template = config.get_prompt("synthesizer_user")
    system_prompt = config.get_prompt("synthesizer_system")
    
    recommendation, rationale, score, breakdown = calculate_recommendation(extractions, config)
    
    extractions_list = [e.model_dump(mode='json') for e in extractions]
    extractions_json = json.dumps(extractions_list, indent=2)
    weights_table = get_weights_table(config.get_criteria())
    
    prompt = prompt_template.format(
        paper_title=paper.metadata.title,
        paper_abstract=paper.metadata.abstract,
        json_dump_of_extractions=extractions_json,
        weights_table=weights_table,
        calculated_score=f"{score:.1f}",
        calculated_recommendation=f"{recommendation} (Rationale: {rationale})"
    )
    
    response = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        provider=llm_config['synthesizer_provider'],
        model=llm_config['synthesizer_model'],
        temperature=llm_config['temperature'],
        max_retries=llm_config['max_retries'],
        role="synthesis"
    )
    
    if not response['success']:
        print(f"[Agent 2 Error] LLM call failed for {paper.filename}: {response['error']}")
        return None
        
    json_text = ""
    try:
        raw_content = response['content']
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}')
        
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise json.JSONDecodeError("Could not find JSON object markers '{}' in response.", raw_content, 0)
            
        json_text = raw_content[start_index : end_index + 1]
        review_data = json.loads(json_text)
        
        total_extraction_cost = sum(e.cost for e in extractions)
        total_cost = total_extraction_cost + response['cost']
        
        if 'criterion_narrative' not in review_data or not isinstance(review_data['criterion_narrative'], dict):
             print(f"[Agent 2 Warning] 'criterion_narrative' not found or not a dict. Setting to empty.")
             review_data['criterion_narrative'] = {}

        extractor_model_name = extractions[0].model_used if extractions else "unknown_extractor"
        synthesizer_model_name = f"{llm_config['synthesizer_provider']}/{llm_config['synthesizer_model']}"

        review = Review(
            paper_id=paper.id,
            paper_title=paper.metadata.title, 
            
            # --- MODIFIED: Pass the filename ---
            paper_filename=paper.filename,
            # --- (End of modification) ---

            overall_score=score,
            weighted_breakdown=breakdown,
            synthesizer_model_used=synthesizer_model_name,
            extractor_model_used=extractor_model_name,
            total_cost=total_cost,
            **review_data
        )
        return review
        
    except json.JSONDecodeError as e:
        print(f"[Agent 2 Error] Failed to parse JSON for {paper.filename}: {e}")
        print(f"Raw LLM Output (full): {response['content']}")
        if json_text: print(f"Attempted to parse: {json_text}")
        return None
    except ValidationError as e:
        print(f"[Agent 2 Error] Pydantic validation failed for {paper.filename}: {e}")
        print(f"Raw LLM Output (full): {response['content']}")
        if json_text: print(f"Attempted to parse: {json_text}")
        return None