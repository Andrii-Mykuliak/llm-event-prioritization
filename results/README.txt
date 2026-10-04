Your LLM runs. Files of one run share the prefix <llm-name>__ and are written by scripts/run_llm.py:
  <llm-name>__main_experiment_responses.jsonl, <llm-name>__controlled_timing_rank_followup_responses.jsonl, <llm-name>__run_meta.json, <llm-name>__errors.jsonl
--limit runs use the prefix <llm-name>__smoke__ and are ignored by the analysis. Reference responses are in ../reference_results/.
