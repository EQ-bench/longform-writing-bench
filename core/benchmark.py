# core/benchmark.py
import os
import re
import uuid
import time
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback
import dotenv

from utils.file_io import load_json_file, update_run_data, save_json_file
from utils.api import APIClient
from core.conversation import LongformCreativeTask, TOTAL_STEPS
from core.scoring import (
    aggregate_longform_scores,
    bootstrap_benchmark_stability_longform
)

dotenv.load_dotenv()

# --- Constants ---
RUNS_FILENAME = "longform_bench_runs.json"
#PROMPTS_FILENAME = "longform_creative_writing_prompts.json"
PROMPTS_FILENAME = "longform_creative_writing_prompts_minimalist.json"
# Prompt template filenames (relative to data dir)
PROMPT_TEMPLATE_FILES = {i: f"prompt{i}.txt" for i in range(1, TOTAL_STEPS + 1)} # 1 to 13
# Criteria filenames
CRITERIA_CHAPTER_FILE = "longform_creative_writing_criteria_chapter.txt"
NEG_CRITERIA_CHAPTER_FILE = "longform_negative_criteria_chapter.txt"
JUDGE_PROMPT_CHAPTER_FILE = "longform_creative_writing_judging_prompt_chapter.txt"
CRITERIA_FINAL_FILE = "longform_creative_writing_criteria_final.txt"
NEG_CRITERIA_FINAL_FILE = "longform_negative_criteria_final.txt"
JUDGE_PROMPT_FINAL_FILE = "longform_creative_writing_judging_prompt_final.txt"

NUM_CHAPTERS = int(os.getenv("NUM_CHAPTERS", 8))
NUM_PLANNING_STEPS = 5
TOTAL_STEPS = NUM_PLANNING_STEPS + NUM_CHAPTERS
FIRST_CHAPTER_STEP_INDEX = NUM_PLANNING_STEPS + 1 # Step number 6
LAST_CHAPTER_STEP_INDEX = TOTAL_STEPS
logging.info(f"Configured for {NUM_CHAPTERS} chapters (Total steps: {TOTAL_STEPS}).")

# --- Helper Functions ---

def load_text_file(filepath: str, data_dir: str) -> Optional[str]:
    """Loads text content from a file within the data directory."""
    full_path = os.path.join(data_dir, filepath)
    if not os.path.exists(full_path):
        logging.error(f"Required data file not found: {full_path}")
        return None
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.exception(f"Error reading file {full_path}: {e}")
        return None

def load_list_from_file(filepath: str, data_dir: str) -> List[str]:
    """Loads a list of strings from a file, one item per line."""
    content = load_text_file(filepath, data_dir)
    if content is None:
        return []
    return [line.strip() for line in content.splitlines() if line.strip()]

def sanitize_model_name(name: str) -> str:
    """Removes characters unsuitable for filenames/keys."""
    # Keep alphanumeric, underscore, hyphen, slash (for model orgs)
    sanitized = re.sub(r'[^\w\-\/]+', '_', name)
    # Replace multiple consecutive underscores/hyphens with a single one
    sanitized = re.sub(r'[_]{2,}', '_', sanitized)
    sanitized = re.sub(r'[-]{2,}', '-', sanitized)
    # Remove leading/trailing underscores/hyphens
    sanitized = sanitized.strip('_-')
    return sanitized

def compute_benchmark_results_longform(
    runs: Dict[str, Any],
    run_key: str,
    runs_file: str,
    negative_criteria_chapter: List[str],
    negative_criteria_final: List[str]
):
    """
    Computes and saves the final benchmark score and bootstrap analysis for a run.
    """
    run_data = runs.get(run_key, {})
    if not run_data:
        logging.error(f"Run key {run_key} not found in runs data. Cannot compute results.")
        return

    longform_tasks_by_iter = run_data.get("longform_tasks", {})
    if not longform_tasks_by_iter:
        logging.warning(f"No 'longform_tasks' found for run {run_key}. No results computed.")
        return

    # Collect all completed task data across iterations
    completed_tasks_data = []
    for iter_str, prompts_dict in longform_tasks_by_iter.items():
        for prompt_id, task_dict in prompts_dict.items():
            if task_dict.get("status") == "completed":
                completed_tasks_data.append(task_dict)
            # Log tasks that didn't complete successfully
            elif task_dict.get("status") == "error":
                 logging.warning(f"Task {prompt_id} (iter {iter_str}) ended with status 'error': {task_dict.get('error_message')}")
            else:
                 logging.warning(f"Task {prompt_id} (iter {iter_str}) has status '{task_dict.get('status')}' and was not included in final scoring.")


    if not completed_tasks_data:
        logging.error(f"No successfully completed tasks found for run {run_key}. Cannot compute final score.")
        # Update run status to reflect this?
        update_run_data(runs_file, run_key, {"status": "completed_no_tasks_scored", "end_time": datetime.now().isoformat()})
        return

    logging.info(f"Calculating final results for run {run_key} based on {len(completed_tasks_data)} completed tasks.")

    # 1. Aggregate Scores
    agg_results = aggregate_longform_scores(completed_tasks_data, negative_criteria_chapter, negative_criteria_final)
    if "error" in agg_results:
        logging.error(f"Failed to aggregate scores: {agg_results['error']}")
        # Save error state?
        update_run_data(runs_file, run_key, {"status": "completed_scoring_error", "results": {"benchmark_results": agg_results}, "end_time": datetime.now().isoformat()})
        return

    logging.info(f"Aggregate Score (0-20): {agg_results['overall_score_0_20']}, EQBench Score (0-100): {agg_results['eqbench_longform_score_0_100']}")
    logging.info(f"Scored {agg_results['num_tasks_scored']} out of {agg_results['num_tasks_total']} potential tasks.")

    # 2. Bootstrap Analysis
    bootstrap_stats = bootstrap_benchmark_stability_longform(completed_tasks_data, negative_criteria_chapter, negative_criteria_final)
    if "error" in bootstrap_stats:
        logging.error(f"Bootstrap analysis failed: {bootstrap_stats['error']}")
    else:
        logging.info(f"Bootstrap 95% CI (0-20 scale): ({bootstrap_stats['ci_lower']:.2f}, {bootstrap_stats['ci_upper']:.2f}), Mean: {bootstrap_stats['bootstrap_mean']:.2f}")


    # 3. Store Results in Run Data
    # Ensure 'results' and 'benchmark_results' keys exist
    results_dict = run_data.get("results", {})
    benchmark_results_dict = results_dict.get("benchmark_results", {})

    # Update benchmark results with aggregation and bootstrap stats
    benchmark_results_dict.update(agg_results) # Add overall scores, counts
    benchmark_results_dict["bootstrap_analysis"] = bootstrap_stats # Add bootstrap results

    results_dict["benchmark_results"] = benchmark_results_dict

    # Save the updated results back to the run file
    if not update_run_data(runs_file, run_key, {"results": results_dict}):
         logging.error(f"Failed to save final benchmark results for run {run_key} to {runs_file}")

def load_prompt_templates(data_dir):
    # Load prompt templates dynamically
    prompt_templates = {}
    required_files_found = True

    # Load planning prompts (1-5)
    for i in range(1, NUM_PLANNING_STEPS + 1):
        filename = f"prompt{i}.txt"
        template_content = load_text_file(filename, data_dir)
        if template_content is None:
            required_files_found = False
            # Error logged within load_text_file
        else:
            prompt_templates[i] = template_content

    # Load specific chapter prompts if NUM_CHAPTERS > 0
    if NUM_CHAPTERS > 0:
        # First Chapter Prompt (Step FIRST_CHAPTER_STEP_INDEX)
        first_chap_file = "prompt_chapter_first.txt"
        first_chap_content = load_text_file(first_chap_file, data_dir)
        if first_chap_content is None:
            required_files_found = False
        else:
            prompt_templates[FIRST_CHAPTER_STEP_INDEX] = first_chap_content

        # Intermediate Chapter Prompt Template (Used for steps > FIRST_CHAPTER_STEP_INDEX and < LAST_CHAPTER_STEP_INDEX)
        intermediate_chap_file = "prompt_chapter_intermediate.txt"
        intermediate_chap_template = load_text_file(intermediate_chap_file, data_dir)
        if intermediate_chap_template is None and NUM_CHAPTERS > 2: # Only required if there are intermediate chapters
            logging.error(f"Intermediate chapter template file missing: {os.path.join(data_dir, intermediate_chap_file)}")
            required_files_found = False

        # Last Chapter Prompt (Step LAST_CHAPTER_STEP_INDEX)
        # Note: If NUM_CHAPTERS is 1, this is the same step as the first chapter. The first chapter prompt takes precedence.
        if NUM_CHAPTERS >= 1: # Need a last chapter prompt if there's at least one chapter
            last_chap_file = "prompt_chapter_last.txt"
            last_chap_template = load_text_file(last_chap_file, data_dir) # Load as template first
            if last_chap_template is None:
                required_files_found = False
            else:
                # Format the last chapter prompt immediately
                last_chapter_number = NUM_CHAPTERS
                try:
                    prompt_templates[LAST_CHAPTER_STEP_INDEX] = last_chap_template.format(chapter_number=last_chapter_number)
                except KeyError:
                    logging.error(f"Placeholder '{{chapter_number}}' missing in {last_chap_file}. Cannot format last chapter prompt.")
                    required_files_found = False


        # Populate intermediate chapter prompts (if any)
        if intermediate_chap_template is not None:
            # Loop from the second chapter up to the second-to-last chapter
            for step_num in range(FIRST_CHAPTER_STEP_INDEX + 1, LAST_CHAPTER_STEP_INDEX):
                chapter_number = step_num - NUM_PLANNING_STEPS
                try:
                    prompt_templates[step_num] = intermediate_chap_template.format(chapter_number=chapter_number)
                except KeyError:
                    logging.error(f"Placeholder '{{chapter_number}}' missing in {intermediate_chap_file}. Cannot format intermediate prompts.")
                    required_files_found = False
                    break # Stop trying to format intermediates

    # Check if all required files were loaded successfully
    if not required_files_found:
        raise FileNotFoundError("One or more required prompt template files were missing or invalid. Check logs.")

    logging.info(f"Loaded {len(prompt_templates)} prompt templates for {TOTAL_STEPS} steps.")
    return prompt_templates

# --- Main Benchmark Function ---

def run_longform_bench(
    test_model: str,
    judge_model: str,
    runs_file: str,
    data_dir: str = "data",
    num_threads: int = 4,
    run_id: Optional[str] = None,
    iterations: int = 1,
    redo_judging: bool = False, # If true, re-runs judging steps
    save_interval: int = 1, # How many generation steps before saving task state
    skip_generation: bool = False, # If true, skips generation (useful for re-judging)
    skip_chapter_judging: bool = False, # If true, skips chapter judging
    skip_final_judging: bool = False, # If true, skips final judging
) -> str:
    """
    Main function to orchestrate the long-form creative writing benchmark.
    """
    logging.info("--- Starting Long-Form Creative Writing Benchmark ---")
    logging.info(f"Test Model: {test_model}")
    logging.info(f"Judge Model: {judge_model}")
    logging.info(f"Iterations: {iterations}")
    logging.info(f"Threads: {num_threads}")
    logging.info(f"Runs File: {runs_file}")
    logging.info(f"Data Dir: {data_dir}")

    # --- Load Data Files ---
    logging.info("Loading data files...")
    # Load initial writing prompts
    prompts_path = os.path.join(data_dir, PROMPTS_FILENAME)    
    writing_prompts_all = load_json_file(prompts_path)
    print(writing_prompts_all)
    if not writing_prompts_all:
        raise FileNotFoundError(f"Initial writing prompts file not found or empty: {prompts_path}")

    # Load prompt templates for steps 1-13
    prompt_templates = load_prompt_templates(data_dir)

    # Load criteria and judge prompts
    criteria_chapter = load_list_from_file(CRITERIA_CHAPTER_FILE, data_dir)
    neg_criteria_chapter = load_list_from_file(NEG_CRITERIA_CHAPTER_FILE, data_dir)
    judge_prompt_chapter_tmpl = load_text_file(JUDGE_PROMPT_CHAPTER_FILE, data_dir)
    criteria_final = load_list_from_file(CRITERIA_FINAL_FILE, data_dir)
    neg_criteria_final = load_list_from_file(NEG_CRITERIA_FINAL_FILE, data_dir)
    judge_prompt_final_tmpl = load_text_file(JUDGE_PROMPT_FINAL_FILE, data_dir)

    # Validate required judge prompts/criteria
    if not judge_prompt_chapter_tmpl or not judge_prompt_final_tmpl:
         raise ValueError("Chapter or Final judging prompt template is missing.")
    if not criteria_chapter or not criteria_final:
         logging.warning("Chapter or Final criteria list is empty. Judging might not be meaningful.")

    logging.info(f"Loaded Chapter Criteria ({len(criteria_chapter)} items, {len(neg_criteria_chapter)} negative)")
    logging.info(f"Loaded Final Criteria ({len(criteria_final)} items, {len(neg_criteria_final)} negative)")

    # --- Initialize Run ---
    sanitized_model = sanitize_model_name(test_model)
    base_id = run_id if run_id else str(uuid.uuid4().hex[:8]) # Shorter UUID
    run_key = f"{base_id}_{sanitized_model}"

    runs = load_json_file(runs_file)
    if run_key not in runs:
        logging.info(f"Creating new run: {run_key}")
        init_dict = {
            "run_id_base": base_id,
            "test_model": test_model,
            "judge_model": judge_model,
            "iterations": iterations,
            "start_time": datetime.now().isoformat(),
            "status": "initializing",
            "data_files": { # Record data files used for this run
                 "prompts": PROMPTS_FILENAME,
                 "criteria_chapter": CRITERIA_CHAPTER_FILE,
                 "neg_criteria_chapter": NEG_CRITERIA_CHAPTER_FILE,
                 "judge_prompt_chapter": JUDGE_PROMPT_CHAPTER_FILE,
                 "criteria_final": CRITERIA_FINAL_FILE,
                 "neg_criteria_final": NEG_CRITERIA_FINAL_FILE,
                 "judge_prompt_final": JUDGE_PROMPT_FINAL_FILE,
            },
            "longform_tasks": {}, # { iter_idx_str: { prompt_id_str: task_dict } }
            "results": {}
        }
        # Use save_json_file for initial creation as well
        runs[run_key] = init_dict
        if not save_json_file(runs, runs_file):
             raise IOError(f"Failed to initialize run file: {runs_file}")
    else:
        logging.info(f"Resuming run: {run_key}")
        # Optionally update iterations if resuming with a different number?
        # runs[run_key]['iterations'] = max(runs[run_key].get('iterations', 1), iterations)
        # update_run_data(runs_file, run_key, {"iterations": runs[run_key]['iterations']})


    # --- Prepare Tasks ---
    tasks_to_process: List[LongformCreativeTask] = []
    run_data = load_json_file(runs_file).get(run_key, {}) # Reload fresh data
    existing_tasks_data = run_data.get("longform_tasks", {})

    for i in range(1, iterations + 1):
        iter_str = str(i)
        iter_tasks_data = existing_tasks_data.get(iter_str, {})

        for prompt_id, writing_prompt_text in writing_prompts_all.items():
            prompt_id_str = str(prompt_id) # Ensure consistent key type
            task_data = iter_tasks_data.get(prompt_id_str)

            if task_data and task_data.get("test_model") == test_model:
                # Resume existing task
                logging.debug(f"Resuming task for prompt {prompt_id_str}, iteration {i}")
                try:
                    task_obj = LongformCreativeTask.from_dict(task_data, prompt_templates)
                    # Handle redo_judging logic
                    if redo_judging:
                        logging.info(f"Resetting judging state for task {prompt_id}, iteration {i} due to --redo-judging flag.")
                        task_obj.chapter_judge_scores = {}
                        task_obj.chapter_raw_judge_text = {}
                        
                        # Here’s the crucial part for final judgments:
                        task_obj.final_judge_scores = []       # Clear them out
                        task_obj.final_raw_judge_texts = []    # Clear them out

                        # Reset status if it was completed:
                        if task_obj.status in ["judging_chapters", "judged_chapters", "judging_final", "completed"]:
                            task_obj.status = "generated"
                        
                        # Then save
                        task_obj._save_state(runs_file, run_key)


                    tasks_to_process.append(task_obj)
                except Exception as e:
                     logging.exception(f"Error loading task state for prompt {prompt_id_str}, iteration {i}. Skipping task. Data: {task_data}")
            else:
                # Create new task
                logging.debug(f"Creating new task for prompt {prompt_id_str}, iteration {i}")
                new_task = LongformCreativeTask(
                    prompt_id=prompt_id_str,
                    writing_prompt=writing_prompt_text,
                    iteration_index=i,
                    test_model=test_model,
                    judge_model=judge_model,
                    prompt_templates=prompt_templates
                )
                tasks_to_process.append(new_task)
                # Save initial state of new task
                new_task._save_state(runs_file, run_key)

    logging.info(f"Prepared {len(tasks_to_process)} tasks across {iterations} iteration(s).")

    # --- Initialize API Clients ---
    # Pass timeout/retry settings from config/env
    api_clients = {
        "test": APIClient(model_type="test"),
        "judge": APIClient(model_type="judge")
    }

    # --- Execute Stages ---
    update_run_data(runs_file, run_key, {"status": "running"})

    # Stage 1: Generation
    if not skip_generation:
        logging.info("--- Stage 1: Generation ---")
        tasks_needing_generation = [t for t in tasks_to_process if t.status in ["initialized", "generating"]]
        if tasks_needing_generation:
            logging.info(f"Running generation for {len(tasks_needing_generation)} tasks...")
            with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="GenWorker") as executor:
                futures = {executor.submit(task.run_generation_sequence, api_clients, runs_file, run_key, save_interval): task for task in tasks_needing_generation}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Stories"):
                    task = futures[future]
                    try:
                        future.result() # Check for exceptions
                    except Exception as exc:
                        logging.error(f"Task {task.prompt_id} (iter {task.iteration_index}) generation failed: {exc}")
                        # Task status should be updated internally to 'error'
        else:
            logging.info("All tasks already generated or in error state. Skipping generation.")
    else:
        logging.info("Skipping generation stage (--skip-generation).")


    # Stage 2: Chapter Judging
    if not skip_chapter_judging:
        logging.info("--- Stage 2: Chapter Judging ---")
        # Reload tasks to get latest status after generation
        run_data_after_gen = load_json_file(runs_file).get(run_key, {})
        tasks_after_gen = []
        for i in range(1, iterations + 1):
             iter_tasks = run_data_after_gen.get("longform_tasks", {}).get(str(i), {})
             for pid, tdata in iter_tasks.items():
                  try:
                       tasks_after_gen.append(LongformCreativeTask.from_dict(tdata, prompt_templates))
                  except Exception as e:
                       logging.exception(f"Error reloading task state for judging (prompt {pid}, iter {i}). Skipping.")

        tasks_needing_chap_judge = [t for t in tasks_after_gen if t.status in ["generated", "judging_chapters"]]
        if tasks_needing_chap_judge:
            logging.info(f"Running chapter judging for {len(tasks_needing_chap_judge)} tasks...")
            with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="ChapJudgeWorker") as executor:
                futures = {executor.submit(task.judge_chapters, api_clients, judge_prompt_chapter_tmpl, criteria_chapter, neg_criteria_chapter, runs_file, run_key): task for task in tasks_needing_chap_judge}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Judging Chapters"):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logging.error(f"Task {task.prompt_id} (iter {task.iteration_index}) chapter judging failed: {exc}")
        else:
            logging.info("No tasks require chapter judging. Skipping.")
    else:
        logging.info("Skipping chapter judging stage (--skip-chapter-judging).")


    # Stage 3: Final Judging
    if not skip_final_judging:
        logging.info("--- Stage 3: Final Piece Judging ---")
        # Reload tasks again
        run_data_after_chap_judge = load_json_file(runs_file).get(run_key, {})
        tasks_after_chap_judge = []
        for i in range(1, iterations + 1):
             iter_tasks = run_data_after_chap_judge.get("longform_tasks", {}).get(str(i), {})
             for pid, tdata in iter_tasks.items():
                  try:
                       tasks_after_chap_judge.append(LongformCreativeTask.from_dict(tdata, prompt_templates))
                  except Exception as e:
                       logging.exception(f"Error reloading task state for final judging (prompt {pid}, iter {i}). Skipping.")

        tasks_needing_final_judge = [t for t in tasks_after_chap_judge if t.status in ["judged_chapters", "judging_final"]]
        if tasks_needing_final_judge:
            logging.info(f"Running final piece judging for {len(tasks_needing_final_judge)} tasks...")
            with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="FinalJudgeWorker") as executor:
                futures = {executor.submit(task.judge_final_piece, api_clients, judge_prompt_final_tmpl, criteria_final, neg_criteria_final, runs_file, run_key): task for task in tasks_needing_final_judge}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Judging Final Piece"):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logging.error(f"Task {task.prompt_id} (iter {task.iteration_index}) final judging failed: {exc}")
        else:
            logging.info("No tasks require final judging. Skipping.")
    else:
        logging.info("Skipping final judging stage (--skip-final-judging).")


    # --- Stage 4: Compute Final Results ---
    logging.info("--- Stage 4: Computing Final Results ---")
    try:
        # Reload the absolute latest run data
        runs_final = load_json_file(runs_file)
        compute_benchmark_results_longform(
            runs_final,
            run_key,
            runs_file,
            neg_criteria_chapter,
            neg_criteria_final
        )
    except Exception as e:
         logging.exception(f"Failed to compute final benchmark results for run {run_key}: {e}")
         update_run_data(runs_file, run_key, {"status": "completed_result_error", "end_time": datetime.now().isoformat()})
         return run_key # Return run key even if results failed


    # --- Mark Run as Completed ---
    logging.info(f"Benchmark run {run_key} finished.")
    update_run_data(
        runs_file,
        run_key,
        {
            "status": "completed",
            "end_time": datetime.now().isoformat()
        }
    )

    # --- Final Output ---
    final_run_data = load_json_file(runs_file).get(run_key, {})
    final_results = final_run_data.get("results", {}).get("benchmark_results", {})
    final_score_100 = final_results.get('eqbench_longform_score_0_100', 'N/A')
    ci_lower = final_results.get('bootstrap_analysis', {}).get('ci_lower', 'N/A')
    ci_upper = final_results.get('bootstrap_analysis', {}).get('ci_upper', 'N/A')

    print("\n--- Benchmark Summary ---")
    print(f"Run Key: {run_key}")
    print(f"Test Model: {test_model}")
    print(f"Longform Writing Score: {final_score_100}")
    if ci_lower != 'N/A':
        print(f"95% CI: ({ci_lower * 5:.2f}, {ci_upper * 5:.2f})")
    print(f"Results saved in: {runs_file}")
    print("------------------------")


    return run_key