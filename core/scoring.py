# core/scoring.py
import logging
import re
import statistics as stats
import random
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional
import json

# Constants
SCORE_RANGE_MIN = 0
SCORE_RANGE_MAX = 20
NUM_CHAPTERS = int(os.getenv("NUM_CHAPTERS", 8))
FINAL_SCORE_WEIGHT = 1 # float(os.getenv("FINAL_SCORE_WEIGHT", 1.0))
# Global dictionary for criteria weights (lowercase keys for case-insensitive matching)
with open('data/criteria_weights.json', 'r') as f:
    CRITERIA_WEIGHTS = json.load(f)

# Default weight for criteria not in the dictionary
DEFAULT_WEIGHT = 1.0

def parse_judge_scores_longform(judge_model_response: str) -> Dict[str, float]:
    """
    Parses scores from judge response text. Expects "Metric Name: Score" format.
    Normalizes scores to a 0-20 range based on SCORE_RANGE_MIN/MAX if needed,
    but currently assumes judge provides scores directly in the 0-20 range.
    """
    scores = {}
    # More robust regex to handle variations like "Metric Name: [Score]", "Metric Name: Score Value"
    # It captures the metric name (non-greedy) and the score (float or int)
    score_pattern = r'^\s*([^:]+?)\s*:\s*\[?(-?\d+(?:\.\d+)?)\]?' # Matches start of line, metric, colon, optional brackets, score
    # Alternative pattern if scores are not at the start of the line
    score_pattern_inline = r'([^:]+?)\s*:\s*\[?(-?\d+(?:\.\d+)?)\]?'

    lines = judge_model_response.splitlines()
    in_scores_section = False # Flag if response has specific score section markers

    # Simple check for common markers - adjust if judge format is known
    if "[Scores]" in judge_model_response or "--- Scores ---" in judge_model_response:
        in_scores_section = True # Assume we only parse within marked sections if markers exist

    collected_lines = []
    temp_collect = False
    for line in lines:
        line_strip = line.strip()
        if not line_strip: continue # Skip empty lines

        # Logic to handle score sections if markers are used
        if "[Scores]" in line_strip or "--- Scores ---" in line_strip:
            temp_collect = True
            continue
        if ("---" in line_strip and temp_collect) or ("[End Scores]" in line_strip): # End markers
             temp_collect = False
             continue

        if in_scores_section and temp_collect:
            collected_lines.append(line_strip)
        elif not in_scores_section: # If no markers, parse all lines
             collected_lines.append(line_strip)

    # Parse the collected lines
    for line in collected_lines:
        match = re.match(score_pattern, line)
        if not match: # Try inline pattern if start-of-line fails
             match = re.search(score_pattern_inline, line)

        if match:
            metric_name = match.group(1).strip()
            # Basic filtering of common non-metric lines
            if metric_name.lower() in ["overall assessment", "summary", "reasoning", "critique", "feedback", "notes"]:
                 continue
            try:
                score = float(match.group(2))
                # Clamp score to expected range (0-20)
                clamped_score = max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
                if clamped_score != score:
                     logging.warning(f"Score for '{metric_name}' ({score}) was outside range [{SCORE_RANGE_MIN}-{SCORE_RANGE_MAX}]. Clamped to {clamped_score}.")
                scores[metric_name] = clamped_score
            except ValueError:
                logging.warning(f"Could not parse score value for metric '{metric_name}' from line: {line}")
            except IndexError:
                 logging.warning(f"Regex pattern failed to capture groups correctly for line: {line}")

    if not scores:
         logging.warning(f"Could not parse any scores from the judge response. Response:\n{judge_model_response}...")

    # logging.debug(f"Parsed scores: {scores}")
    return scores


def invert_if_negative(metric: str, score: float, negative_criteria: List[str]) -> float:
    """
    If metric is a negative criterion, invert score on the 0-20 scale.
    e.g., 20 (bad) -> 0 (good), 0 (good) -> 20 (bad). Formula: new = MAX - old.
    """
    # Normalize metric name for comparison (lowercase, strip whitespace)
    normalized_metric = metric.lower().strip()
    normalized_neg_criteria = [nc.lower().strip() for nc in negative_criteria]

    if normalized_metric in normalized_neg_criteria:
        inverted_score = SCORE_RANGE_MAX - score
        # logging.debug(f"Inverting negative metric '{metric}': {score} -> {inverted_score}")
        return inverted_score
    return score


def calculate_task_score(
    task_data: Dict[str, Any],
    negative_criteria_chapter: List[str],
    negative_criteria_final: List[str]
) -> Optional[float]:
    """
    Calculates the weighted average score for a single completed LongformCreativeTask,
    factoring in multiple final judgments if present.
    Returns the final combined score in [0..20] or None if scoring is not possible.
    """

    chapter_scores_raw = task_data.get("chapter_judge_scores", {})
    final_scores_list = task_data.get("final_judge_scores", [])  # Now a list of dicts

    # --- Calculate Average Chapter Score ---
    valid_chapter_scores = []
    for chap_num_str, chap_scores_dict in chapter_scores_raw.items():
        if isinstance(chap_scores_dict, dict) and chap_scores_dict:
            # Weighted sum for each chapter
            chapter_weighted_sum = 0.0
            total_weight = 0.0
            for metric, value in chap_scores_dict.items():
                if isinstance(value, (int, float)):
                    from core.scoring import invert_if_negative, CRITERIA_WEIGHTS, DEFAULT_WEIGHT, SCORE_RANGE_MAX
                    processed_value = invert_if_negative(metric, value, negative_criteria_chapter)
                    weight = CRITERIA_WEIGHTS.get(metric.lower().strip(), DEFAULT_WEIGHT)
                    if metric.lower() == 'forced poetry or metaphor':
                        processed_value = (processed_value / 20.0) ** 1.7 * 20.0
                    #print(metric,weight)
                    chapter_weighted_sum += processed_value * weight
                    total_weight += weight
            if total_weight > 0:
                valid_chapter_scores.append(chapter_weighted_sum / total_weight)

    avg_chapter_score = None
    if valid_chapter_scores:
        import statistics as stats
        avg_chapter_score = stats.mean(valid_chapter_scores)

    # --- Calculate Average Final Score (across multiple final judgments) ---
    final_scores_averages = []  # store each final-judgment's average
    if final_scores_list and isinstance(final_scores_list, list):
        for final_scores_dict in final_scores_list:
            if final_scores_dict and isinstance(final_scores_dict, dict):
                final_weighted_sum = 0.0
                total_weight = 0.0
                for metric, value in final_scores_dict.items():
                    if isinstance(value, (int, float)):
                        from core.scoring import invert_if_negative, CRITERIA_WEIGHTS, DEFAULT_WEIGHT
                        processed_value = invert_if_negative(metric, value, negative_criteria_final)
                        weight = CRITERIA_WEIGHTS.get(metric.lower().strip(), DEFAULT_WEIGHT)
                        if metric.lower() == 'forced poetry or metaphor':
                            processed_value = (processed_value / 20.0) ** 1.7 * 20.0
                        #print(metric,weight)
                        final_weighted_sum += processed_value * weight
                        total_weight += weight
                if total_weight > 0:
                    final_avg_for_one_judge = final_weighted_sum / total_weight
                    final_scores_averages.append(final_avg_for_one_judge)

    if final_scores_averages:
        import statistics
        avg_final_score = statistics.mean(final_scores_averages)
    else:
        avg_final_score = None

    # --- Combine chapter + final using weighting ---
    # We'll use the same logic: each chapter has weight=1, final piece has weight=FINAL_SCORE_WEIGHT
    if avg_chapter_score is None and avg_final_score is None:
        return None

    total_score = 0.0
    total_wt = 0.0

    # Use the number of chapters as the 'weight' for chapter average:
    if avg_chapter_score is not None and len(valid_chapter_scores) > 0:
        total_score += avg_chapter_score * len(valid_chapter_scores)
        total_wt += len(valid_chapter_scores)

    # Then final piece has a constant global weight:
    if avg_final_score is not None:
        total_score += avg_final_score * FINAL_SCORE_WEIGHT
        total_wt += FINAL_SCORE_WEIGHT

    if total_wt == 0:
        return None

    final_weighted_score = total_score / total_wt
    return final_weighted_score



def aggregate_longform_scores(
    tasks: List[Dict[str, Any]],
    negative_criteria_chapter: List[str],
    negative_criteria_final: List[str]
) -> Dict[str, Any]:
    """
    Aggregates scores across all completed tasks in a run.

    Returns a dict:
      {
        "overall_score_0_20": float (average weighted score across tasks),
        "eqbench_longform_score_0_100": float (overall_score_0_20 * 5),
        "num_tasks_scored": int,
        "num_tasks_total": int
      }
    """
    task_scores = []
    num_tasks_total = len(tasks)
    num_tasks_scored = 0

    for task_data in tasks:
        # Ensure task is actually completed and judged
        status = task_data.get("status")
        if status != "completed":
             logging.debug(f"Skipping task {task_data.get('prompt_id')} (iter {task_data.get('iteration_index')}) for aggregation - status is '{status}'.")
             continue

        task_score = calculate_task_score(task_data, negative_criteria_chapter, negative_criteria_final)
        if task_score is not None:
            task_scores.append(task_score)
            num_tasks_scored += 1
        else:
             logging.warning(f"Could not calculate score for completed task {task_data.get('prompt_id')} (iter {task_data.get('iteration_index')}).")


    if not task_scores:
        logging.error("No tasks could be scored in this run.")
        return {
            "overall_score_0_20": 0.0,
            "eqbench_longform_score_0_100": 0.0,
            "num_tasks_scored": 0,
            "num_tasks_total": num_tasks_total,
            "error": "No tasks were successfully scored."
        }

    # Calculate final average score across tasks
    overall_avg_score_0_20 = stats.mean(task_scores)
    # Scale to 0-100 for eqbench-like score
    eqbench_score_0_100 = overall_avg_score_0_20 * (100.0 / SCORE_RANGE_MAX) # More general scaling

    return {
        "overall_score_0_20": round(overall_avg_score_0_20, 2),
        "eqbench_longform_score_0_100": round(eqbench_score_0_100, 2),
        "num_tasks_scored": num_tasks_scored,
        "num_tasks_total": num_tasks_total
    }


def bootstrap_benchmark_stability_longform(
    tasks: List[Dict[str, Any]],
    negative_criteria_chapter: List[str],
    negative_criteria_final: List[str],
    n_bootstrap: int = 500,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Performs bootstrap resampling on the overall benchmark score (0-20 scale).
    """
    if not tasks:
        return {"error": "No tasks provided for bootstrap analysis."}

    # Calculate the original score
    original_result = aggregate_longform_scores(tasks, negative_criteria_chapter, negative_criteria_final)
    if "error" in original_result:
         return {"error": f"Could not calculate original score for bootstrap: {original_result['error']}"}
    original_score = original_result["overall_score_0_20"]

    boot_scores = []
    num_tasks = len(tasks)
    if num_tasks == 0:
         return {"error": "Zero tasks available for resampling."}

    logging.info(f"Starting bootstrap analysis with {n_bootstrap} iterations on {num_tasks} tasks...")

    for i in range(n_bootstrap):
        # Sample tasks with replacement
        sample_tasks = random.choices(tasks, k=num_tasks)
        # Calculate score for the resampled set
        sample_result = aggregate_longform_scores(sample_tasks, negative_criteria_chapter, negative_criteria_final)
        if "error" not in sample_result:
            boot_scores.append(sample_result["overall_score_0_20"])
        else:
             # Log if a bootstrap iteration fails, but continue
             logging.warning(f"Bootstrap iteration {i+1} failed to produce a score.")


    if not boot_scores:
        return {
            "error": "Bootstrap analysis failed: No scores generated during resampling.",
            "original": original_score,
            "n_bootstrap": n_bootstrap,
            "num_tasks": num_tasks
        }

    # Calculate confidence interval using percentile method
    boot_scores.sort()
    alpha = (1.0 - confidence_level)
    lower_idx = int(alpha / 2.0 * len(boot_scores))
    upper_idx = int((1.0 - alpha / 2.0) * len(boot_scores))

    # Ensure indices are within bounds
    lower_idx = max(0, lower_idx)
    upper_idx = min(len(boot_scores) - 1, upper_idx) # -1 because index is 0-based

    ci_lower = boot_scores[lower_idx]
    ci_upper = boot_scores[upper_idx]

    # Calculate other statistics
    mean_ = np.mean(boot_scores)
    std_ = np.std(boot_scores, ddof=1) # Use sample standard deviation
    stderr_ = std_ / np.sqrt(len(boot_scores)) if len(boot_scores) > 0 else 0

    logging.info(f"Bootstrap analysis complete. Mean: {mean_:.2f}, StdDev: {std_:.2f}, CI: ({ci_lower:.2f}, {ci_upper:.2f})")

    return {
        "original": original_score,
        "bootstrap_mean": float(mean_),
        "bootstrap_std": float(std_),
        "standard_error": float(stderr_), # Standard error of the mean
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "n_successful_samples": len(boot_scores)
    }