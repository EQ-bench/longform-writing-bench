# utils/file_io.py
import os
import json
import logging
import threading
from typing import Dict, Any
import time

_file_locks = {}
_file_locks_lock = threading.Lock()

def get_file_lock(file_path: str) -> threading.Lock:
    """
    Acquire or create a per-file lock to avoid concurrent writes.
    """
    with _file_locks_lock:
        if file_path not in _file_locks:
            _file_locks[file_path] = threading.Lock()
        return _file_locks[file_path]

def load_json_file(file_path: str) -> dict:
    """
    Thread-safe read of a JSON file, returning an empty dict if not found or error.
    """
    lock = get_file_lock(file_path)
    with lock:
        if not os.path.exists(file_path):
            logging.debug(f"File not found: {file_path}, returning empty dict.")
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip(): # Handle empty file
                    logging.warning(f"File is empty: {file_path}, returning empty dict.")
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {file_path}: {e}. Content snippet: '{content[:100]}...'")
            # Optionally, try to recover or backup the corrupted file here
            return {} # Return empty dict to avoid crashing, but log the error
        except Exception as e:
            logging.exception(f"Error reading {file_path}: {e}") # Use .exception for stack trace
            return {}

def _atomic_write_json(data: Dict[str, Any], file_path: str):
    """Writes JSON data atomically using a temporary file."""
    temp_path = file_path + ".tmp"
    try:
        # Ensure the directory exists
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic replace operation
        os.replace(temp_path, file_path)
        logging.debug(f"Successfully wrote JSON atomically to {file_path}")
    except Exception as e:
        logging.exception(f"Error during atomic write to {file_path}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as rm_err:
                logging.error(f"Error removing temporary file {temp_path}: {rm_err}")
        raise # Re-raise the exception to be caught by the caller

def save_json_file(data: Dict[str, Any], file_path: str, max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    """
    Thread-safe and atomic save of a dictionary to a JSON file with retries.
    """
    lock = get_file_lock(file_path)
    for attempt in range(max_retries):
        acquired_lock = lock.acquire(timeout=10) # Add a timeout to prevent deadlocks
        if not acquired_lock:
             logging.warning(f"Could not acquire lock for {file_path} on attempt {attempt+1}. Retrying...")
             if attempt < max_retries - 1:
                 time.sleep(retry_delay * (attempt + 1)) # Exponential backoff for lock contention
             continue

        try:
            _atomic_write_json(data, file_path)
            return True # Success
        except Exception as e:
            logging.error(f"save_json_file() attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay) # Wait before retrying write operation
        finally:
            if acquired_lock:
                lock.release()

    logging.error(f"Failed to save JSON to {file_path} after {max_retries} attempts.")
    return False

def update_run_data(runs_file: str, run_key: str, update_dict: Dict[str, Any],
                    max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    """
    Thread-safe function to MERGE partial run data into the existing run file.
    Uses atomic writes and handles potential read/write conflicts.
    Specific merge logic for 'longform_tasks'.
    """
    lock = get_file_lock(runs_file)
    for attempt in range(max_retries):
        acquired_lock = lock.acquire(timeout=10)
        if not acquired_lock:
            logging.warning(f"Could not acquire lock for {runs_file} (update) on attempt {attempt+1}. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            continue

        current_runs = {} # Initialize fresh for each attempt inside the lock
        try:
            # --- Read Phase ---
            if os.path.exists(runs_file):
                try:
                    with open(runs_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            current_runs = json.loads(content)
                        else:
                             logging.warning(f"Run file {runs_file} is empty. Starting fresh.")
                             current_runs = {}
                except (json.JSONDecodeError, IOError) as e:
                    logging.error(f"Error reading/parsing run file {runs_file} on attempt {attempt+1}: {e}. Aborting update for this attempt.")
                    # Optionally backup corrupted file here
                    if attempt < max_retries - 1:
                         time.sleep(retry_delay)
                    continue # Go to next attempt
                except Exception as e:
                     logging.exception(f"Unexpected error reading run file {runs_file} on attempt {attempt+1}: {e}")
                     if attempt < max_retries - 1:
                         time.sleep(retry_delay)
                     continue

            if not isinstance(current_runs, dict):
                logging.warning(f"Run file {runs_file} does not contain a valid JSON object. Resetting to empty dict.")
                current_runs = {}

            # --- Merge Phase ---
            if run_key not in current_runs:
                current_runs[run_key] = {}

            # Deep merge logic specifically for 'longform_tasks'
            # update_dict structure expected: { "longform_tasks": { iter_idx_str: { prompt_id_str: task_dict } } }
            # or other top-level keys like "status", "end_time"
            for top_key, new_val in update_dict.items():
                if top_key == "longform_tasks":
                    if top_key not in current_runs[run_key]:
                        current_runs[run_key][top_key] = {}
                    if not isinstance(current_runs[run_key][top_key], dict):
                         logging.warning(f"Overwriting non-dict '{top_key}' in run {run_key} with new task data.")
                         current_runs[run_key][top_key] = {} # Reset if corrupted

                    # new_val should be: { iteration_idx_str => { prompt_id_str => {...task_data...} } }
                    if isinstance(new_val, dict):
                        for iter_idx_str, prompt_map in new_val.items():
                            if iter_idx_str not in current_runs[run_key][top_key]:
                                current_runs[run_key][top_key][iter_idx_str] = {}
                            if not isinstance(current_runs[run_key][top_key][iter_idx_str], dict):
                                logging.warning(f"Overwriting non-dict iteration '{iter_idx_str}' in run {run_key} with new prompt data.")
                                current_runs[run_key][top_key][iter_idx_str] = {} # Reset if corrupted

                            # Merge at the prompt_id level (overwriting the entire task dict)
                            if isinstance(prompt_map, dict):
                                for p_id_str, p_data in prompt_map.items():
                                    # This overwrites the existing task data for this iter/prompt
                                    # which is correct as the task object manages its internal state
                                    current_runs[run_key][top_key][iter_idx_str][p_id_str] = p_data
                            else:
                                 logging.warning(f"Expected dict for prompt_map at iteration '{iter_idx_str}', got {type(prompt_map)}. Skipping merge for this iteration.")
                    else:
                         logging.warning(f"Expected dict for '{top_key}', got {type(new_val)}. Skipping merge for tasks.")

                elif top_key in ["results"]: # Handle results dictionary merge (e.g., benchmark_results)
                    if top_key not in current_runs[run_key]:
                        current_runs[run_key][top_key] = {}
                    if isinstance(new_val, dict) and isinstance(current_runs[run_key][top_key], dict):
                         # Simple shallow merge for results keys (like benchmark_results, bootstrap_analysis)
                         current_runs[run_key][top_key].update(new_val)
                    else:
                         # Overwrite if types don't match or new_val isn't a dict
                         current_runs[run_key][top_key] = new_val
                else:
                    # Overwrite any other top-level key (like status, start_time, end_time, test_model)
                    current_runs[run_key][top_key] = new_val

            # --- Write Phase ---
            try:
                _atomic_write_json(current_runs, runs_file)
                logging.debug(f"Successfully updated run_key={run_key} in {runs_file}")
                return True # Success
            except Exception as e:
                logging.error(f"Error saving merged run data on attempt {attempt+1}: {e}")
                # The error in _atomic_write_json is already logged with exception details

        except Exception as e:
             # Catch any unexpected errors during read/merge phase
             logging.exception(f"Unexpected error during update_run_data attempt {attempt+1}: {e}")
        finally:
            if acquired_lock:
                lock.release()

        # If we reach here, the attempt failed, wait before the next one
        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    logging.error(f"update_run_data() failed after {max_retries} attempts for run {run_key} in {runs_file}")
    return False