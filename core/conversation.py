# core/conversation.py
import time
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from utils.api import APIClient
from utils.file_io import update_run_data
from core.scoring import parse_judge_scores_longform # Use the new parsing function

# Constants
NUM_FINAL_JUDGMENTS = 1 # int(os.getenv("NUM_FINAL_JUDGMENTS", 6))
NUM_PLANNING_STEPS = 5 # Prompts 1-5 (Plan, Plan, Critique, Final Plan, Characters)
NUM_CHAPTERS = int(os.getenv("NUM_CHAPTERS", 8))
TOTAL_STEPS = NUM_PLANNING_STEPS + NUM_CHAPTERS # 5 + 8 = 13 steps (Prompts 1-13)
PLAN_STEP_INDEX = 4 # Index (0-based) of the final plan output (prompt 4)
CHAR_PROFILE_STEP_INDEX = 5 # Index (0-based) of the character profile output (prompt 5)
FIRST_CHAPTER_STEP_INDEX = 6 # Index (0-based) of the first chapter writing step (prompt 6)

class LongformCreativeTask:
    """
    Represents a single long-form creative writing task for one initial prompt
    across one iteration, etc.
    """

    def __init__(
        self,
        prompt_id: str,
        writing_prompt: str,
        iteration_index: int,
        test_model: str,
        judge_model: str,
        prompt_templates: Dict[int, str]
    ):
        self.prompt_id = prompt_id
        self.writing_prompt = writing_prompt
        self.iteration_index = iteration_index
        self.test_model = test_model
        self.judge_model = judge_model
        self.prompt_templates = prompt_templates

        self.status = "initialized"
        self.current_step = 0
        self.start_time = None
        self.end_time = None
        self.error_message = None

        # Stores the raw text output of each generation step (1..TOTAL_STEPS)
        self.step_outputs: Dict[str, str] = {}

        # Chapter-level judging:
        self.chapter_judge_scores: Dict[int, Dict[str, float]] = {}
        self.chapter_raw_judge_text: Dict[int, str] = {}

        # Instead of a single dict/string, store multiple final-judge runs:
        self.final_judge_scores: List[Dict[str, float]] = []
        self.final_raw_judge_texts: List[str] = []

        self._final_plan_text: Optional[str] = None
        self._character_profiles_text: Optional[str] = None
        self._full_story_text: Optional[str] = None

    def _log(self, level, message):
        """Helper for logging with task context."""
        prefix = f"[Task prompt_id={self.prompt_id}, iter={self.iteration_index}]"
        if level == 'info':
            logging.info(f"{prefix} {message}")
        elif level == 'warning':
            logging.warning(f"{prefix} {message}")
        elif level == 'error':
            logging.error(f"{prefix} {message}")
        elif level == 'debug':
            logging.debug(f"{prefix} {message}")
        elif level == 'exception':
             logging.exception(f"{prefix} {message}")

    def _save_state(self, runs_file: str, run_key: str):
        """Saves the current state of the task."""
        if runs_file and run_key:
            task_data = self.to_dict()
            update_dict = {
                "longform_tasks": {
                    str(self.iteration_index): {
                        str(self.prompt_id): task_data
                    }
                }
            }
            if not update_run_data(runs_file, run_key, update_dict):
                 self._log('error', f"Failed to save state to {runs_file}")

    def _build_messages_for_step(self, step_num: int) -> List[Dict[str, str]]:
        """Constructs the message history for the API call at a given step."""
        messages = []
        # Step 1: Only the initial writing prompt within the first template
        #if step_num == 1:
        #    print(self.writing_prompt)
        #    prompt_text = self.prompt_templates[step_num].replace("{writing_prompt}", self.writing_prompt["writing_prompt"])
        #    messages.append({"role": "user", "content": prompt_text})
        # Subsequent steps: Include previous user prompts and assistant responses
        #else:
        # Add history up to the previous step
        for i in range(1, step_num):
            prev_prompt_text = self.prompt_templates[i]
            # Substitute original prompt only in the first step's template text
            #if i == 1:
            prev_prompt_text = prev_prompt_text.replace("{writing_prompt}", self.writing_prompt["writing_prompt"])
            prev_prompt_text = prev_prompt_text.replace("{n_chapters}", str(NUM_CHAPTERS))

            messages.append({"role": "user", "content": prev_prompt_text})
            if str(i) in self.step_outputs:
                messages.append({"role": "assistant", "content": self.step_outputs[str(i)]})
            else:
                # This shouldn't happen if run sequentially, but handle defensively
                self._log('warning', f"Missing output for step {i} when building history for step {step_num}. History may be incomplete.")

        # Add the current step's prompt
        current_prompt_text = self.prompt_templates[step_num]
        current_prompt_text = current_prompt_text.replace("{writing_prompt}", self.writing_prompt["writing_prompt"])
        current_prompt_text = current_prompt_text.replace("{n_chapters}", str(NUM_CHAPTERS))
        messages.append({"role": "user", "content": current_prompt_text})

        return messages

    def run_generation_sequence(self, api_clients: Dict[str, APIClient], runs_file: str, run_key: str, save_interval: int = 1, retries: int = 5):
        """Runs the generation steps 1 through TOTAL_STEPS sequentially."""
        if self.status not in ["initialized", "generating"]:
            self._log('info', f"Skipping generation, status is '{self.status}'.")
            return

        self.status = "generating"
        if not self.start_time:
            self.start_time = time.time()

        test_api = api_clients["test"]        

        for step in range(self.current_step + 1, TOTAL_STEPS + 1):
            self.current_step = step
            self._log('info', f"Starting generation step {step}/{TOTAL_STEPS}...")

            # Check if output already exists (e.g., resuming)
            if str(step) in self.step_outputs:
                self._log('info', f"Output for step {step} already exists, skipping generation.")
                continue

            messages = self._build_messages_for_step(step)
            # Determine max_tokens based on step type
            # Planning/Critique/Chars steps can be shorter, Chapters need ~1000 words (=> more tokens)
            if step < FIRST_CHAPTER_STEP_INDEX: # Steps 1-5
                max_tokens = 4000 # Generous buffer for planning/chars
            else: # Steps 6-13 (Chapters 1-8)
                max_tokens = 4000 # Needs to accommodate ~1000 words + prompt overhead

            try:
                attempt = 0
                response = None

                while attempt < retries:
                    response = test_api.generate(
                        model=self.test_model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=max_tokens,
                        min_p=0.1
                    )
                    if response and len(response.strip()) > 500:
                        break
                    else:
                        attempt += 1
                        self._log('error', f"Response too short. Attempt {str(attempt)}. Response: {response}")
                    

                if not response or len(response.strip()) < 500: # Basic check for empty/very short response
                    raise ValueError(f"Generation for step {step} produced an unexpectedly short or empty response.")

                self.step_outputs[str(step)] = response.strip()
                self._log('info', f"Completed generation step {step}. Output length: ~{len(response)} chars.")

                # Save state periodically
                if step % save_interval == 0 or step == TOTAL_STEPS:
                    self._save_state(runs_file, run_key)

            except Exception as e:
                self.error_message = f"Error during generation step {step}: {str(e)}"
                self.status = "error"
                self._log('exception', self.error_message)
                self._save_state(runs_file, run_key)
                return # Stop generation sequence on error

        # After successful generation of all steps
        self._final_plan_text = self.step_outputs.get(str(PLAN_STEP_INDEX))
        self._character_profiles_text = self.step_outputs.get(str(CHAR_PROFILE_STEP_INDEX))
        self._full_story_text = "\n\n".join([self.step_outputs.get(str(i), "") for i in range(FIRST_CHAPTER_STEP_INDEX, TOTAL_STEPS + 1)])

        self.status = "generated" # Ready for judging
        self._log('info', "Generation sequence completed successfully.")
        self._save_state(runs_file, run_key)


    def judge_chapters(
        self,
        api_clients: Dict[str, APIClient],
        judge_prompt_template: str,
        criteria: List[str],
        negative_criteria: List[str],
        runs_file: str,
        run_key: str
    ):
        """Judges each generated chapter (steps 6-13) individually."""
        if self.status not in ["generated", "judging_chapters"]:
             if self.status != "error": # Don't log warning if already errored
                 self._log('warning', f"Cannot judge chapters, status is '{self.status}'. Generation might be incomplete or failed.")
             return

        # Ensure plan and characters are available
        final_plan = self.step_outputs.get(str(PLAN_STEP_INDEX))
        char_profiles = self.step_outputs.get(str(CHAR_PROFILE_STEP_INDEX))
        if not final_plan or not char_profiles:
             self.error_message = "Cannot judge chapters: Missing final plan or character profiles."
             self.status = "error"
             self._log('error', self.error_message)
             self._save_state(runs_file, run_key)
             return

        self.status = "judging_chapters"
        judge_api = api_clients["judge"]

        for chapter_num in range(1, NUM_CHAPTERS + 1):
            step_num = FIRST_CHAPTER_STEP_INDEX + chapter_num - 1

            # Check if already judged
            if str(chapter_num) in self.chapter_judge_scores:
                self._log('info', f"Chapter {chapter_num} already judged, skipping.")
                continue

            chapter_text = self.step_outputs.get(str(step_num))
            if not chapter_text:
                self._log('warning', f"Missing text for chapter {chapter_num} (step {step_num}). Cannot judge.")
                self.chapter_judge_scores[str(chapter_num)] = {}
                self.chapter_raw_judge_text[str(chapter_num)] = "[ERROR: Chapter text missing]"
                continue # Skip judging this chapter

            self._log('info', f"Judging chapter {chapter_num} (step {step_num})...")

            # Prepare judge prompt
            criteria_formatted = "\n".join([f"- {c}" for c in criteria])
            neg_criteria_formatted = ", ".join(negative_criteria) if negative_criteria else "None"

            final_judge_prompt = judge_prompt_template \
                .replace("{writing_prompt}", self.writing_prompt['writing_prompt']) \
                .replace("{final_plan}", final_plan) \
                .replace("{character_profiles}", char_profiles) \
                .replace("{chapter_text}", chapter_text) \
                .replace("{chapter_number}", str(chapter_num)) \
                .replace("{creative_writing_criteria}", criteria_formatted) \
                .replace("{lower_is_better_criteria}", neg_criteria_formatted)


            messages = [{"role": "user", "content": final_judge_prompt}]

            try:
                # Use temperature 0 for consistent judging
                judge_resp = judge_api.generate(self.judge_model, messages, temperature=0.0, max_tokens=16000, use_random_seed=True) # Allow ample tokens for judge reasoning + scores
                scores_dict = parse_judge_scores_longform(judge_resp) # Use the specific parser

                if not scores_dict:
                     self._log('warning', f"Could not parse scores from judge response for chapter {chapter_num}.")
                     # Store empty scores but keep raw text for debugging
                     self.chapter_judge_scores[chapter_num] = {}
                else:
                     self.chapter_judge_scores[chapter_num] = scores_dict

                self.chapter_raw_judge_text[chapter_num] = judge_resp
                self._log('info', f"Completed judging chapter {chapter_num}.")

            except Exception as e:
                error_msg = f"Error during judging chapter {chapter_num}: {str(e)}"
                self._log('exception', error_msg)
                self.chapter_judge_scores[chapter_num] = {}
                self.chapter_raw_judge_text[chapter_num] = f"[ERROR: {error_msg}]"
                # Optionally set task status to error here, or allow final judging attempt

            # Save state after each chapter judgement
            self._save_state(runs_file, run_key)

        self._log('info', "Chapter judging completed.")
        self.status = "judged_chapters" # Ready for final judging
        self._save_state(runs_file, run_key)


    def judge_final_piece(
        self,
        api_clients: Dict[str, Any],
        judge_prompt_template: str,
        criteria: List[str],
        negative_criteria: List[str],
        runs_file: str,
        run_key: str
    ):
        """
        Judges the complete story multiple times (NUM_FINAL_JUDGMENTS) and
        appends all scores/results to self.final_judge_scores / self.final_raw_judge_texts.
        """
        if self.status not in ["judged_chapters", "judging_final"]:
            if self.status != "error":
                self._log('warning', f"Cannot judge final piece, status is '{self.status}'.")
            return

        # Prepare the full story from chapters
        full_story_parts = []
        for chapter_num in range(1, NUM_CHAPTERS + 1):
            step_num = FIRST_CHAPTER_STEP_INDEX + chapter_num - 1
            chapter_text = self.step_outputs.get(str(step_num), "")
            if chapter_text:
                full_story_parts.append(f"# Chapter {chapter_num}\n\n{chapter_text}")
            else:
                full_story_parts.append(f"# Chapter {chapter_num}\n\n[ERROR: Text missing]")
        full_story = "\n\n---\n\n".join(full_story_parts)

        self._log('info', f"Judging final piece {NUM_FINAL_JUDGMENTS} time(s) ...")
        judge_api = api_clients["judge"]

        # Build the base prompt template
        criteria_formatted = "\n".join([f"- {c}" for c in criteria])
        neg_criteria_formatted = ", ".join(negative_criteria) if negative_criteria else "None"

        base_prompt = judge_prompt_template \
            .replace("{writing_prompt}", self.writing_prompt['writing_prompt']) \
            .replace("{full_story}", full_story) \
            .replace("{creative_writing_criteria}", criteria_formatted) \
            .replace("{lower_is_better_criteria}", neg_criteria_formatted)

        # Loop multiple times
        for run_index in range(NUM_FINAL_JUDGMENTS):
            messages = [{"role": "user", "content": base_prompt}]
            try:
                judge_resp = judge_api.generate(
                    self.judge_model,
                    messages,
                    temperature=0.0,
                    max_tokens=16000,
                    use_random_seed=True
                )
                # Parse scores
                from core.scoring import parse_judge_scores_longform
                scores_dict = parse_judge_scores_longform(judge_resp)

                self.final_judge_scores.append(scores_dict if scores_dict else {})
                self.final_raw_judge_texts.append(judge_resp)

                self._log('info', f"Final piece judged (run {run_index+1}/{NUM_FINAL_JUDGMENTS}).")

            except Exception as e:
                error_msg = f"Error during final judging run {run_index+1}: {str(e)}"
                self._log('exception', error_msg)
                self.final_judge_scores.append({})
                self.final_raw_judge_texts.append(f"[ERROR: {error_msg}]")

            # Save after each final-judge run
            self._save_state(runs_file, run_key)

        # After all final judgements
        self.status = "completed"
        self.end_time = time.time()
        self._save_state(runs_file, run_key)
        self._log('info', "All final judging runs completed.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the task state to a dictionary.
        """
        return {
            "prompt_id": self.prompt_id,
            "writing_prompt": self.writing_prompt,
            "iteration_index": self.iteration_index,
            "test_model": self.test_model,
            "judge_model": self.judge_model,
            "status": self.status,
            "current_step": self.current_step,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
            "step_outputs": self.step_outputs,
            "chapter_judge_scores": self.chapter_judge_scores,
            "chapter_raw_judge_text": self.chapter_raw_judge_text,
            # Now storing a list of dicts, plus a list of strings:
            "final_judge_scores": self.final_judge_scores,
            "final_raw_judge_texts": self.final_raw_judge_texts
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prompt_templates: Dict[int, str]):
        """
        Deserializes a task from a dictionary.
        """
        writing_prompt = data.get("writing_prompt", "Unknown Prompt")

        obj = cls(
            prompt_id=data["prompt_id"],
            writing_prompt=writing_prompt,
            iteration_index=data["iteration_index"],
            test_model=data["test_model"],
            judge_model=data["judge_model"],
            prompt_templates=prompt_templates
        )
        obj.status = data.get("status", "initialized")
        obj.current_step = data.get("current_step", 0)
        obj.start_time = data.get("start_time")
        obj.end_time = data.get("end_time")
        obj.error_message = data.get("error_message")
        obj.step_outputs = data.get("step_outputs", {})
        obj.chapter_judge_scores = data.get("chapter_judge_scores", {})
        obj.chapter_raw_judge_text = data.get("chapter_raw_judge_text", {})

        # Load final_judge_scores and raw texts as lists
        obj.final_judge_scores = data.get("final_judge_scores", [])
        obj.final_raw_judge_texts = data.get("final_raw_judge_texts", [])

        # Re-populate convenience accessors if already generated
        if obj.status in ["generated", "judging_chapters", "judged_chapters", "judging_final", "completed"]:
            from core.conversation import PLAN_STEP_INDEX, CHAR_PROFILE_STEP_INDEX, FIRST_CHAPTER_STEP_INDEX, TOTAL_STEPS
            obj._final_plan_text = obj.step_outputs.get(str(PLAN_STEP_INDEX))
            obj._character_profiles_text = obj.step_outputs.get(str(CHAR_PROFILE_STEP_INDEX))
            obj._full_story_text = "\n\n".join([
                obj.step_outputs.get(str(i), "") for i in range(FIRST_CHAPTER_STEP_INDEX, TOTAL_STEPS + 1)
            ])

        return obj