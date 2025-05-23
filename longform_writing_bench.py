# longform_creative_writing_bench.py
import argparse
import sys
import signal
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure core and utils are importable
# If running from the 'longform-creative-writing' directory:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# If running from parent 'ai' directory:
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.logging_setup import setup_logging, get_verbosity
from core.benchmark import run_longform_bench, RUNS_FILENAME # Import the new main function

def signal_handler(signum, frame):
    """Handles graceful shutdown on SIGINT or SIGTERM."""
    print(f"\n[INFO] Signal {signal.Signals(signum).name} received. Shutting down gracefully...")
    # Perform any necessary cleanup here if needed
    logging.info(f"Shutdown initiated by signal {signum}.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Long-Form Creative Writing Benchmark.")
    parser.add_argument("--test-model", required=True, help="Identifier for the model being tested (e.g., 'openai/gpt-4o').")
    parser.add_argument("--judge-model", required=True, help="Identifier for the model used for judging (e.g., 'anthropic/claude-3.5-sonnet').")
    parser.add_argument("--runs-file", default=os.path.join("results", RUNS_FILENAME), help=f"JSON file to store run data and results (default: results/{RUNS_FILENAME}).")
    parser.add_argument("--data-dir", default="data", help="Directory containing prompts, criteria, etc. (default: data).")
    parser.add_argument("--run-id", help="Optional: Specify a base ID for the run key to group runs or resume.")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads for processing tasks (default: 4).")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to run each initial prompt (default: 1).")
    parser.add_argument("--verbosity", choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default=None, help="Logging level (overrides .env setting).")
    parser.add_argument("--redo-judging", action="store_true", default=False, help="Force re-judging of existing generated content.")
    parser.add_argument("--save-interval", type=int, default=1, help="Save task state every N generation steps (default: 1).")
    parser.add_argument("--skip-generation", action="store_true", default=False, help="Skip the generation phase (useful for re-judging only).")
    parser.add_argument("--skip-chapter-judging", action="store_true", default=False, help="Skip the chapter-by-chapter judging phase.")
    parser.add_argument("--skip-final-judging", action="store_true", default=False, help="Skip the final piece judging phase.")


    args = parser.parse_args()

    # Setup logging
    verbosity = get_verbosity(args.verbosity)
    setup_logging(verbosity)

    # Ensure results directory exists
    results_dir = os.path.dirname(args.runs_file)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logging.info(f"Created results directory: {results_dir}")

    # Hook signals for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # kill command

    try:
        run_key = run_longform_bench(
            test_model=args.test_model,
            judge_model=args.judge_model,
            runs_file=args.runs_file,
            data_dir=args.data_dir,
            num_threads=args.threads,
            run_id=args.run_id,
            iterations=args.iterations,
            redo_judging=args.redo_judging,
            save_interval=args.save_interval,
            skip_generation=args.skip_generation,
            skip_chapter_judging=args.skip_chapter_judging,
            skip_final_judging=args.skip_final_judging
        )
        logging.info(f"Benchmark run completed successfully. Run key: {run_key}")

    except FileNotFoundError as e:
         logging.critical(f"Initialization failed: Required file not found. {e}")
         print(f"\nERROR: Required file not found. Please check your --data-dir path and contents. Details: {e}", file=sys.stderr)
         sys.exit(1)
    except ValueError as e:
         logging.critical(f"Initialization failed: Invalid value or configuration. {e}")
         print(f"\nERROR: Invalid value or configuration. Details: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        logging.exception("An unexpected error occurred during the benchmark run.")
        print(f"\nFATAL ERROR: An unexpected error occurred. Check logs for details. Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()