#!/usr/bin/env python3
import os, sys, logging, json
from dag import InferenceNode, ConfidenceCheckNode, FallbackNode, SelfHealingDAG
from model_utils import safe_load_model, load_backup_model
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "project.log")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("cli")

def interact(prompt):
    print(prompt, flush=True)
    try:
        resp = input("User: ")
    except EOFError:
        return ""
    return resp

def main(prefer_transformer=False, confidence_threshold=0.75, fallback_strategy='ask_user'):
    print(f"[CLI] Loading model (prefer_transformer={prefer_transformer})", flush=True)
    try:
        model, mtype = safe_load_model(prefer_transformer=prefer_transformer)
    except Exception as e:
        print(f"[CLI] Failed to load any model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        backup = load_backup_model()
    except Exception:
        backup = None

    inference = InferenceNode(model, model_type=mtype)
    confnode = ConfidenceCheckNode(threshold=confidence_threshold)
    fallbacknode = FallbackNode(backup_strategy=fallback_strategy, backup_model=backup)
    dag = SelfHealingDAG(inference, confnode, fallbacknode)

    print("Self-Healing Classification CLI\nType 'exit' to quit.\n", flush=True)
    try:
        while True:
            text = input("Input: ").strip()
            if text.lower() in ['exit', 'quit']:
                print("Bye.", flush=True)
                break
            if not text:
                continue
            result = dag.run(text, interact_fn=interact)
            print(f"Final Label: {result.get('final_label')} | Confidence: {result.get('confidence')}", flush=True)
            print("---", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...", flush=True)

if __name__ == "__main__":
    main()
