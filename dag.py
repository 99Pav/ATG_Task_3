#!/usr/bin/env python3
import logging, time

logger = logging.getLogger("SelfHealingDAG")

class InferenceNode:
    def __init__(self, model, model_type='backup'):
        self.model = model
        self.model_type = model_type

    def run(self, text):
        if self.model_type == 'backup':
            probs = self.model.predict_proba([text])[0]
            classes = self.model.classes_
            max_i = int(probs.argmax())
            return {"label": classes[max_i], "confidence": float(probs[max_i]), "raw": probs.tolist()}
        else:
            out = self.model(text)
            if isinstance(out, list) and len(out) > 0:
                o = out[0]
                return {"label": o.get("label"), "confidence": float(o.get("score")), "raw": out}
            return {"label": str(out), "confidence": 0.0, "raw": out}

class ConfidenceCheckNode:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def run(self, inference_result):
        conf = inference_result.get("confidence", 0.0)
        return {"accept": conf >= self.threshold, "confidence": conf, "threshold": self.threshold}

class FallbackNode:
    def __init__(self, backup_strategy='ask_user', backup_model=None):
        self.backup_strategy = backup_strategy
        self.backup_model = backup_model

    def run(self, text, last_inference):
        if self.backup_strategy == 'backup_model' and self.backup_model is not None:
            probs = self.backup_model.predict_proba([text])[0]
            classes = self.backup_model.classes_
            max_i = int(probs.argmax())
            return {"label": classes[max_i], "confidence": float(probs[max_i]), "method": "backup_model"}
        return {"label": None, "confidence": 0.0, "method": "ask_user"}

class SelfHealingDAG:
    def __init__(self, inference_node, confidence_node, fallback_node):
        self.inference = inference_node
        self.confcheck = confidence_node
        self.fallback = fallback_node

    def run(self, text, interact_fn=None):
        t0 = time.time()
        inf = self.inference.run(text)
        logger.info(f"[InferenceNode] Predicted label: {inf.get('label')} | Confidence: {inf.get('confidence'):.2f}")
        chk = self.confcheck.run(inf)

        if chk.get('accept'):
            return {"final_label": inf.get('label'), "confidence": inf.get('confidence'),
                    "source": self.inference.model_type, "resolved_by": "model",
                    "latency": time.time()-t0}

        fb = self.fallback.run(text, inf)
        if fb.get("method") == "backup_model" and fb.get("label") is not None:
            return {"final_label": fb.get("label"), "confidence": fb.get("confidence"),
                    "source": "backup_model", "resolved_by": "backup_model",
                    "latency": time.time()-t0}

        if interact_fn is None:
            return {"final_label": None, "confidence": inf.get('confidence'),
                    "source": "ask_user", "resolved_by": "none", "latency": time.time()-t0}

        prompt = (
            f"Model predicted '{inf.get('label')}' with confidence "
            f"{inf.get('confidence'):.2f}. Could you clarify the correct label? (yes/no/other)"
        )
        user_resp = interact_fn(prompt)
        resp_lower = str(user_resp).strip().lower()
        if resp_lower in ['yes', 'y', 'negative', 'neg']:
            corrected = 'negative'
        elif resp_lower in ['no', 'n', 'positive', 'pos']:
            corrected = 'positive'
        else:
            corrected = resp_lower or inf.get('label')
        return {"final_label": corrected, "confidence": 1.0, "source": "user",
                "resolved_by": "user", "latency": time.time()-t0}
