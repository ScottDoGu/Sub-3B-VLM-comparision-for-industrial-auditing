from typing import Dict, Any
from exhaustive_strategies.base_strategy import Strategy


class CoTVerify(Strategy):
    name = "cot_verify"

    def build_prompt(
        self,
        base_prompt: str,
        metadata: Dict[str, Any],
        model_name: str,
    ) -> str:
        # Simple, model‑agnostic CoT wrapper
        return (
            base_prompt
            + "\n\nThink step by step about whether the image shows a safety violation. "
              "Explain your reasoning briefly, then give a final verdict as "
              "\"VIOLATION\", \"NO_VIOLATION\", or \"NO_VERDICT\"."
        )

    def postprocess(
        self,
        model_output: str,
        parsed_output: str,
        metadata: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Very lightweight self‑verification:
        - If the model expresses uncertainty, map to NO_VERDICT.
        - If it confidently contradicts obvious metadata (optional), flag as wrong.
        """
        lower = model_output.lower()

        is_no_verdict = any(
            phrase in lower
            for phrase in [
                "not sure",
                "cannot determine",
                "uncertain",
                "unclear",
            ]
        )

        if is_no_verdict:
            final = "NO_VERDICT"
        else:
            final = parsed_output

        gt = metadata["expected_verdict"]

        return {
            "model_output": model_output,
            "parsed_output": final,
            "is_correct": final == gt,
            "is_wrong": (final != gt) and (final != "NO_VERDICT"),
            "is_no_verdict": final == "NO_VERDICT",
            "verification_trace": {
                "original_parsed": parsed_output,
                "uncertainty_detected": is_no_verdict,
            },
        }
