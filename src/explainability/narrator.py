"""
Local LLM Narrative Reasoning Layer for ST-GRAD (Phase 6).
Translates mathematical explainability records (Integrated Gradients, MVCA views,
and graph channel interventions) into structured, executive-grade English narratives.

Model: Qwen2.5-3B-Instruct (Q4_K_M GGUF) via llama-cpp-python.
Features:
- Sequential GPU swap to respect hardware limits (RTX 2050 / 3.7GB VRAM)
- Strict non-causal grounding system prompt preventing external hallucinations
- Auto-download of GGUF weights from HuggingFace Hub with local cache
- Structured output: Prediction Summary, Key Drivers, Temporal Focus, Peer Influence, Diagnostics
"""

import os
import gc
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import torch

logger = logging.getLogger(__name__)


@dataclass
class NarratorConfig:
    """Configuration for the local LLM narrator."""
    model_repo: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    model_filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    model_path: Optional[str] = None
    cache_dir: str = "models/llm"
    n_gpu_layers: int = -1          # -1 offloads all layers to GPU if supported
    n_ctx: int = 4096
    max_tokens: int = 800           # 500-800 tokens for detailed analyst narrative
    temperature: float = 0.3        # Low temperature for factual fidelity


class ExplanationNarrator:
    """
    Synthesizes neural network prediction diagnostics into plain English.
    Enforces strict grounding against the diagnostic record and manages GPU VRAM swapping.
    """

    SYSTEM_PROMPT = """You are a senior quantitative financial model analyst. Your task is to explain neural network prediction diagnostics in structured, professional English.

STRICT GROUNDING RULES:
1. Grounding: Only reference data explicitly present in the provided diagnostic record. Never fabricate external news, company announcements, earnings reports, or macroeconomic events.
2. Non-Causal Framing: Attribution scores and intervention deltas quantify the neural network's internal computational reliance and sensitivity. Never use causal economic phrasing (e.g., do not say "Stock A caused Stock B to rise"). Instead use model-dependence phrasing (e.g., "The model relied on...", "The model was sensitive to...", "Counterfactual ablation revealed...").
3. Signal Resolution: If technical indicators present conflicting signals (e.g., bullish RSI vs bearish moving average), explicitly explain how the model balanced them based on their respective attribution magnitudes.
4. Structure: Organize your narrative into the following exact sections with markdown headers:
   ## Prediction Summary
   ## Key Feature Drivers
   ## Temporal Focus
   ## Peer & Spatial Influence
   ## Reliability & Convergence Diagnostics
5. Tone: Professional, objective, analytical. Do not use conversational filler or emojis."""

    def __init__(self, config: Optional[NarratorConfig] = None):
        self.config = config or NarratorConfig()

    def _ensure_model_downloaded(self) -> str:
        """
        Verify presence of model weights or download from HuggingFace Hub.
        Returns the absolute path to the GGUF model file.
        """
        # 1. Custom model path override
        if self.config.model_path and os.path.exists(self.config.model_path):
            return os.path.abspath(self.config.model_path)

        # 2. Local cache directory check
        os.makedirs(self.config.cache_dir, exist_ok=True)
        local_file = os.path.join(self.config.cache_dir, self.config.model_filename)
        if os.path.exists(local_file) and os.path.getsize(local_file) > 1024 * 1024:
            return os.path.abspath(local_file)

        # 3. Auto-download from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to auto-download GGUF weights. "
                "Install it via: pip install huggingface_hub"
            )

        print(f"\n[Narrator] Downloading {self.config.model_filename} from {self.config.model_repo}...")
        print(f"[Narrator] Destination: {self.config.cache_dir} (~2.1 GB, one-time download)...")

        downloaded_path = hf_hub_download(
            repo_id=self.config.model_repo,
            filename=self.config.model_filename,
            local_dir=self.config.cache_dir
        )
        return os.path.abspath(downloaded_path)

    def _build_system_prompt(self) -> str:
        """Return the strict non-causal grounding system prompt."""
        return self.SYSTEM_PROMPT.strip()

    def _build_user_prompt(self, record: Dict[str, Any]) -> str:
        """
        Format the case record into a structured, factual diagnostic prompt.
        """
        ticker = record.get('ticker', 'TARGET')
        target_date = record.get('target_date', 'N/A')
        pred = record.get('prediction', {})
        direction = pred.get('trend_direction', 'N/A')
        prob = pred.get('probability', 0.5)
        logit = pred.get('logit', 0.0)
        confidence = pred.get('confidence', 0.0)

        gt = record.get('ground_truth', {})
        gt_info = ""
        if gt:
            gt_dir = gt.get('direction', 'N/A')
            gt_match = "CORRECT" if pred.get('predicted_class') == gt.get('label') else "INCORRECT"
            gt_info = f"\nGROUND TRUTH: {gt_dir} | OUTCOME: {gt_match}"

        # 1. Feature Drivers
        attr = record.get('attribution', {})
        top_features = attr.get('top_magnitude_features', [])
        feat_lines = []
        for f in top_features[:5]:
            feat_name = f.get('feature', 'N/A')
            imp = f.get('importance_score', 0.0)
            sign_dir = f.get('direction', 'NEUTRAL')
            signed_val = f.get('signed_impact', 0.0)
            feat_lines.append(f"  - {feat_name}: importance={imp:.3f}, impact={signed_val:+.3f} ({sign_dir})")
        feat_str = "\n".join(feat_lines) if feat_lines else "  - None reported"

        # Bullish vs Bearish breakdown
        top_bullish = [b['feature'] for b in attr.get('top_bullish_features', [])[:3]]
        top_bearish = [b['feature'] for b in attr.get('top_bearish_features', [])[:3]]
        bullish_str = ", ".join(top_bullish) if top_bullish else "None"
        bearish_str = ", ".join(top_bearish) if top_bearish else "None"

        # 2. Temporal dynamics
        recency_ratio = attr.get('temporal_recency_ratio', 1.0)

        # 3. MVCA Views
        views = record.get('views', {})
        views_ranking = views.get('importance_ranking', ['ST', 'DS', 'DT'])
        v_interv = views.get('interventions', {})
        st_delta = v_interv.get('ST', {}).get('delta_logit', 0.0)
        ds_delta = v_interv.get('DS', {}).get('delta_logit', 0.0)
        dt_delta = v_interv.get('DT', {}).get('delta_logit', 0.0)

        # 4. Graph channels & peers
        graph = record.get('graph', {})
        static_delta = graph.get('static_prior_ablation_delta_logit', 0.0)
        adp_delta = graph.get('adaptive_graph_ablation_delta_logit', 0.0)
        intrinsic_share = attr.get('target_stock_intrinsic_share', 1.0)
        top_peers = graph.get('top_influential_peer_tickers', [])
        peer_lines = [f"{p.get('ticker', 'PEER')} (|Δlogit|={p.get('delta_logit', 0.0):.3f})" for p in top_peers[:3]]
        peer_str = ", ".join(peer_lines) if peer_lines else "None reported"

        # 5. Convergence & Stability
        conv = attr.get('convergence', {})
        conv_delta = conv.get('convergence_delta', 0.0)
        is_converged = conv.get('is_converged', True)
        steps_used = conv.get('steps_used', 50)
        quad = conv.get('quadrature_method', 'gauss_legendre')

        prompt = f"""Analyze this ST-GRAD stock prediction diagnostic record and synthesize a comprehensive 500-800 word analyst explanation:

TARGET STOCK: {ticker}
TARGET DATE: {target_date}
PREDICTION: {direction} (Probability: {prob:.1%}, Logit: {logit:+.3f}, Confidence: {confidence:.1%}){gt_info}

TOP TECHNICAL FEATURE DRIVERS (Integrated Gradients magnitude):
{feat_str}
Top Bullish Indicators: {bullish_str}
Top Bearish Indicators: {bearish_str}

TEMPORAL PROFILE (30-Day Trading Window):
- Temporal Recency Ratio: {recency_ratio:.2f}x (attribution in trailing 5 days vs leading 5 days)

MVCA LATENT VIEW COUNTERFACTUAL INTERVENTIONS:
- Spatio-Temporal View (ST): |Δlogit| = {abs(st_delta):.3f}
- Feature-Spatial View (DS): |Δlogit| = {abs(ds_delta):.3f}
- Feature-Temporal View (DT): |Δlogit| = {abs(dt_delta):.3f}
- View Reliance Ranking: {" > ".join(views_ranking)}

GRAPH CHANNEL DISENTANGLEMENT & SPATIAL DIFFUSION:
- Static Prior Graph Ablation: |Δlogit| = {abs(static_delta):.3f}
- Adaptive Dynamic Graph Ablation: |Δlogit| = {abs(adp_delta):.3f}
- Target Stock Intrinsic Share: {intrinsic_share:.1%} (Peer diffusion share: {1.0 - intrinsic_share:.1%})
- Top Influential Peers: {peer_str}

NUMERICAL CONVERGENCE & STABILITY:
- Integrated Gradients Completeness Delta: {conv_delta:.4f} ({'Converged' if is_converged else 'Approximate'})
- Quadrature Integration Steps: {steps_used} steps ({quad})

Please generate the structured analyst narrative following the five required section headers."""

        return prompt.strip()

    def generate_narrative(
        self,
        record: Dict[str, Any],
        stgrad_adapter: Optional[Any] = None
    ) -> str:
        """
        Generate a detailed natural language explanation from a case record.
        Implements sequential GPU memory swapping to respect hardware limits.
        """
        try:
            import llama_cpp
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for local LLM narrative generation.\n"
                "Install it using: pip install llama-cpp-python"
            )

        # 1. Sequential GPU Swap: Move ST-GRAD model to CPU if on CUDA
        original_device = None
        if stgrad_adapter is not None:
            try:
                # Detect current device of parameters
                param = next(stgrad_adapter.parameters(), None)
                if param is not None and param.is_cuda:
                    original_device = param.device
                    stgrad_adapter.to(torch.device("cpu"))
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Moved ST-GRAD model to CPU to free VRAM for language model.")
            except Exception as e:
                logger.warning(f"Could not swap ST-GRAD model to CPU: {e}")

        # 2. Resolve or download model path
        model_path = self._ensure_model_downloaded()

        # 3. Initialize Llama engine
        print(f"[Narrator] Loading Qwen2.5-3B-Instruct from {os.path.basename(model_path)}...")
        try:
            llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=self.config.n_gpu_layers,
                n_ctx=self.config.n_ctx,
                verbose=False
            )
        except Exception as e:
            # Fallback to CPU execution if CUDA memory allocation failed
            logger.warning(f"GPU offloading failed ({e}). Falling back to CPU execution.")
            print("[Narrator] GPU offloading failed, falling back to CPU execution...")
            llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=0,
                n_ctx=self.config.n_ctx,
                verbose=False
            )

        # 4. Construct prompts and generate completion
        sys_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(record)

        print("[Narrator] Synthesizing narrative reasoning in plain English...")
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        narrative = response["choices"][0]["message"]["content"].strip()

        # 5. Clean up LLM to free VRAM
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 6. Restore ST-GRAD model back to original GPU device
        if stgrad_adapter is not None and original_device is not None:
            stgrad_adapter.to(original_device)
            logger.info(f"Restored ST-GRAD model back to {original_device}.")

        return narrative
