import logging
import json
from typing import Optional
from pathlib import Path

import pandas as pd
import difflib
import numpy as np
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
    ContextRecall,
)
from datasets import Dataset

from src.agents_src.tools.rag_qa_tool import rag_query_tool_sync
from src.rag_doc_ingestion.config.doc_ingestion_settings import DocIngestionSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics."""

    def __init__(self, eval_dataset_path: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            eval_dataset_path: Path to evaluation dataset JSON file.
                             If None, uses default path.
        """
        if eval_dataset_path is None:
            eval_dataset_path = (
                Path(__file__).parent / "eval_dataset.json"
            )
        self.eval_dataset_path = Path(eval_dataset_path)
        logger.info(f"Using eval dataset: {self.eval_dataset_path}")

    def load_eval_dataset(self) -> list:
        """Load evaluation dataset from JSON file."""
        if not self.eval_dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found at {self.eval_dataset_path}"
            )
        with open(self.eval_dataset_path, "r") as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} evaluation samples")
        return dataset

    def run_rag_pipeline(self, query: str) -> dict:
        """
        Run the RAG pipeline for a query.

        Args:
            query: Input query string

        Returns:
            dict with 'answer' and 'source_files' keys
        """
        try:
            result = rag_query_tool_sync(query)
            logger.debug(f"RAG result for '{query}': {result}")
            return result
        except Exception as e:
            logger.error(f"Error running RAG pipeline: {e}")
            return {"answer": "", "source_files": []}

    def prepare_evaluation_data(self, eval_samples: list) -> dict:
        """
        Prepare data for RAGAS evaluation.

        Args:
            eval_samples: List of evaluation samples from dataset

        Returns:
            dict with queries, contexts, responses, and ground_truths
        """
        queries = []
        contexts = []
        responses = []
        ground_truths = []

        for sample in eval_samples:
            user_input = sample.get("user_input", "")
            reference = sample.get("reference", "")
            retrieval_ground_truth = sample.get("retrieval_ground_truth", [])

            # Run RAG pipeline
            rag_result = self.run_rag_pipeline(user_input)
            answer = rag_result.get("answer", "")

            queries.append(user_input)
            contexts.append([retrieval_ground_truth])  # RAGAS expects nested list
            responses.append(answer)
            ground_truths.append(reference)

            logger.info(
                f"Processed query: '{user_input[:50]}...' -> "
                f"Answer: '{answer[:50]}...'"
            )

        return {
            "question": queries,
            "contexts": contexts,
            "answer": responses,
            "ground_truths": ground_truths,
        }

    def evaluate(self, eval_samples: Optional[list] = None) -> pd.DataFrame:
        """
        Run RAGAS evaluation on RAG pipeline.

        Args:
            eval_samples: List of evaluation samples. If None, loads from file.

        Returns:
            pd.DataFrame with evaluation results for each sample
        """
        if eval_samples is None:
            eval_samples = self.load_eval_dataset()

        logger.info("Preparing evaluation data...")
        eval_data = self.prepare_evaluation_data(eval_samples)

        logger.info("Creating RAGAS dataset...")
        ragas_dataset = Dataset.from_dict(eval_data)

        logger.info("Running RAGAS evaluation with metrics: answer_relevancy, "
                    "context_relevance, context_recall (faithfulness optional)")

        # Metrics will be instantiated dynamically below. We attempt to build
        # optional llm and embeddings objects and then instantiate candidate
        # metric classes using available resources.

        # Prepare optional llm and embeddings objects if possible
        llm = None
        embeddings = None
        try:
            import os
            # Prefer ragas-provided wrappers
            try:
                from ragas.llm import OpenAI  # type: ignore
                from ragas.embeddings import SentenceTransformerEmbeddings  # type: ignore
                api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
                if api_key:
                    llm = OpenAI(api_key=api_key)
                    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    logger.info("Using ragas OpenAI + SentenceTransformerEmbeddings for advanced metrics")
            except Exception:
                # Fallback to local sentence-transformers embeddings if available
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore

                    class SimpleEmbeddings:
                        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                            self._model = SentenceTransformer(model_name)

                        def embed(self, texts):
                            return self._model.encode(texts, show_progress_bar=False)

                    embeddings = SimpleEmbeddings()
                    # Try to create an LLM wrapper if ragas OpenAI is available
                    try:
                        from ragas.llm import OpenAI  # type: ignore
                        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
                        if api_key:
                            llm = OpenAI(api_key=api_key)
                            logger.info("Using OpenAI wrapper with local sentence-transformers embeddings for advanced metrics")
                    except Exception:
                        logger.debug("No ragas LLM wrapper available for advanced metrics")
                except Exception:
                    logger.debug("sentence-transformers not installed; embeddings unavailable for advanced metrics")
        except Exception as e:
            logger.debug("Error preparing llm/embeddings: %s", e)

        # Candidate metric classes in preferred order
        candidate_metrics = [
            Faithfulness,
            AnswerRelevancy,
            ContextRelevance,
            ContextRecall,
        ]

        def try_create(metric_cls):
            try:
                return metric_cls()
            except TypeError:
                # Try combinations of available resources
                try:
                    if llm is not None and embeddings is not None:
                        return metric_cls(llm=llm, embeddings=embeddings)
                except TypeError:
                    pass
                try:
                    if llm is not None:
                        return metric_cls(llm=llm)
                except TypeError:
                    pass
                try:
                    if embeddings is not None:
                        return metric_cls(embeddings=embeddings)
                except TypeError:
                    pass
            except Exception as e:
                logger.debug("Unexpected error instantiating %s: %s", metric_cls, e)
            logger.warning("Skipping metric %s: could not instantiate with available resources", metric_cls.__name__)
            return None

        metrics = []
        for mcls in candidate_metrics:
            inst = try_create(mcls)
            if inst is not None:
                metrics.append(inst)

        if not metrics:
            logger.warning("No RAGAS metrics could be instantiated. Falling back to simple local evaluation.")
            return self.fallback_evaluate(eval_samples)

        try:
            result = evaluate(ragas_dataset, metrics=metrics)
        except Exception as e:
            logger.warning("RAGAS evaluation failed (%s). Falling back to simple local evaluation.", e)
            return self.fallback_evaluate(eval_samples)

        # Convert to DataFrame for easier analysis
        results_df = result.to_pandas()
        logger.info("Evaluation completed successfully")
        return results_df

    def fallback_evaluate(self, eval_samples: list) -> pd.DataFrame:
        """Simple fallback evaluation when RAGAS metrics are unavailable.

        Computes basic string-similarity metrics between generated answers
        and references using SequenceMatcher and basic token overlap. Also
        returns the raw answers for per-sample inspection.
        """
        rows = []

        # Detect optional libraries for enhanced metrics
        has_sacrebleu = False
        has_rouge = False
        has_st = False
        try:
            import sacrebleu  # type: ignore
            has_sacrebleu = True
        except Exception:
            logger.debug("sacrebleu not available; BLEU will be skipped")
        try:
            from rouge_score import rouge_scorer  # type: ignore
            has_rouge = True
        except Exception:
            logger.debug("rouge_score not avvailable; ROUGE-L will be skipped")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            has_st = True
        except Exception:
            logger.debug("sentence-transformers not available; cosine similarity will be skipped")

        # If sentence-transformers is available, instantiate a model once
        st_model = None
        if has_st:
            try:
                st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.debug("Failed to load sentence-transformers model: %s", e)
                has_st = False
        for sample in eval_samples:
            query = sample.get("user_input", "")
            reference = sample.get("reference", "")
            # Run RAG to get the answer
            rag_result = self.run_rag_pipeline(query)
            answer = rag_result.get("answer", "")
            source_files = rag_result.get("source_files", [])

            # Similarity ratio (0..1)
            ratio = difflib.SequenceMatcher(None, reference, answer).ratio()

            # Token overlap (simple)
            ref_tokens = set(reference.lower().split())
            ans_tokens = set(answer.lower().split())
            if ref_tokens:
                token_overlap = len(ref_tokens & ans_tokens) / len(ref_tokens)
            else:
                token_overlap = 0.0

            # BLEU (0..1) via sacrebleu if available
            bleu_score = None
            if has_sacrebleu and answer.strip():
                try:
                    import sacrebleu  # type: ignore
                    # sacrebleu expects list of hypotheses and list of references
                    bleu = sacrebleu.corpus_bleu([answer], [[reference]])
                    bleu_score = float(bleu.score) / 100.0
                except Exception:
                    bleu_score = None

            # ROUGE-L F1 (0..1) if available
            rouge_l = None
            if has_rouge and answer.strip():
                try:
                    from rouge_score import rouge_scorer  # type: ignore
                    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                    sc = scorer.score(reference, answer)
                    rouge_l = sc.get("rougeL").fmeasure
                except Exception:
                    rouge_l = None

            # Cosine similarity of embeddings (0..1) if sentence-transformers is available
            cosine_sim = None
            if has_st and st_model and answer.strip():
                try:
                    ref_emb = st_model.encode(reference, convert_to_numpy=True)
                    ans_emb = st_model.encode(answer, convert_to_numpy=True)
                    # cosine similarity
                    denom = (np.linalg.norm(ref_emb) * np.linalg.norm(ans_emb))
                    if denom > 0:
                        cosine_sim = float(np.dot(ref_emb, ans_emb) / denom)
                    else:
                        cosine_sim = 0.0
                except Exception:
                    cosine_sim = None

            rows.append({
                "question": query,
                "answer": answer,
                "ground_truths": reference,
                "similarity_ratio": ratio,
                "token_overlap": token_overlap,
                "source_files": source_files,
                "bleu": bleu_score,
                "rouge_l": rouge_l,
                "cosine_sim": cosine_sim,
            })

        df = pd.DataFrame(rows)
        logger.info("Fallback evaluation completed (%d samples)", len(df))
        return df

    def print_results(self, results_df: pd.DataFrame) -> None:
        """
        Print evaluation results in a readable format.

        Args:
            results_df: DataFrame with evaluation results
        """
        print("\n" + "=" * 80)
        print("RAGAS Evaluation Results")
        print("=" * 80)

        # Overall metrics (average across all numeric columns)
        print("\n[Overall Metrics (Average)]")
        # Only consider numeric columns for averages (avoid lists/objects)
        numeric_cols = results_df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            print("  (no numeric metrics available)")
        else:
            for col in numeric_cols:
                avg_value = results_df[col].mean()
                print(f"  {col}: {avg_value:.4f}")

        # Per-sample results
        print("\n[Per-Sample Results]")
        print(results_df.to_string())

        # Summary statistics for numeric columns
        print("\n[Summary Statistics]")
        if numeric_cols:
            print(results_df[numeric_cols].describe())
        else:
            print("  (no numeric metrics to summarize)")

        print("\n" + "=" * 80)

    def save_results(self, results_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Save evaluation results to CSV and JSON.

        Args:
            results_df: DataFrame with evaluation results
            output_path: Output directory. If None, uses default.
        """
        if output_path is None:
            output_path = Path(__file__).parent / "results"
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        csv_file = output_path / "ragas_results.csv"
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to CSV: {csv_file}")

        # Save as JSON
        json_file = output_path / "ragas_results.json"
        results_df.to_json(json_file, orient="records", indent=2)
        logger.info(f"Results saved to JSON: {json_file}")

        # Save summary (only numeric columns)
        summary_file = output_path / "summary.txt"
        numeric_cols = results_df.select_dtypes(include=["number"]).columns.tolist()
        with open(summary_file, "w") as f:
            if not numeric_cols:
                f.write("(no numeric metrics available)\n")
            else:
                for col in numeric_cols:
                    avg_value = results_df[col].mean()
                    f.write(f"{col}: {avg_value:.4f}\n")
        logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main entry point for evaluation."""
    try:
        evaluator = RAGASEvaluator()
        results_df = evaluator.evaluate()
        evaluator.print_results(results_df)
        evaluator.save_results(results_df)
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
