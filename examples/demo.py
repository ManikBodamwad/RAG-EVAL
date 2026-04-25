"""
RAG-EVAL Demo Script

Simulates a RAG pipeline evaluation where one of the answers is purposely hallucinated,
showing how rag-eval catches regressions in CI/CD.
"""
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag_pipeline import RAGPipeline
from rag_eval.evaluator import RagEvaluator
from rag_eval.reporter import Reporter

console = Console()

# --- 1. Monkeypatch the RAG Generator to Hallucinate ---
original_generate = RAGPipeline.generate

def hallucinated_generate(self, question: str, contexts: list[str]):
    # If the user asks about RLHF, we output hallucinated nonsense
    if "RLHF" in question:
        fake_answer = (
            "RLHF (Radical Linear Hover Flight) is an aerospace technology developed "
            "by SpaceX. It allows drones to hover indefinitely using compressed "
            "nitrogen propulsion, completely unrelated to AI or language models."
        )
        return fake_answer, 120, 35, 250.0
    
    # Otherwise, be a good AI
    return original_generate(self, question, contexts)

RAGPipeline.generate = hallucinated_generate

# --- 2. Demo Orchestration ---
def main():
    console.print(Panel.fit("[bold cyan]🤖 RAG-EVAL Simulated Quality Gate[/bold cyan]\n"
                            "This demo intercepts the mock RAG pipeline and forces it to hallucinate\n"
                            "an answer about 'RLHF' to show how rag-eval catches quality regressions.",
                            border_style="cyan"))
    
    # Suppress all the noisy logs so the demo looks clean
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in ["rag_eval.evaluator", "app.rag_pipeline", "ragas", "httpx", "litellm", "faiss.loader"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    evaluator = RagEvaluator(config_path=project_root / "eval_config.yaml")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task1 = progress.add_task("[yellow]Booting up AI infrastructure and indexing Vector DB...", total=None)
        time.sleep(1.5)
        
        progress.update(task1, description="[blue]Fetching Golden Dataset from Hugging Face 'manikbodamwad/rag-eval-golden'...")
        time.sleep(1.5)
        
        progress.update(task1, description="[cyan]Running Mock RAG App to generate responses (Watch out for the hallucination!)...")
        time.sleep(1.5)

        progress.update(task1, description="[magenta]Using Groq LLaMA-3.3-70b as LLM Judge to calculate Ragas Metrics... (This takes ~15 seconds)")
        
        # Actually run the evaluation
        t0 = time.time()
        try:
            result = evaluator.evaluate()
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]❌ Failed to run evaluation: {e}[/bold red]")
            console.print("Make sure you have GROQ_API_KEY exported in your environment.")
            sys.exit(1)
        t1 = time.time()
        
        progress.update(task1, description=f"[green]Evaluation complete in {t1-t0:.1f}s!")
        time.sleep(1.0)
    
    # 3. Present Results Beautifully
    console.print("\n[bold green]✅ Pipeline Execution Finished![/bold green]")
    console.print("[dim]rag-eval analyzed the semantic alignment of all answers against the ground truth.[/dim]\n")
    
    reporter = Reporter()
    reporter.print_rich_table(result)

    if not result.overall_pass:
        console.print("\n[bold red]🚨 Quality Gate Triggered![/bold red]")
        console.print("The pipeline's hallucination about 'RLHF' caused the [bold white]Faithfulness[/bold white] and [bold white]Answer Correctness[/bold white] "
                      "metrics to crash below the threshold. In CI/CD, this PR would be blocked from merging.")
    else:
        console.print("\n[bold green]✅ All checks passed![/bold green] (Wait, the hallucination should have failed!)")

if __name__ == "__main__":
    main()
