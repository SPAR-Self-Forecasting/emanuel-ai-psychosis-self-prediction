#!/usr/bin/env python3
"""
Stage 1: Run AI Psychosis red-teaming experiment.

For each {evaluated_model} x {psychosis_character}, run N rollouts,
judge each, and store a 3D table (model x character x judge_metric) of mean scores.
"""

import asyncio
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add the ai-psychosis repo to path
REPO_DIR = Path(__file__).parent / "ai-psychosis"
sys.path.insert(0, str(REPO_DIR))

load_dotenv(REPO_DIR / ".env")

from redteaming_systematic import (
    AutomatedRedTeaming,
    load_characters,
    format_prompt_for_character,
    console,
)
from rich.panel import Panel
from rich.text import Text

# Models routed through OpenRouter — subset from the Tim Hua paper
# that completed full runs (all 9 characters × 12 turns)
TARGET_MODELS = [
    "openrouter/openai/gpt-5",
    "openrouter/openai/gpt-oss-20b",
    "openrouter/x-ai/grok-3",
    "openrouter/x-ai/grok-4",
    "openrouter/google/gemini-2.5-flash",
]

# Same as the paper: Grok for red-teaming and grading
REDTEAM_MODEL = "openrouter/x-ai/grok-3"
GRADER_MODEL = "openrouter/x-ai/grok-3"

N_TURNS = 12
N_REPETITIONS = 1  # Start with 1; set to 100 for full experiment
MAX_CONNECTIONS = 5  # Conservative to avoid rate limits

# Judge metrics that are numeric scores (for probability table)
NUMERIC_METRICS = [
    "pushback_rating",
    "delusion_confirmation_rating",
    "collaborative_stance",
    "focus_on_distress",
    "therapeutic_validation",
    "non_confrontational_curiosity",
    "normalizing_vs_exceptionalizing",
    "avoiding_delusional_amplification",
    "promoting_safety_and_grounding",
    "gentle_reality_testing",
    "fostering_real_world_connections",
]

# Binary metric derived from mention_professional_help
BINARY_METRICS = ["mention_professional_help"]

OUTPUT_DIR = Path(__file__).parent / "stage1_results"


async def run_single(
    target_model: str,
    character_name: str,
    character_scenario: str,
    red_team_prompt: str,
    grader_prompt: str,
    rep_id: int,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame | None:
    """Run a single rollout and return the grade DataFrame."""
    async with semaphore:
        session = f"{character_name}_{target_model.replace('/', '-')}_rep{rep_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            console.print(
                f"  [cyan]Starting[/cyan] {target_model} x {character_name} (rep {rep_id})"
            )
            formatted_prompt = format_prompt_for_character(
                red_team_prompt, character_name, character_scenario
            )
            rt = AutomatedRedTeaming(
                target_model_name=target_model,
                redteam_model_name=REDTEAM_MODEL,
                redteam_prompt=formatted_prompt,
                patient_name=character_name,
                grader_prompt=grader_prompt,
                grader_model_name=GRADER_MODEL,
            )
            await rt.run_redteaming(n_turns=N_TURNS)

            # Save transcript
            transcript_dir = OUTPUT_DIR / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            rt.save_conversation_to_markdown("target", session)

            if rt.grade_df is not None:
                rt.grade_df["repetition"] = rep_id
                console.print(
                    f"  [green]Done[/green] {target_model} x {character_name} (rep {rep_id})"
                )
                return rt.grade_df
            else:
                console.print(
                    f"  [red]No grades[/red] {target_model} x {character_name} (rep {rep_id})"
                )
                return None
        except Exception as e:
            console.print(
                f"  [red]Error[/red] {target_model} x {character_name} (rep {rep_id}): {e}"
            )
            return None


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts from repo
    with open(REPO_DIR / "red_team_prompt.txt") as f:
        red_team_prompt = f.read()
    with open(REPO_DIR / "grader_prompt.txt") as f:
        grader_prompt = f.read()

    # Load characters
    os.chdir(REPO_DIR)  # so load_characters finds the characters/ dir
    characters = load_characters()

    console.print(
        Panel.fit(
            Text(
                f"Stage 1: {len(TARGET_MODELS)} models x {len(characters)} characters x {N_REPETITIONS} reps = "
                f"{len(TARGET_MODELS) * len(characters) * N_REPETITIONS} total rollouts",
                style="bold magenta",
            ),
            border_style="magenta",
        )
    )

    semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
    tasks = []

    for target_model in TARGET_MODELS:
        for char_name, char_scenario in characters.items():
            for rep in range(N_REPETITIONS):
                tasks.append(
                    run_single(
                        target_model,
                        char_name,
                        char_scenario,
                        red_team_prompt,
                        grader_prompt,
                        rep,
                        semaphore,
                    )
                )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful DataFrames
    all_dfs = []
    successes = 0
    failures = 0
    for r in results:
        if isinstance(r, pd.DataFrame):
            all_dfs.append(r)
            successes += 1
        else:
            failures += 1

    console.print(f"\n[bold]Results: {successes} successes, {failures} failures[/bold]")

    if not all_dfs:
        console.print("[red]No successful runs. Exiting.[/red]")
        return

    # Save raw turn-level results
    raw_df = pd.concat(all_dfs, ignore_index=True)
    raw_path = OUTPUT_DIR / f"raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    raw_df.to_csv(raw_path, index=False)
    console.print(f"[green]Raw results saved to {raw_path}[/green]")

    # Build 3D probability table: model x character x metric
    # For each (model, character, metric), compute mean across all turns and repetitions
    # Also compute mention_professional_help as fraction of "Yes" responses
    raw_df["mention_professional_help_binary"] = (
        raw_df["mention_professional_help"].str.startswith("Yes").astype(float)
    )

    all_metrics = NUMERIC_METRICS + ["mention_professional_help_binary"]

    prob_rows = []
    for model in raw_df["target_model"].unique():
        for char in raw_df["patient_name"].unique():
            mask = (raw_df["target_model"] == model) & (raw_df["patient_name"] == char)
            subset = raw_df.loc[mask]
            if subset.empty:
                continue
            row = {"model": model, "character": char}
            for metric in all_metrics:
                if metric in subset.columns:
                    row[metric] = subset[metric].mean()
            prob_rows.append(row)

    prob_df = pd.DataFrame(prob_rows)
    prob_path = OUTPUT_DIR / f"probability_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    prob_df.to_csv(prob_path, index=False)
    console.print(f"[green]Probability table saved to {prob_path}[/green]")

    # Also save as a more structured 3D format (one CSV per metric, models as rows, characters as cols)
    pivot_dir = OUTPUT_DIR / "pivot_tables"
    pivot_dir.mkdir(exist_ok=True)
    for metric in all_metrics:
        if metric in prob_df.columns:
            pivot = prob_df.pivot(index="model", columns="character", values=metric)
            pivot.to_csv(pivot_dir / f"{metric}.csv")

    console.print(f"[green]Pivot tables saved to {pivot_dir}/[/green]")
    console.print("[bold green]Stage 1 complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
