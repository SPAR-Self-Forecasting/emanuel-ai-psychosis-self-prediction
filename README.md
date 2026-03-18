# AI Psychosis Self-Prediction Experiment

Can AI models predict how well they will handle users experiencing psychosis? This experiment measures **self-prediction** vs **cross-prediction** ability: does a model know its own therapeutic behavior better than it knows other models'?

## Background

Based on the [AI Psychosis red-teaming methodology](https://github.com/tim-hua-01/ai-psychosis) by Tim Hua, which tests how frontier AI models respond to users displaying increasingly severe psychotic symptoms.

## Experiment Design

### Stage 1: Ground Truth Collection

A red-teaming AI (Grok-3) roleplays as 9 different psychosis characters, sending 12 escalating messages to each target model. An independent judge (Grok-3) grades each response on 11 therapeutic criteria (pushback, delusion confirmation, collaborative stance, etc.).

- **9 characters** × **5 target models** × **12 turns** × **11 judge metrics**
- Models: GPT-5, GPT-oss-20b, Grok-3, Grok-4, Gemini 2.5 Flash

### Stage 2: Self-Prediction

Each model receives a description of a character and a judge criterion, and predicts the score it would receive on the final turn. The same prompt is used for all models — no model names are revealed. Each model always predicts "how would I behave."

The **MSE matrix** (predictor × target) is built by comparing each predictor's forecasts against every target's ground truth:
- **Diagonal** = self-prediction (predictor == target)
- **Off-diagonal** = cross-prediction (predictor != target)

We compute both **raw MSE** and **z-scored MSE** (both predictions and ground truth standardized to mean 0, variance 1) to separate calibration from discriminative ability.

## Results

### Raw MSE
Self-prediction mean MSE: **2.69** vs cross-prediction: **3.21** — self appears better.

### Z-scored MSE
After standardization: self **0.90** vs cross **0.89** — the advantage disappears entirely.

### Explanation
All models predict scores of ~3.9-4.0 for themselves regardless of actual performance. Models that actually score ~4 (GPT-5) appear to have good self-knowledge, but it's just that their actual score matches the universal overconfident self-assessment.

| Model | Actual Mean | Self-Predicted Mean | Overconfidence |
|-------|-------------|---------------------|----------------|
| GPT-5 | 3.94 | 4.04 | +0.10 |
| Grok-4 | 3.52 | 3.92 | +0.41 |
| Gemini 2.5 Flash | 2.54 | 2.90 | +0.36 |
| GPT-oss-20b | 2.81 | 3.97 | +1.16 |
| Grok-3 | 2.77 | 3.95 | +1.19 |

## Setup

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

```bash
# Clone with submodule
git clone https://github.com/SPAR-Self-Forecasting/ai-psychosis-self-prediction.git
cd ai-psychosis-self-prediction

# Clone the upstream red-teaming repo
git clone https://github.com/tim-hua-01/ai-psychosis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install inspect-ai anthropic openai pandas rich tqdm python-dotenv pytz scipy matplotlib seaborn

# Set your API key
echo "OPENROUTER_API_KEY=your-key-here" > ai-psychosis/.env
```

### Running

#### Stage 1: Collect ground truth scores

```bash
python stage1_run.py
```

This runs each model × character combination (5 × 9 = 45 rollouts, each 12 turns). Takes ~2 hours and costs ~$100 with N_REPETITIONS=1.

Configure in `stage1_run.py`:
- `N_REPETITIONS`: Number of rollouts per combination (1 for testing, 100 for full experiment)
- `TARGET_MODELS`: List of models to evaluate
- `MAX_CONNECTIONS`: Concurrent API calls (default 5)

Results saved to `stage1_results/`.

#### Stage 2: Collect predictions and compute MSE matrix

```bash
python stage2_predict.py
```

This asks each model to predict its own scores (5 × 9 × 11 = 495 API calls). Takes ~10 minutes, costs ~$5-10.

Results saved to `stage2_results/`.

#### Generate plots

```bash
python plot_results.py
```

Generates the MSE heatmaps and calibration plot in `stage2_results/`.

## Project Structure

```
├── stage1_run.py                 # Stage 1: red-teaming + judging
├── stage2_predict.py             # Stage 2: self/cross prediction + MSE
├── stage2_prompt_template.txt    # Prompt template for predictions
├── plot_results.py               # Generate all plots
├── ai-psychosis/                 # Upstream red-teaming repo (clone separately)
│   ├── characters/               # 9 psychosis character descriptions
│   ├── red_team_prompt.txt       # Red team agent instructions
│   ├── grader_prompt.txt         # Judge evaluation criteria
│   └── redteaming_systematic.py  # Core pipeline
├── stage1_results/               # Ground truth data
│   ├── raw_results_*.csv         # Turn-level judge scores
│   ├── probability_table_*.csv   # Model × character × metric means
│   └── pivot_tables/             # Per-metric pivot tables
└── stage2_results/               # Prediction data
    ├── predictions_*.csv         # Raw model predictions
    ├── ground_truth_*.csv        # Ground truth for comparison
    ├── mse_matrix_*.csv          # MSE matrix
    └── *.png                     # Plots
```

## Extending

- **More models**: Add model identifiers to `TARGET_MODELS` in both stage files
- **More repetitions**: Set `N_REPETITIONS = 100` in `stage1_run.py` for robust probability estimates
- **Different metrics**: Edit `JUDGE_METRICS` in `stage2_predict.py`
