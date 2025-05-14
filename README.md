# Software-Based Communication Protocol to Address Multi-Agent Coordination Failures in AI-Driven Vehicles


## Usage

Before running the fine-tuning scripts, set your OpenAI API key in your environment:

export OPENAI_API_KEY=your_api_key

Then, run the finetuning script for the desired region:

python finetuning.py usa # or 'india' for the other region

After fine-tuning the models, use the model IDs to run experiments:

python run_experiment.py "model_id_for_usa" "model_id_for_india" --iterations 100

Replace `model_id_for_usa` and `model_id_for_india` with the actual model IDs you get after fine-tuning.


