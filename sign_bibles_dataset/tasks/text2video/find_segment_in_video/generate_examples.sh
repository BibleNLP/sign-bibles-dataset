#!/bin/bash
set -euo pipefail

num_predictions=40

# generate "ground truth"
python generate_funny_examples.py --video-id-count 20 --num-segments 100 --total-frames 10000 --output-path example_data/example_queries.csv

# Generate Predictions
# python evaluation/predict_by_random_guess.py --input-path examples/example_queries.csv --output-path examples/random_predictions.csv
# python evaluation/predict_by_cheating.py --ground-truth examples/example_queries.csv --output-path examples/wrong_predictions.csv  --recall 0 --precision 0 --num-predictions "$num_predictions"
# python evaluation/predict_by_cheating.py --ground-truth examples/example_queries.csv --output-path examples/halfrecallhalfprecision_predictions.csv --recall 0.5 --precision 0.5 --num-predictions "$num_predictions"
# python evaluation/predict_by_cheating.py --ground-truth examples/example_queries.csv --output-path examples/lowrecallhighprecision_predictions.csv --recall 0.1 --precision 1.0 --num-predictions "$num_predictions"
# python evaluation/predict_by_cheating.py --ground-truth examples/example_queries.csv --output-path examples/highrecalllowprecision_predictions.csv --recall 1.0 --precision 0.1 --num-predictions "$num_predictions"
# python evaluation/predict_by_cheating.py --ground-truth examples/example_queries.csv --output-path examples/perfect_predictions.csv --recall 1.0 --precision 1.0 --num-predictions "$num_predictions"

