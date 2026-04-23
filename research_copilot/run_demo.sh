#!/bin/zsh
python cli.py fetch "vision transformers" --max 5 --source arxiv
python cli.py summarize --input output/results.json --output output/summaries.json --provider ollama --model llama3.2
python cli.py insights --input output/summaries.json --output output/insights.json --topic "vision transformers" --provider ollama --model llama3.2 --print
python cli.py gaps --input output/papers_with_pdf.json --output output/gaps.json --topic "vision transformers" --provider ollama --model llama3.2 --print
python cli.py hypotheses --papers output/papers_with_pdf.json --insights output/insights.json --gaps output/gaps.json --output output/hypotheses.json --topic "vision transformers" --provider ollama --model llama3.2 --print
