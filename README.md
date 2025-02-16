to build: docker build -t paraphraser:latest .

to run: docker run -e INPUT_TEXT="How can I rephrase this sentence?" -v $(pwd)/outputs:/outputs paraphraser:latest
