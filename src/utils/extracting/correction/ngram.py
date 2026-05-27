from collections import Counter
from typing import List

class BigramLanguageModel:

    def __init__(self):
        self.bigram_counts = Counter()
        self.unigram_counts = Counter()

    def train(self, corpus: List[str]):
        for sentence in corpus:
            words = sentence.lower().split()
            if not words:
                continue
            for i in range(len(words) - 1):
                w1 = words[i]
                w2 = words[i + 1]
                self.bigram_counts[(w1, w2)] += 1
                self.unigram_counts[w1] += 1
            if words:
                self.unigram_counts[words[-1]] += 1

    def bigram_score(self, prev_word, word):
        bigram_count = self.bigram_counts[(prev_word, word)]
        unigram_count = self.unigram_counts[prev_word]
        if unigram_count == 0:
            return 0.0
        return bigram_count / unigram_count

    def load_corpus(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.train(lines)

    def rank_candidates(self, next_word, candidates):
        best_candidate = None
        best_score = -1
        for candidate in candidates:
            score = self.bigram_score(next_word.lower(), candidate.lower())
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_score == 0:
            return None
        return best_candidate
    
model = BigramLanguageModel()
model.load_corpus("data/corpus.txt")
print(f"Tổng số cặp từ (Bigrams) học được: {len(model.bigram_counts)}")
