class FastText {
    constructor(vectorSize = 50, windowSize = 5, minCount = 1, nGramRange = [3, 6]) {
        this.vectorSize = vectorSize;
        this.windowSize = windowSize;
        this.minCount = minCount;
        this.nGramRange = nGramRange;
        this.vocabulary = {};
        this.wordVectors = {};
        this.nGramVectors = {};
    }

    tokenize(text) {
        return text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(word => word.length > 0);
    }

    generateNGrams(word) {
        const nGrams = [];
        const [minN, maxN] = this.nGramRange;
        for (let n = minN; n <= maxN; n++) {
            for (let i = 0; i <= word.length - n; i++) {
                nGrams.push(word.slice(i, i + n));
            }
        }
        return nGrams;
    }

    buildVocabulary(texts) {
        const tokenCounts = {};
        texts.forEach(text => {
            const tokens = this.tokenize(text);
            tokens.forEach(token => {
                if (!tokenCounts[token]) {
                    tokenCounts[token] = 0;
                }
                tokenCounts[token]++;
            });
        });

        Object.keys(tokenCounts).forEach(token => {
            if (tokenCounts[token] >= this.minCount) {
                this.vocabulary[token] = { count: tokenCounts[token], nGrams: this.generateNGrams(token) };
                this.wordVectors[token] = Array.from({ length: this.vectorSize }, () => Math.random() - 0.5);
                this.vocabulary[token].nGrams.forEach(nGram => {
                    if (!this.nGramVectors[nGram]) {
                        this.nGramVectors[nGram] = Array.from({ length: this.vectorSize }, () => Math.random() - 0.5);
                    }
                });
            }
        });
    }

    trainWordVectors(texts) {
        texts.forEach(text => {
            const tokens = this.tokenize(text);
            tokens.forEach((token, index) => {
                const context = tokens.slice(Math.max(0, index - this.windowSize), index).concat(tokens.slice(index + 1, index + 1 + this.windowSize));
                context.forEach(contextToken => {
                    this.updateVectors(token, contextToken);
                });
            });
        });
    }

    updateVectors(targetToken, contextToken) {
        const targetVector = this.wordVectors[targetToken];
        const contextVector = this.wordVectors[contextToken];
        const gradient = targetVector.map((value, index) => value - contextVector[index]);
        targetVector.forEach((value, index) => {
            targetVector[index] -= 0.01 * gradient[index];
            contextVector[index] += 0.01 * gradient[index];
        });
    }

    createWordVector(word) {
        if (!this.vocabulary[word]) {
            return Array(this.vectorSize).fill(0);
        }
        const wordVector = this.wordVectors[word];
        const nGramVectors = this.vocabulary[word].nGrams.map(nGram => this.nGramVectors[nGram] || Array(this.vectorSize).fill(0));
        const avgNGramVector = nGramVectors[0].map((_, index) => nGramVectors.reduce((sum, vector) => sum + vector[index], 0) / nGramVectors.length);
        return wordVector.map((value, index) => value + avgNGramVector[index]);
    }

    fit(texts) {
        this.buildVocabulary(texts);
        this.trainWordVectors(texts);
    }

    transform(text) {
        const tokens = this.tokenize(text);
        const vectors = tokens.map(token => this.createWordVector(token));
        return vectors[0].map((_, colIndex) => vectors.map(row => row[colIndex]).reduce((a, b) => a + b, 0) / vectors.length);
    }
}

// Example usage:
const texts = [
    'The quick brown fox jumps over the lazy dog',
    'A journey of a thousand miles begins with a single step',
    'To be or not to be, that is the question',
    'All that glitters is not gold'
];

const fastText = new FastText();
fastText.fit(texts);

texts.forEach(text => {
    const vector = fastText.transform(text);
    console.log(`Vector for "${text}":`, vector);
});
