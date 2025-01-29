class BagOfWords {
    constructor() {
        this.vocabulary = new Set();
        this.wordIndex = {};
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^a-z\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 0);
    }

    buildVocabulary(texts) {
        texts.forEach(text => {
            const tokens = this.tokenize(text);
            tokens.forEach(token => this.vocabulary.add(token));
        });

        this.vocabulary = Array.from(this.vocabulary);
        this.vocabulary.forEach((word, index) => {
            this.wordIndex[word] = index;
        });
    }

    createVector(text) {
        const tokens = this.tokenize(text);
        const vector = Array(this.vocabulary.length).fill(0);

        tokens.forEach(token => {
            const index = this.wordIndex[token];
            if (index !== undefined) {
                vector[index]++;
            }
        });

        return vector;
    }

    createVectors(texts) {
        this.buildVocabulary(texts);
        return texts.map(text => this.createVector(text));
    }
}

// Example usage:
const texts = [
    'The quick brown fox jumps over the lazy dog',
    'A journey of a thousand miles begins with a single step',
    'To be or not to be, that is the question',
    'All that glitters is not gold'
];

const bow = new BagOfWords();
const vectors = bow.createVectors(texts);

console.log('Vocabulary:', bow.vocabulary);
console.log('Vectors:', vectors);
