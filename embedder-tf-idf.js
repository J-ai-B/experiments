class TextEmbeddingModel {
    constructor() {
        this.documents = [];
        this.vocabulary = new Set();
        this.termFrequency = {};
        this.documentFrequency = {};
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^a-z\s]/g, '')
            .split(/\s+/);
    }

    addDocument(text) {
        const tokens = this.tokenize(text);
        const tf = {};

        tokens.forEach(token => {
            if (!tf[token]) {
                tf[token] = 0;
            }
            tf[token]++;
            this.vocabulary.add(token);
        });

        this.documents.push(tokens);
        this.termFrequency[text] = tf;

        tokens.forEach(token => {
            if (!this.documentFrequency[token]) {
                this.documentFrequency[token] = 0;
            }
            this.documentFrequency[token]++;
        });
    }

    calculateTFIDF(text) {
        const tokens = this.tokenize(text);
        const tfidf = {};

        tokens.forEach(token => {
            const tf = this.termFrequency[text][token] / tokens.length;
            const idf = Math.log(this.documents.length / (this.documentFrequency[token] || 1));
            tfidf[token] = tf * idf;
        });

        return tfidf;
    }

    createVector(text) {
        const tfidf = this.calculateTFIDF(text);
        const vector = [];

        this.vocabulary.forEach(term => {
            vector.push(tfidf[term] || 0);
        });

        return vector;
    }
}

// Example usage:
const embeddingModel = new TextEmbeddingModel();

// Add documents to the embedding model
embeddingModel.addDocument('The quick brown fox jumps over the lazy dog');
embeddingModel.addDocument('A journey of a thousand miles begins with a single step');
embeddingModel.addDocument('To be or not to be, that is the question');
embeddingModel.addDocument('All that glitters is not gold');

// Create vectors for the documents
const vector1 = embeddingModel.createVector('The quick brown fox jumps over the lazy dog');
const vector2 = embeddingModel.createVector('A journey of a thousand miles begins with a single step');
const vector3 = embeddingModel.createVector('To be or not to be, that is the question');
const vector4 = embeddingModel.createVector('All that glitters is not gold');

console.log('Vector for document 1:', vector1);
console.log('Vector for document 2:', vector2);
console.log('Vector for document 3:', vector3);
console.log('Vector for document 4:', vector4);
