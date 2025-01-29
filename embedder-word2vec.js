class embedder_word2vec {
    constructor() {
        this.vocabulary = {};
        this.vectorSize = 100; // Dimension of word vectors
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^a-z\s]/g, '')
            .split(/\s+/);
    }

    initializeVectors(tokens) {
        tokens.forEach(token => {
            if (!this.vocabulary[token]) {
                this.vocabulary[token] = Array.from({ length: this.vectorSize }, () => Math.random());
            }
        });
    }

    addText(text) {
        const tokens = this.tokenize(text);
        this.initializeVectors(tokens);
    }

    createVector(text) {

        const tokens = this.tokenize(text);
        const vectors = tokens.map(token => this.vocabulary[token] || Array(this.vectorSize).fill(0));

        const avgVector = vectors[0].map((_, colIndex) => {
            return vectors.map(row => row[colIndex]).reduce((a, b) => a + b, 0) / vectors.length;
        });

        return avgVector;
    }
}

function create_vectors(texts) {
    const embedder_word2vec = new embedder_word2vec();
    // Create vocabulary
    texts.forEach(function(text) {
        let vector = embedder_word2vec.addText(text)
    });
    // Create vectors 
    let vectors = []
    texts.forEach(function(text) {
        let vector = embedder_word2vec.createVector(text)
        console.log(vector)
        vectors.push(vector)
    });
    return vectors
}

export {
    create_vectors
}