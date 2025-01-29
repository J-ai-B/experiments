import fs from 'fs';

class GloVeEmbedding {
    constructor(filePath, vectorSize = 50) {
        this.filePath = filePath;
        this.vectorSize = vectorSize;
        this.embeddings = {};
    }

    loadGloveVectors() {
        const lines = fs.readFileSync(this.filePath, 'utf-8').split('\n');
        lines.forEach(line => {
            const [word, ...vector] = line.split(' ');
            if (word) {
                this.embeddings[word] = vector.map(Number);
            }
        });
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^a-z\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 0);
    }

    createVector(text) {
        const tokens = this.tokenize(text);
        const vectors = tokens.map(token => this.embeddings[token] || Array(this.vectorSize).fill(0));

        const avgVector = vectors[0].map((_, colIndex) => {
            return vectors.map(row => row[colIndex]).reduce((a, b) => a + b, 0) / vectors.length;
        });

        return avgVector;
    }
}

function init_embedder() {
    const glove_path = './embedding_models/glove.6B.50d.txt';
    const glove = new GloVeEmbedding(glove_path);
    glove.loadGloveVectors();
    return glove
}

function embed_text(e, text) {
    const vector = e.createVector(text);
    console.log(`Vector for "${text}":`, vector);
    return vector
}

export {
    init_embedder,
    embed_text
}
