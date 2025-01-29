import { init_embedder, embed_text } from './embedder-glove.js';

class VectorStore {
    constructor(distanceFunction = 'cosine') {
        this.embedder = init_embedder()
        this.distanceFunction = distanceFunction;
        this.vectors = [];
        this.texts = [];
    }

    async addVectors(textArray) {
        for (let text of textArray) {
            const embedding = await embed_text(this.embedder, text);
            this.vectors.push(embedding);
            this.texts.push(text);
        }
    }

    async queryVector(query) {
        const embedding = await embed_text(this.embedder, query);
        return embedding
    }

    /*
    High or Low Score: Higher scores represent closer matches. 
    A cosine similarity of 1 means the vectors are identical in orientation, 
    while 0 means they are orthogonal (no similarity).
    */
    cosineSimilarityO(vec1, vec2) {
        const dotProduct = vec1.reduce((sum, value, index) => sum + value * vec2[index], 0);
        const magnitudeA = Math.sqrt(vec1.reduce((sum, value) => sum + value * value, 0));
        const magnitudeB = Math.sqrt(vec2.reduce((sum, value) => sum + value * value, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }

    cosineSimilarity(vec1, vec2) {
        if (vec1.length !== vec2.length) { 
            throw new Error('Vectors must be of the same length'); 
        } 
        let dotProduct = 0; 
        let magnitudeVec1 = 0; 
        let magnitudeVec2 = 0; 
        for (let i = 0; i < vec1.length; i++) { 
            dotProduct += vec1[i] * vec2[i]; 
            magnitudeVec1 += vec1[i] * vec1[i]; 
            magnitudeVec2 += vec2[i] * vec2[i]; } 
            if (magnitudeVec1 === 0 || magnitudeVec2 === 0) { 
                return 1; 
                // If either vector is zero, the cosine similarity is undefined, so we return maximum distance. 
            } 
            magnitudeVec1 = Math.sqrt(magnitudeVec1); 
            magnitudeVec2 = Math.sqrt(magnitudeVec2); 
            return dotProduct / (magnitudeVec1 * magnitudeVec2);
    }

    /*
    High or Low Score: Lower scores represent closer matches. 
    A Euclidean distance of 0 means the vectors are identical, 
    while larger values indicate greater dissimilarity.
    */
    euclideanDistance(vec1, vec2) {
        const sum = vec1.reduce((acc, value, index) => acc + Math.pow(value - vec2[index], 2), 0);
        return Math.sqrt(sum);
    }

    /*
    High or Low Score: Higher scores represent closer matches. 
    Larger dot products indicate vectors pointing in a similar direction with large magnitudes.
    */
    dotProduct(vec1, vec2) {
        return vec1.reduce((sum, value, index) => sum + value * vec2[index], 0);
    }

    /*
    High or Low Score: Lower scores represent closer matches. 
    A Manhattan distance of 0 means the vectors are identical, while larger values indicate greater dissimilarity.
    */
    manhattanDistance(vec1, vec2) {
        return vec1.reduce((acc, value, index) => acc + Math.abs(value - vec2[index]), 0);
    }

    calculateDistance(vec1, vec2) {
        
        switch (this.distanceFunction) {
            case 'cosine':
                return 1- this.cosineSimilarity(vec1, vec2);
            case 'euclidean':
                return this.euclideanDistance(vec1, vec2);
            case 'dot':
                return this.dotProduct(vec1, vec2);
            case 'manhattan':
                return this.manhattanDistance(vec1, vec2);
            default:
                throw new Error('Unknown distance function');
        }
    }

    async similaritySearch(query, k = 5) {
        const queryEmbedding = await this.queryVector(query);

        const distances = this.vectors.map((vector, index) => ({
            text: this.texts[index],
            score: this.calculateDistance(queryEmbedding, vector)
        }));

        distances.sort((a, b) => a.score - b.score);

        return distances.slice(0, k);
    }
}

// Example usage:
(async () => {
    // Initialize vector store with an example embedding model and cosine similarity
    const vectorStore = new VectorStore('cosine');

    // Add texts to the vector store
    await vectorStore.addVectors([
        'The quick brown fox jumps over the lazy dog',
        'A journey of a thousand miles begins with a single step',
        'To be or not to be, that is the question',
        'All that glitters is not gold'
    ]);

    // Perform a similarity search with a query
    const query = 'A single step towards a journey';
    const results = await vectorStore.similaritySearch(query, 3);

    console.log('Top k similar texts:', results);
})();
