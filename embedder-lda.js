import natural from 'natural';

class lda_simple {
    constructor(numTopics, alpha = 0.1, beta = 0.01, iterations = 1000) {
        this.numTopics = numTopics;
        this.alpha = alpha;
        this.beta = beta;
        this.iterations = iterations;
        this.vocabulary = {};
        this.documents = [];
        this.wordTopicCounts = [];
        this.topicCounts = [];
        this.docTopicCounts = [];
        this.docLengths = [];
        this.topicAssignments = [];
    }

    tokenize(text) {
        return text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(word => word.length > 0);
    }

    buildVocabulary(texts) {
        let index = 0;
        texts.forEach(text => {
            const tokens = this.tokenize(text);
            tokens.forEach(token => {
                if (!this.vocabulary[token]) {
                    this.vocabulary[token] = index++;
                }
            });
        });
    }

    initialize(texts) {
        this.buildVocabulary(texts);
        const vocabSize = Object.keys(this.vocabulary).length;

        texts.forEach((text, docIndex) => {
            const tokens = this.tokenize(text);
            const docLength = tokens.length;
            this.docLengths[docIndex] = docLength;
            this.documents[docIndex] = tokens;

            this.docTopicCounts[docIndex] = Array(this.numTopics).fill(0);
            this.topicAssignments[docIndex] = [];

            tokens.forEach(token => {
                const topic = Math.floor(Math.random() * this.numTopics);
                this.topicAssignments[docIndex].push(topic);

                if (!this.wordTopicCounts[this.vocabulary[token]]) {
                    this.wordTopicCounts[this.vocabulary[token]] = Array(this.numTopics).fill(0);
                }
                this.wordTopicCounts[this.vocabulary[token]][topic]++;
                this.topicCounts[topic] = (this.topicCounts[topic] || 0) + 1;
                this.docTopicCounts[docIndex][topic]++;
            });
        });
    }

    sampleFullConditional(wordIndex, docIndex, topic) {
        const vocabSize = Object.keys(this.vocabulary).length;
        return (this.wordTopicCounts[wordIndex][topic] + this.beta) /
               (this.topicCounts[topic] + vocabSize * this.beta) *
               (this.docTopicCounts[docIndex][topic] + this.alpha) /
               (this.docLengths[docIndex] + this.numTopics * this.alpha);
    }

    gibbsSampling() {
        for (let iter = 0; iter < this.iterations; iter++) {
            this.documents.forEach((tokens, docIndex) => {
                tokens.forEach((token, wordIndex) => {
                    const currentTopic = this.topicAssignments[docIndex][wordIndex];
                    const vocabIndex = this.vocabulary[token];

                    // Decrement counts
                    this.wordTopicCounts[vocabIndex][currentTopic]--;
                    this.topicCounts[currentTopic]--;
                    this.docTopicCounts[docIndex][currentTopic]--;

                    // Sample new topic
                    const topicDistribution = Array(this.numTopics).fill(0).map((_, topic) => 
                        this.sampleFullConditional(vocabIndex, docIndex, topic)
                    );

                    const newTopic = this.sampleFromDistribution(topicDistribution);

                    // Increment counts
                    this.wordTopicCounts[vocabIndex][newTopic]++;
                    this.topicCounts[newTopic]++;
                    this.docTopicCounts[docIndex][newTopic]++;
                    this.topicAssignments[docIndex][wordIndex] = newTopic;
                });
            });
        }
    }

    sampleFromDistribution(probabilities) {
        const sum = probabilities.reduce((a, b) => a + b, 0);
        const threshold = Math.random() * sum;
        let cumulative = 0;
        for (let i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (cumulative > threshold) {
                return i;
            }
        }
        return probabilities.length - 1;
    }

    fit(texts) {
        this.initialize(texts);
        this.gibbsSampling();
    }

    transform(text) {
        const tokens = this.tokenize(text);
        const topicDistribution = Array(this.numTopics).fill(this.alpha);

        tokens.forEach(token => {
            const vocabIndex = this.vocabulary[token];
            if (vocabIndex !== undefined) {
                const wordTopicDistribution = this.wordTopicCounts[vocabIndex].map(
                    (count, topic) => (count + this.beta) / (this.topicCounts[topic] + Object.keys(this.vocabulary).length * this.beta)
                );
                wordTopicDistribution.forEach((prob, topic) => {
                    topicDistribution[topic] *= prob;
                });
            }
        });

        const sum = topicDistribution.reduce((a, b) => a + b, 0);
        return topicDistribution.map(prob => prob / sum);
    }
}

function lda_simple_test() {
    
// Example usage:
const texts = [
    'The quick brown fox jumps over the lazy dog',
    'A journey of a thousand miles begins with a single step',
    'To be or not to be, that is the question',
    'All that glitters is not gold'
];

const lda = new lda_simple(2); // For simplicity, assume 2 topics
lda.fit(texts);

texts.forEach(text => {
    const vector = lda.transform(text);
    console.log(`Vector for "${text}":`, vector);
});
}


class LDA {
    constructor(numTopics) {
        this.numTopics = numTopics;
        this.ldaModel = new natural.LDA(this.numTopics, 1000); // 1000 iterations
        this.documents = [];
    }

    tokenize(text) {
        return text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(word => word.length > 0);
    }

    addDocument(text) {
        const tokens = this.tokenize(text);
        this.documents.push(tokens);
    }

    fit() {
        this.ldaModel.initialize(this.documents);
        this.ldaModel.train();
    }

    getTopicDistribution(text) {
        const tokens = this.tokenize(text);
        const topicDistribution = this.ldaModel.getDocuments(tokens);
        return topicDistribution;
    }
}

function lda_test() {

// Example usage:
const texts = [
    'The quick brown fox jumps over the lazy dog',
    'A journey of a thousand miles begins with a single step',
    'To be or not to be, that is the question',
    'All that glitters is not gold'
];

const numTopics = 2;
const lda = new LDA(numTopics);

texts.forEach(text => {
    lda.addDocument(text);
});

lda.fit();

texts.forEach(text => {
    const vector = lda.getTopicDistribution(text);
    console.log(`Vector for "${text}":`, vector);
});

}

lda_simple_test()
lda_test()
