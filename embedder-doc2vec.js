import fs from 'fs';
import natural from 'natural';
import path from 'path';

class Doc2VecSimple {
    constructor(vectorSize = 50) {
        this.vectorSize = vectorSize;
        this.vocabulary = {};
        this.wordVectors = {};
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^a-z\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 0);
    }

    buildVocabulary(texts) {
        let index = 0;
        texts.forEach(text => {
            const tokens = this.tokenize(text);
            tokens.forEach(token => {
                if (!this.vocabulary[token]) {
                    this.vocabulary[token] = index++;
                    this.wordVectors[token] = Array.from({ length: this.vectorSize }, () => Math.random() - 0.5);
                }
            });
        });
    }

    createWordVectors(texts) {
        this.buildVocabulary(texts);
    }

    createDocumentVector(text) {
        const tokens = this.tokenize(text);
        const vectors = tokens.map(token => this.wordVectors[token] || Array(this.vectorSize).fill(0));

        const docVector = vectors[0].map((_, colIndex) => {
            return vectors.map(row => row[colIndex]).reduce((a, b) => a + b, 0) / vectors.length;
        });

        return docVector;
    }

    fit(texts) {
        this.createWordVectors(texts);
    }

    inferVector(text) {
        return this.createDocumentVector(text);
    }
}

// Example usage:
const texts = [
    'The quick brown fox jumps over the lazy dog',
    'A journey of a thousand miles begins with a single step',
    'To be or not to be, that is the question',
    'All that glitters is not gold'
];

const doc2vec = new Doc2VecSimple();
doc2vec.fit(texts);

texts.forEach(text => {
    const vector = doc2vec.inferVector(text);
    console.log(`Vector for "${text}":`, vector);
});




class Doc2Vec {
    constructor(vectorSize = 100) {
        this.vectorSize = vectorSize;
        this.documents = [];
        this.docVectors = {};
        this.tokenizer = new natural.WordTokenizer();
    }

    tokenize(text) {
        return this.tokenizer.tokenize(text.toLowerCase().replace(/[^a-z\s]/g, ''));
    }

    addDocument(text, label) {
        const tokens = this.tokenize(text);
        const document = {
            label: label,
            tokens: tokens,
        };
        this.documents.push(document);
    }

    async train() {
        const TfIdf = natural.TfIdf;
        const tfidf = new TfIdf();

        this.documents.forEach((doc) => {
            tfidf.addDocument(doc.tokens.join(' '));
        });

        const vocab = tfidf.documents.reduce((acc, doc) => {
            doc.terms.forEach((term) => {
                if (!acc.includes(term.term)) {
                    acc.push(term.term);
                }
            });
            return acc;
        }, []);

        this.documents.forEach((doc, index) => {
            const vector = Array(vocab.length).fill(0);
            doc.tokens.forEach((token) => {
                const tokenIndex = vocab.indexOf(token);
                if (tokenIndex !== -1) {
                    vector[tokenIndex] = tfidf.tfidf(token, index);
                }
            });
            this.docVectors[doc.label] = vector;
        });
    }

    getVector(label) {
        return this.docVectors[label];
    }
}

// Example usage:
(async () => {
    const doc2Vec = new Doc2Vec();

    const texts = [
        {text: 'The quick brown fox jumps over the lazy dog', label: 'doc1'},
        {text: 'A journey of a thousand miles begins with a single step', label: 'doc2'},
        {text: 'To be or not to be, that is the question', label: 'doc3'},
        {text: 'All that glitters is not gold', label: 'doc4'}
    ];

    texts.forEach((item) => {
        doc2Vec.addDocument(item.text, item.label);
    });

    await doc2Vec.train();

    texts.forEach((item) => {
        const vector = doc2Vec.getVector(item.label);
        console.log(`Vector for "${item.label}":`, vector);
    });
})();
