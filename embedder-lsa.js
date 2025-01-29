import natural from 'natural';

class lsa_simple {
    constructor() {
        this.vocabulary = {};
        this.documents = [];
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
                }
            });
        });
    }

    buildTermDocumentMatrix(texts) {
        this.buildVocabulary(texts);
        const termDocumentMatrix = Array(Object.keys(this.vocabulary).length)
            .fill(null)
            .map(() => Array(texts.length).fill(0));

        texts.forEach((text, docIndex) => {
            const tokens = this.tokenize(text);
            tokens.forEach(token => {
                const termIndex = this.vocabulary[token];
                termDocumentMatrix[termIndex][docIndex]++;
            });
        });

        return termDocumentMatrix;
    }

    transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    multiplyMatrix(A, B) {
        const result = Array(A.length).fill(null).map(() => Array(B[0].length).fill(0));
        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < B[0].length; j++) {
                for (let k = 0; k < B.length; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    svd(matrix) {
        // Simplified SVD implementation
        // This is a placeholder, actual SVD implementation would be more complex
        const transposeMatrix = this.transpose(matrix);
        const matrixProduct = this.multiplyMatrix(matrix, transposeMatrix);
        const eigenvalues = Array(matrix.length).fill(0); // Placeholder for eigenvalues
        const U = matrix; // Placeholder for U matrix
        const S = eigenvalues.map(value => Math.sqrt(value)); // Placeholder for singular values
        const Vt = transposeMatrix; // Placeholder for V^T matrix

        return { U, S, Vt };
    }

    fit(texts) {
        const termDocumentMatrix = this.buildTermDocumentMatrix(texts);
        const { U } = this.svd(termDocumentMatrix);
        this.documents = U;
    }

    transform(text) {
        const tokens = this.tokenize(text);
        const termVector = Array(Object.keys(this.vocabulary).length).fill(0);
        tokens.forEach(token => {
            const termIndex = this.vocabulary[token];
            if (termIndex !== undefined) {
                termVector[termIndex]++;
            }
        });
        return this.multiplyMatrix([termVector], this.documents)[0];
    }
}

function lsa_simple_test() {

    // Example usage:
    const texts = [
        'The quick brown fox jumps over the lazy dog',
        'A journey of a thousand miles begins with a single step',
        'To be or not to be, that is the question',
        'All that glitters is not gold'
    ];

    const lsa = new lsa_simple();
    lsa.fit(texts);
    
    texts.forEach(text => {
        const vector = lsa.transform(text);
        console.log(`Vector for "${text}":`, vector);
    });
}


class LSA {
    constructor() {
        this.tfidf = new natural.TfIdf();
        this.documents = [];
    }

    tokenize(text) {
        return text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(word => word.length > 0);
    }

    addDocument(text) {
        this.documents.push(text);
        this.tfidf.addDocument(text);
    }

    fit() {
        this.termDocumentMatrix = [];
        this.tfidf.documents.forEach((doc, docIndex) => {
            const terms = {};
            this.tfidf.listTerms(docIndex).forEach(item => {
                terms[item.term] = item.tfidf;
            });
            this.termDocumentMatrix.push(terms);
        });

        const vocabulary = this.tfidf.terms().map(term => term.term);
        const matrix = this.termDocumentMatrix.map(terms => {
            return vocabulary.map(term => terms[term] || 0);
        });

        // Transpose the matrix for SVD
        const transpose = (matrix) => {
            return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
        };

        this.transposedMatrix = transpose(matrix);

        const svd = new natural.SVD();
        svd.decompose(this.transposedMatrix);

        this.U = svd.U;
        this.S = svd.S;
        this.V = svd.V;
    }

    transform(text) {
        const tokens = this.tokenize(text);
        const vocabulary = this.tfidf.terms().map(term => term.term);
        const termVector = vocabulary.map(term => tokens.includes(term) ? 1 : 0);

        const multiplyMatrixVector = (matrix, vector) => {
            return matrix.map(row => row.reduce((sum, value, index) => sum + value * vector[index], 0));
        };

        const documentVector = multiplyMatrixVector(this.U, termVector);
        return documentVector;
    }
}

function lsa_test() {
// Example usage:
const texts = [
    'The quick brown fox jumps over the lazy dog',
    'A journey of a thousand miles begins with a single step',
    'To be or not to be, that is the question',
    'All that glitters is not gold'
];

const lsa = new LSA();

texts.forEach(text => {
    lsa.addDocument(text);
});

lsa.fit();

texts.forEach(text => {
    const vector = lsa.transform(text);
    console.log(`Vector for "${text}":`, vector);
});
}

lsa_simple_test()
lsa_test()