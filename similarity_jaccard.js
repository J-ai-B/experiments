function tokenize(text) {
    return new Set(text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/).filter(word => word.length > 0));
}

function jaccardSimilarity(text1, text2) {
    const set1 = tokenize(text1);
    const set2 = tokenize(text2);

    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    return intersection.size / union.size;
}

const text1 = "A single step towards a journey";
const texts = [
    'The quick brown fox jumps over the lazy dog',
    'A journey of a thousand miles begins with a single step',
    'To be or not to be, that is the question',
    'All that glitters is not gold'
]

for (let text of texts) {
    const similarityScore = jaccardSimilarity(text1, text);
    console.log(`Jaccard Similarity Score: ${similarityScore}`);
}


