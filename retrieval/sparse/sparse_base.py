from sklearn.feature_extraction.text import TfidfVectorizer
from ..base_retrieval import Retrieval

# 일단 모든 base는 다르게 tfidf로 대표해서 구현됨. 동일한거 나오면 다시 추상화할 것임. 
class TfidfRetrieval(Retrieval): # 이거는 tf-idf한정. 근데 bm25 나오면 Sparse base로 다시 추상화해야 함. 
    def __init__(self, args, tokenize_fn):
        super().__init__(args, tokenize_fn) 
        self.name = 'tfidf' # Please input lower case. (eg. tfidf/bm25/dpr)
        self.embedding_arc = TfidfVectorizer(tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000) # vectorizer in sparse retrieval / encoder in dense retrieval

    # override
    def exec_embedding(self): # tf-idf의 경우는 vecotrizer, bm25도 일단 tf-idf 쓰나? 
        p_embedding = self.embedding_arc.fit_transform(self.contexts)
        return p_embedding, self.embedding_arc
    
    # override
    def get_relevant_docc_bulk(self, queries, k=1):
        print('='*30)
        query_vec = self.embedding_arc.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])  # 스코어 리스트
            doc_indices.append(sorted_result.tolist()[:k])  # 예측 index 리스트

        return doc_scores, doc_indices