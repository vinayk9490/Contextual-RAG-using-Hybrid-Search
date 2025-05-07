Created a RAG application that uses Hybrid Techniques inorder to retrieve the query asked by the user.

Hybrid Search uses (semantic similarity(from FAISS)+(BM25 KeyWord Based Similarity))

Given a weightage of 0.7 to semantic similarity and 0.3 to Keyword based similarity

used FAISS (Facebook based Vector Database) to retrieve the related data from a user query.