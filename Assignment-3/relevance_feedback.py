#relevance_feedback.py
import numpy as np
from evaluation import read_gt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix

def relevance_feedback(vec_docs, vec_queries, sim, gt, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    y_true = read_gt(gt, sim.shape)
    for i in range(30):    
        n_rel = 0
        n_nonrel = 0
        rel = []
        non_rel = []
        gt_cur = y_true[:,i]

        topk = np.argsort(-sim[:, i])[:n]
        # print(topk)
        # for j in range(1033):
        #     if(gt_cur[j] == 1):
        #         print(j+1)
        for res in topk:
            if(gt_cur[res-1] == 1):
                n_rel += 1
                rel.append(res-1)
            else:
                n_nonrel += 1
                non_rel.append(res-1)
        # print(n_rel,n_nonrel)
        alpha = 0.7
        beta = 0.3
        if(len(rel) != 0):
            sum_rel = vec_docs[rel[0]]
            for doc in range(1,n_rel):
                sum_rel += vec_docs[rel[doc]]
        if(len(non_rel) != 0):
            sum_nonrel = vec_docs[non_rel[0]]
            for doc in range(1,n_nonrel):
               sum_nonrel += vec_docs[non_rel[doc]]
        vec_queries[i] += alpha*sum_rel - beta*sum_nonrel
        # print(alpha*sum_rel - beta*sum_nonrel)

    rf_sim = cosine_similarity(vec_docs,vec_queries) # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, gt, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    y_true = read_gt(gt, sim.shape)

    for i in range(30):    
        n_rel = 0
        n_nonrel = 0
        rel = []
        non_rel = []
        gt_cur = y_true[:,i] #relevant docs for ith query
        # print(gt_cur)
        topk = np.argsort(-sim[:, i])[:n]
        # print(topk)
        # for j in range(1033):
        #     if(gt_cur[j] == 1):
        #         print(j+1)
        for res in topk:
            if(gt_cur[res-1] == 1):
                n_rel += 1
                rel.append(res-1)
            else:
                n_nonrel += 1
                non_rel.append(res-1)
        # print(n_rel,n_nonrel)
        alpha = 0.7
        beta = 0.3
        if(len(rel) != 0):
           sum_rel = vec_docs[rel[0]]
           for doc in range(1,n_rel):
              sum_rel += vec_docs[rel[doc]]
        if(len(non_rel) != 0):
           sum_nonrel = vec_docs[non_rel[0]]
           for doc in range(1,n_nonrel):
               sum_nonrel += vec_docs[non_rel[doc]]
        vec_queries[i] += alpha*sum_rel - beta*sum_nonrel

        query_arr = vec_queries[i].toarray()    
        # docs_arr = vec_docs.toarray()
        rel_docs = []
        for j in gt:
            if(j[0] == i+1):
                rel_docs.append(j[1] - 1)
        top_overall = np.zeros((len(rel_docs)*10,2), dtype = int)
        ind = 0
        doc_array = vec_docs.toarray()
        for rel_doc in rel_docs:
            cur_doc = vec_docs[rel_doc].toarray()
            # print(cur_doc.shape)
            top_terms = np.argsort(-cur_doc[0])[:n]
            # print(top_terms)
            for k in range(10):
                top_overall[ind][0] = top_terms[k]
                top_overall[ind][1] = rel_doc
                ind += 1
            # for term in top_terms:
            #     query_arr[0][term] += cur_doc[0][term]
        top_n = np.argsort(-top_overall[:,0])[:n]

        for term in top_n:
            doc = top_overall[term][1]
            term_changed = top_overall[term][0]
            query_arr[0][term_changed] += doc_array[doc][term_changed]
        vec_queries[i] = csr_matrix(query_arr)



    rf_sim = cosine_similarity(vec_docs,vec_queries)  # change
    return rf_sim
