from gensim.models import word2vec
from tools import make_train_vec, W2V, extract_words
import pickle


#word2vec 생성
source_sentences = word2vec.LineSentence('#원시언어 말뭉치')
pivot1_sentences = word2vec.LineSentence('#중간언어1 말뭉치')
pivot2_sentences = word2vec.LineSentence('#중간언어2 말뭉치')
target_sentences = word2vec.LineSentence('#대상언어 말뭉치')
source_model = word2vec.Word2Vec(source_sentences, size=200, sg=1, negative=5, workers=4, min_count=10)
pivot1_model = word2vec.Word2Vec(pivot1_sentences, size=200, sg=1, negative=5, workers=4, min_count=10)
pivot2_model = word2vec.Word2Vec(pivot2_sentences, size=200, sg=1, negative=5, workers=4, min_count=10)
target_model = word2vec.Word2Vec(target_sentences, size=200, sg=1, negative=5, workers=4, min_count=10)

#학습벡터 쌍 생성
with open('#원시언어-중간언어 초기사전', 'rb') as f:
    source_pivot_dict = pickle.load(f)
with open('#중간언어-대상언어 초기사전', 'rb') as f:
    pivot_target_dict = pickle.load(f)

train1 = make_train_vec(source_model, pivot1_model, source_pivot_dict)
train2 = make_train_vec(pivot2_model, target_model, pivot_target_dict)
with open('#원시언어-중간언어 학습벡터', 'wb') as f:
    pickle.dump(train1, f)
with open('#중간언어-대상언어 학습벡터', 'wb') as f:
    pickle.dump(train2, f)

#W2V 생성
source_index2vec, source_syn = extract_words(source_model, 'Noun')
target_index2vec, target_syn = extract_words(target_model, 'Noun')
source_W2V = W2V(source_index2vec, source_syn)
target_W2V = W2V(target_index2vec, target_syn)
pivot1_W2V = W2V(pivot1_model.index2word, pivot1_model.syn0)
pivot2_W2V = W2V(pivot2_model.index2word, pivot2_model.syn0)
source_W2V.save("#원시언어 W2V")
target_W2V.save("#대상언어 W2V")
pivot1_W2V.save("#중간언어1 W2V")
pivot2_W2V.save("#중간언어2 W2V")