import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
Arquivo = open('stopwords.txt')
stopwords_data = Arquivo.read()
stopwords_data = str(stopwords_data).split()
Arquivo.close()

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))

dictionary = p1.bag_of_words(train_texts,stopwords_data)
train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
T=25
L=0.01
thetas_pegasos = p1.pegasos(train_bow_features, train_labels, T, L,)
run=1
while(1):
    input_texts=input('Input your review: ')
    input_bow_features = p1.extract_bow_feature_vectors(['blah',input_texts],dictionary)
    output=p1.classify(input_bow_features,thetas_pegasos[0],thetas_pegasos[1])
    if (output[-1])==1:
        print('_______________________________________________________________')
        print('This is a possitive review!')
    else:
        print('_______________________________________________________________')
        print('This is a negative review')
