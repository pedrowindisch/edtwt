# entregas

## entrega 1
refere-se à tokenização, remoção de stopwords e stemming com as bibliotecas sugeridas pelo professor. ele cria o arquivo `entregas/p1/entrega_1.csv` a partir de `data/tweets.csv`, adicionando as colunas `tokenizacao_nltk`, `remocao_stopwords_spacy` e `stemming_nltk`.

para essa etapa, também precisamos remover dos textos caracteres decorativos, como "☆．。．:*･ﾟ, ｡･:*:･ﾟ'☆". além disso, os tokens puramente numéricos também foram retirados.

## entrega 2
refere-se à normalização textual com expressões regulares e à seleção de features com tf-idf. ele cria o arquivo `entregas/p2/entrega_2.csv` a partir de `entregas/p1/entrega_1.csv`, adicionando as colunas `normalizacao_re` e `features_tfidf_sklearn`.

## entrega 3
refere-se aos embeddings de palavras com word2vec (gensim). ele cria o arquivo `entregas/p3/entrega_3.csv` a partir de `entregas/p2/entrega_2.csv`, adicionando as colunas `tokens_word2vec` e `vetor_medio_word2vec`, além do arquivo `entrega_3_word2vec_features.json` com os hiperparâmetros do modelo.

## entrega 4
refere-se aos embeddings contextuais de sentenças com bertimbau (transformers). ele cria o arquivo `entregas/p4/entrega_4.csv` a partir de `entregas/p3/entrega_3.csv`, adicionando as colunas `emb_bertimbau_input` (texto enviado ao modelo, com normalização + hashtags) e `emb_bertimbau_mean` (vetor de 768 dimensões com mean pooling), além do arquivo `entrega_4_bert_features.json` com os metadados do modelo.

o notebook de análise (`entregas/p4/embeddings.ipynb`) realiza diagnóstico de similaridade, heatmap interativo, busca semântica, clustering com kmeans, visualização 2d/3d (pca e t-sne) e comparação com os embeddings word2vec do p3.