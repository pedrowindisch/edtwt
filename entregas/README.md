# entregas

## entrega 1
refere-se à tokenização, remoção de stopwords e stemming com as bibliotecas sugeridas pelo professor. ele cria o arquivo `entregas/p1/entrega_1.csv` a partir de `data/tweets.csv`, adicionando as colunas `tokenizacao_nltk`, `remocao_stopwords_spacy` e `stemming_nltk`.

para essa etapa, também precisamos remover dos textos caracteres decorativos, como "☆．。．:*･ﾟ, ｡･:*:･ﾟ'☆". além disso, os tokens puramente numéricos também foram retirados.

## entrega 2
refere-se à normalização textual com expressões regulares e à seleção de features com tf-idf. ele cria o arquivo `entregas/p2/entrega_2.csv` a partir de `entregas/p1/entrega_1.csv`, adicionando as colunas `normalizacao_re` e `features_tfidf_sklearn`.