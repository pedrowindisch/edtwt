# edtwt

projeto de classificação de conteúdo (em três categorias) de posts provindos do edtwt, comunidade voltada para discussões relacionadas a transtornos alimentares 

ver [entregas](/entregas/).

## como rodar o projeto

será necessário ter, em sua máquina, python >= 3.13 e < 3.15, com o poetry instalado (`pipx install poetry`, instale o pipx por [aqui](https://pipx.pypa.io/stable/installation/)).

crie um arquivo .env em seu diretório raiz e, nele, cole os conteúdos presentes no [.env.example](/.env.example), informando os valores reais.

rode o projeto com `poetry run python edtwt.py`

## extração

## anonimização

na tela inicial existe a opção `Exportar dataset anonimizado`. a exportação lê o banco SQLite e gera um CSV anonimizado no path definido na env `TWITTER_ANONYMIZED_OUTPUT_CSV`.

o CSV anonimizado remove ids, urls, usernames e nomes, reduz datas completas apenas para o dia, mascara menções/hashtags/links no texto e agrupa métricas (likes, seguidores, views) em faixas.

## entregas (partes)

na tela inicial também existem as opções `Entrega 1` e `Entrega 2`.

- `Entrega 1` se refere à tokenização, remoção de stopwords e stemming com as bibliotecas sugeridas pelo professor. ele cria o arquivo `entregas/p1/entrega_1.csv` a partir de `data/tweets.csv`, adicionando as colunas `tokenizacao_nltk`, `remocao_stopwords_spacy` e `stemming_nltk`.
- `Entrega 2`: cria `entregas/p1/entrega_2.csv`. ainda não está finalizado.
