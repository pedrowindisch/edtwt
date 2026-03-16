# edtwt

projeto de classificação de conteúdo (em três categorias) de posts provindos do edtwt, comunidade voltada para discussões relacionadas a transtornos alimentares 

## como rodar o projeto

será necessário ter, em sua máquina, python >= 3.13 e o poetry instalado (`pipx install poetry`, instale o pipx por [aqui](https://pipx.pypa.io/stable/installation/)).

crie um arquivo .env em seu diretório raiz e, nele, cole os conteúdos presentes no [.env.example](/.env.example), informando os valores reais.

rode o projeto com `poetry run python edtwt.py`

## extração

## anonimização

na tela inicial existe a opção `Exportar dataset anonimizado`. a exportação lê o banco SQLite e gera um CSV anonimizado no path definido na env `TWITTER_ANONYMIZED_OUTPUT_CSV`.

o CSV anonimizado remove ids, urls, usernames e nomes, reduz datas completas apenas para o dia, mascara menções/hashtags/links no texto e agrupa métricas (likes, seguidores, views) em faixas.
