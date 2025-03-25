# MoedaRara

### Setup

#### Criando ambiente virtual

```shell
python -m venv Moeda
cd Moeda
cd Scripts
activate
cd ..
```

A partir de agora, tudo será executado dentro do venv Moeda pelo terminal.

#### Instalando bibliotecas

```shell
pip install -r requirements.txt
```

### Rodando o back

```shell
cd backend

python app.py
```

ou

```shell
cd backend

python -m flask run
```

Deve mostrar as previsões do modelo

Feche a aba de imagem aberta e deixe o back rodando no terminal. Ao fazer isso deve aparecer:

```shell
Running on http://127.0.0.1:5000
```

### Rodando o front

Vá no arquivo index.html e clique no "Go Live" do VSCode.

Deve abrir nesse link: http://127.0.0.1:5500/frontend/templates/index.html

A cada envio de imagem que você fizer pela interface, no terminal deve aparecer:

```shell
127.0.0.1 - - [{data e hora}] "POST /classificar HTTP/1.1" 200 -
```

### Sources:

#### Fotos boas de moedas flor de estampa:

https://coinscatalog.net/brazil
https://coin-brothers.com/ (pesquisar por cruzeiro 1967)
https://en.numista.com/ (pesquisar por cruzeiro novo)

#### Qualidade variada:

https://pt.ucoin.net/table/?country=brazil&period=174
https://www.coinsbook.net/catalog/brazil-federative-republic-cruzeiro-novo-coins
https://mycoinsworld.com/BrazilR.html
https://www.numiscorner.com/ (tem mt moeda com erro)

#### Outros:

https://www.coinsbook.net/ (tem várias imagens que vem de outros sites, com link para eles)
