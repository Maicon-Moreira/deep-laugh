# Gerador de risadas

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Deep Laugh é um script em python que cria risos textuais de a partir de exemplos
fornecidos pelo usuário,utilizando um modelo de aprendizagem artificial
profunda implementado em Python com o framework Keras.

Alguns exemplos de risadas geradas:

  - kasdksddsagjd
  - kadfjsaskjassj
  - jjsfjsjjsdfasf
  - kasdgjdasgkjd
  - jssdgsjssgfj
  - kasdgkaasdkas
 
### Tecnologias

Deep Laugh usa as seguintes tecnologias:

* [Python] - Linguagem de programação 
* [Keras] - Framework de inteligência artificial profunda
* [Wandb] - Site para facilitar a visualização de modelos

### Instalação

Deep Laugh requer Python 3.

```sh
$ git clone https://github.com/Maicon-Moreira/deep-laugh
```

### Uso

Adicione suas risadas no arquivo sample_laughs.txt, uma risada por linha.

Para treinar o modelo de inteligência artificial execute:

```sh
$ python3 train_model.py
```

Agora um modelo com os padrões e informações de suas risadas estará salvo em model.h5.

Para criar as risadas execute:

```sh
$ python3 test_model.py
```

E pronto, risadas geradas automaticamente parecidas com as suas estão salvas em created_laughs.txt.

**MIT**
**Free Software, Hell Yeah!**


[Python]: <https://www.python.org/>
[Keras]: <https://keras.io/>
[Wandb]: <https://www.wandb.com/>
