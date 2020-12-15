# Robot-Trader

Robo para daytrade no win da bovespa.

[Aprendizagem por reforço (RL) é uma estrutura geral onde os agentes aprendem a realizar ações em um ambiente de forma a maximizar uma recompensa. Os dois componentes principais são o ambiente, que representa o problema a ser resolvido, e o agente, que representa o algoritmo de aprendizagem.](https://www.tensorflow.org/agents/tutorials/0_intro_rl?hl=pt-br#copyright_2018_the_tf-agents_authors "tf-agents, retirado 15/12/2020")
[Aprendizagem por reforço (RL) é uma estrutura geral onde os agentes aprendem a realizar ações em um ambiente de forma a maximizar uma recompensa. Os dois componentes principais são o ambiente, que representa o problema a ser resolvido, e o agente, que representa o algoritmo de aprendizagem.] (https://www.tensorflow.org/agents/tutorials/0_intro_rl?hl=pt-br#copyright_2018_the_tf-agents_authors)

Neste caso, o ambiente é a bolsa de valores brasileira, os estados são os dados OHLC retirados do time-frame de 1 min dos ultimos 6 anos, as ações são operações de compra, venda e obsevação.O agente é uma rede neural que busca otimizar a politica. E por fim, as recompensas são os retornos das operações em pontos. Esse robô utiliza técnicas de aprendizado por reforço com tf-agents, após o treinamento é criado uma rede socket para receber e enviar dados com a plataforma Metatrader.
