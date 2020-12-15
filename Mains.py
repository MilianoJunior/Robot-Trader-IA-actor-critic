# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:18:15 2020

@author: Administrador
1- primeiro passo: criar um ambiente conhecido minimo para verificar se a rede neural encontra a soluçao

"""
# import collections
import numpy as np
import tensorflow as tf
import tqdm
import chardet
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from typing import Any, List, Sequence, Tuple
import plotly.express as px
from Trade import Trade
import pandas as pd
import plotly.graph_objects as go

" 1 passo: importar os dados"
try:
    with open('M3.csv', 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
    base = pd.read_csv('M3.csv', encoding=result['encoding'])    
except:
    print('Erro, é preciso fazer o download dos dados OHLC em csv')

seed = 42
# env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()
    model = keras.Sequential([
        layers.Dense(num_hidden_units, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
    ])
    model1 = keras.Sequential([
        keras.layers.Dense(num_actions,activation="linear")
    ])
    model2 = keras.Sequential([
        keras.layers.Dense(1,activation="linear")
    ])
    self.common = model
    self.actor = model1
    self.critic = model2

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)


def Duracao(base):
    index = 0
    for i in base.values:
        base1 = i[0].split(':')
        base.at[index, 'Hora'] = float(base1[0])*100 + float(base1[1])
        index += 1
    return base
num_actions = 3
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)
trader = Trade()

colunas = ['Hora','dif', 'retacao +','retracao -', 'RSI', 'M22M44', 'M22M66', 'M66M44', 'ADX', 'ATR',
           'Momentum', 'Force','VOL', 'CCI', 'Bears', 'Bulls', 'Stock1',
           'Stock2', 'Wilians', 'Std', 'MFI', 'band1', 'band2','band3']

colunas1 = ['Hora', 'open', 'high', 'low', 'close'] 

dados3 = pd.DataFrame(data=base[len(base)-50000:len(base)-10].values,columns=base.columns)      
dados2 = pd.DataFrame(data=base[len(base)-50000:len(base)-10].values,columns=base.columns)
dados4 = pd.DataFrame(data=base[600000:600500].values,columns=base.columns)
dados5 = pd.DataFrame(data=base[600000:600500].values,columns=base.columns)
dados2 = dados2[colunas]
dados3 = dados3[colunas1]
dados4 = dados4[colunas]
dados5 = dados5[colunas1]
dados4 = Duracao(dados4)
dados2 = Duracao(dados2)
train_mean = dados2.mean(axis=0)
train_std = dados2.std(axis=0)
dados2 = (dados2 - train_mean) / train_std
dados4 = (dados4 - train_mean) / train_std

class ambiente():
    def __init__(self,dados2,dados3,dados4,dados5,trader,model,name="Trade"):
        self.name = name
        self.contador = 0
        self.cont = 0
        self.token1 = False
        self.token2 = False
        self.dados3 = dados3
        self.dados2 = dados2
        self.dados4 = dados4
        self.dados5 = dados5
        self.trader = trader
        self.model = model
        self.comprado = False
        self.valor = 0
        self.media = []
        self.metal = False
        self.actionA = 0
        self.A =[0,0]
    def teste(self):
        stop = -500
        gain = 500
        teste.reset()
        trader.reset()
        
        for i in range(0,500):
            initial_state = tf.constant([self.dados4.values[i]], dtype=tf.float32)
            # print('dados: ',dados3.values[i])
            action2 = model.predict(initial_state)
            action2 = np.argmax(action2[0])
            compra,venda,neg,ficha,comprado,vendido,posicao=trader.agente(self.dados5.values[i],self.actionA,stop,gain,1)
        print('              ')
        print('ganho atual: ',sum(neg.ganhofinal),'Numero de operacoes: ',len(neg.ganhofinal))
        self.print.append(sum(neg.ganhofinal))
        plt.plot(self.print)
        plt.show()
        
    def trade(self,action):
        p = 0
        done = False
        stop = -300
        gain = 300
        compra,venda,neg,ficha,comprado,vendido,recompensa = trader.agente(self.dados3.values[self.cont],self.A[self.contador],stop,gain,0)
        self.contador += 1
        self.cont += 1
        self.A.append(action)
        if comprado or vendido:
            self.metal = True
       
        if self.metal and (comprado == False and vendido == False):
            self.metal = False 
        if self.cont >= (len(dados3)-10):
            self.cont =0
        if self.dados3.values[self.cont][0] == '09:00':

            # self.cont = 0
            self.contador = 0
            self.A =[0,0]
            done = True
            self.media.append(sum(neg.ganhofinal))
            print('Final')
            print('ganho atual: ',sum(neg.ganhofinal),'N operacoes: ',len(neg.ganhofinal),' media: ',sum(self.media)/len(self.media))
            trader.reset()
        return self.dados2.values[self.cont],recompensa,done
    def reset(self):
        self.contador = 0
        


teste = ambiente(trader=trader,dados2=dados2,dados3=dados3,dados4=dados4,dados5=dados5,model=model)

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""  
  # print('passo 5: ',passo)
  state, reward, done =  teste.trade(action) # env.step(action)
  # print(state,reward,done,action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  # print('passo 4: ',passo)
  return tf.numpy_function(env_step, [action], 
                            [tf.float32, tf.int32, tf.int32])


def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> List[tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state
  
  for t in tf.range(max_steps):
    # print('passo 3: ',passo,' : max_steps: ',t)
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)
    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    # print('-----------------------')
    # print('state',state,reward.stack(),done)
    # print('-----------------------')
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()


  return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
                (tf.math.reduce_std(returns) + eps))

  return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.RMSprop(0.001)


@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    # print('passo 2',passo)
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode) 

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)
  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward


# self.dados2.values[self.cont]
rr = len(dados3)/5000
# print('rr: ',rr)
max_episodes = 10000
max_steps_per_episode = 550000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 10000000
running_reward = 120

# Discount factor for future rewards
gamma = 0.95


with tqdm.trange(max_episodes) as t:
    for i in range(max_episodes):
      # teste.reset()
      # trader.reset()
      initial_state = tf.constant(dados2.values[0], dtype=tf.float32)
      episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))
    
      running_reward = episode_reward*0.01 + running_reward*.99
    
      t.set_description(f'Episode {i}')
      t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
    
      # Show average episode reward every 10 episodes
      if i % 10 == 0:
        pass # print(f'Episode {i}: average reward: {avg_reward}')
    
      if running_reward > reward_threshold:  
          break
    
    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


" Realizando algumas previsoes"
comprado = False
stop = -300
gain = 300
trader = Trade()
trader.reset()
A =[0,0]
for i in range(len(dados2)-1):
    initial_state = tf.constant([dados2.values[i]], dtype=tf.float32)
    action2 = model.predict(initial_state)
    action2 = np.argmax(action2[0][0])
    compra,venda,neg,ficha,comprado,vendido,posicao=trader.agente(dados3.values[i],A[i],stop,gain,0)
    
    A.append(action2)
  

fig = go.Figure(data=[go.Candlestick(x=dados3.Hora[0:len(dados2)-1],
                open=dados3.open[0:len(dados2)-1], high=dados3.high[0:len(dados2)-1],
                low=dados3.low[0:len(dados2)-1], close=dados3.close[0:len(dados2)-1])
                      ])

op = []
for j in range(len(neg)-1):
    if neg.tipo.values[j] == 'compra':
        op.append(dict(x0=neg.inicio.values[j], x1=neg.fim.values[j], y0=0, y1=0.5, xref='x', yref='paper',line_width=2,name='compra'))
    if neg.tipo.values[j] == 'venda':
        op.append(dict(x0=neg.inicio.values[j], x1=neg.fim.values[j], y0=0, y1=1, xref='x', yref='paper',line_width=4,name='venda'))
    


fig.update_layout(
    title='Ambiente controlado',
    yaxis_title='WIN',
    shapes = op,
    annotations=[dict(
        x='2016-12-09', y=0.05, xref='x', yref='paper',
        showarrow=False, xanchor='left', text='Increase Period Begins')]
)


fig.write_html("teste.html")

model.save('M15/modelo_15')  
