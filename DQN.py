import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import deque
import random
import time

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8,activation="relu"))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def train_batch(self, states, actions, rewards, next_states, dones, batch_size=64, epochs=1):
        current_q = self.model.predict(states, verbose=0)#q value to match 

        target_q = np.copy(current_q)

        for i in range(len(states)):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                future_qs = self.model.predict(next_states[i:i+1], verbose=0)
                target_q[i, actions[i]] = rewards[i] + 0.95 * np.max(future_qs[0])

        self.model.fit(states, target_q, batch_size=batch_size, epochs=epochs, verbose=0)

    def act(self, state, epsilon=0):
        if np.random.random() < epsilon:#here model is selecting epsilion randomly 
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def save(self, name):
        self.model.save_weights(name)

def fast_train_dqn(data, feature_cols=None, target_col='status', episodes=10,
                   learning_rate=0.001, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9,
                   batch_size=256):



    X = data[feature_cols].fillna(0)
    y = data[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    agent = DQNAgent(X_train.shape[1], 2, learning_rate)#passing to NN to make a prediction and based on that will map it.

    train_accuracies = []
    test_accuracies = []

    memory_states = []
    memory_actions = []
    memory_rewards = []
    memory_next_states = []
    memory_dones = []

    for episode in range(episodes):
        start_episode = time.time()

        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))#epsilion is slowly changing and decaying after every epoch

        memory_states,memory_actions,memory_rewards,memory_next_states,memory_dones = []


        indices = np.random.permutation(len(X_train))

        correct_train = 0

        for i in indices:
            state = X_train[i].reshape(1, -1)
            action = agent.act(state, epsilon)
            reward = 1 if action == y_train[i] else -1#MAIN STEP ,POSITIVE REWARD FOR GOOD,NEGATIVE FOR BAD
            next_state = state 
            done = True

            memory_states.append(state[0])
            memory_actions.append(action)
            memory_rewards.append(reward)
            memory_next_states.append(next_state[0])
            memory_dones.append(done)

            if action == y_train[i]:
                correct_train += 1

            if len(memory_states) >= batch_size:
                states_batch = np.array(memory_states)
                next_states_batch = np.array(memory_next_states)

                agent.train_batch(
                    states_batch,
                    np.array(memory_actions),
                    np.array(memory_rewards),
                    next_states_batch,
                    np.array(memory_dones),
                    batch_size=batch_size
                )

                memory_states,memory_actions,memory_rewards,memory_next_states,memory_dones = []

        if memory_states:
            agent.train_batch(
                np.array(memory_states),
                np.array(memory_actions),
                np.array(memory_rewards),
                np.array(memory_next_states),
                np.array(memory_dones)
            )

        train_acc = correct_train / len(X_train)
        train_accuracies.append(train_acc)

        test_correct = 0
        for i in range(len(X_test)):
            state = X_test[i].reshape(1, -1)
            action = agent.act(state, epsilon=0)  
            if action == y_test[i]:
                test_correct += 1

        test_acc = test_correct / len(X_test)
        test_accuracies.append(test_acc)

        episode_time = time.time() - start_episode
        print(f"Epoch {episode+1}/{episodes} - Time: {episode_time:.2f}s - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f} - Epsilon: {epsilon:.4f}")#DISPLAY


    feature_importance = analyze_feature_importance(agent, feature_cols)


    return agent, scaler, feature_importance, train_accuracies, test_accuracies

def analyze_feature_importance(agent, feature_names):
    weights = agent.model.layers[0].get_weights()[0]
    importance = np.sum(np.abs(weights), axis=1)
    importance = importance / np.sum(importance)  

    importance_dict = {feature_names[i]: importance[i] for i in range(len(feature_names))}
    return importance_dict

def predict_phishing(data, agent, scaler, feature_cols=None, target_col='status'):

    X = data[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)

    predictions = []
    for i in range(len(X_scaled)):
        state = X_scaled[i].reshape(1, -1)
        action = agent.act(state, epsilon=0)  
        predictions.append(action)

    return np.array(predictions)

def main():

    feature_cols = [
        'google_index', 'page_rank', 'nb_www', 'ratio_digits_url',
        'domain_in_title', 'nb_hyperlinks', 'phish_hints', 'domain_age',
        'ip', 'nb_qm', 'length_url', 'ratio_intHyperlinks', 'nb_slash',
        'length_hostname', 'nb_eq', 'ratio_digits_host', 'shortest_word_host',
        'prefix_suffix', 'longest_word_path', 'tld_in_subdomain'
        agent, scaler, importance, train_acc, test_acc = fast_train_dqn(data, episodes=5)

    ]
