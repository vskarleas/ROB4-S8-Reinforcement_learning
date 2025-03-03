# ROB4-S8: Reinforcement Learning

[English](#english-version) | [Français](#french-version)

![Reinforcement Learning](https://raw.githubusercontent.com/vskarleas/ROB4-S8-Reinforcement_learning/main/assets/rl_banner.png)

---

## English Version

### 🤖 Course Overview

This repository contains materials for the Reinforcement Learning course (ROB4-S8), focusing on the theoretical foundations and practical applications of reinforcement learning algorithms in robotics and autonomous systems.

Reinforcement Learning (RL) is a machine learning paradigm where agents learn to make decisions by taking actions in an environment to maximize cumulative rewards. Unlike supervised learning, RL doesn't require labeled examples but instead learns through trial and error, making it particularly suitable for robotics, game playing, and autonomous systems.

### 📚 Prerequisites

To make the most of this course, students should have:
- Solid understanding of Python programming
- Familiarity with basic machine learning concepts
- Knowledge of linear algebra and probability theory
- Experience with NumPy, Pandas, and Matplotlib libraries

### 🧩 Repository Structure

This repository contains Jupyter notebooks for three practical sessions (TPs) that progressively build understanding of reinforcement learning concepts:

```
.
├── TP1/
│   └── [Jupyter notebooks for first practical session]
├── TP2/
│   └── [Jupyter notebooks for second practical session]
├── TP3/
│   └── [Jupyter notebooks for third practical session]
└── README.md
```

### 🔍 Practical Sessions

#### TP1: Foundations of Reinforcement Learning

The first practical session introduces fundamental concepts of reinforcement learning:
- Markov Decision Processes (MDPs)
- Value functions (state-value function V(s) and action-value function Q(s,a))
- The Bellman equation
- Value iteration and policy iteration algorithms
- Implementation of a simple grid world environment

During this session, students learn to formalize problems as MDPs and solve them using dynamic programming methods.

#### TP2: Temporal Difference Learning

The second practical session explores model-free reinforcement learning methods:
- Monte Carlo methods
- Temporal Difference (TD) learning
- Q-learning
- SARSA (State-Action-Reward-State-Action) algorithm
- Exploration vs. exploitation dilemma
- ε-greedy policy

Students implement these algorithms in various environments and analyze their convergence properties and performance.

#### TP3: Advanced Reinforcement Learning Techniques

The third practical session covers more advanced topics:
- Function approximation with neural networks
- Deep Q-Networks (DQN)
- Policy gradient methods
- Actor-Critic algorithms
- Multi-agent reinforcement learning
- Applications in robotics simulations

This session bridges the gap between theoretical understanding and practical applications in complex environments.

### 🛠️ Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/vskarleas/ROB4-S8-Reinforcement_learning.git
cd ROB4-S8-Reinforcement_learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### 📖 Additional Resources

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) by Richard S. Sutton and Andrew G. Barto
- [Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course/unit0/introduction) by Hugging Face
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [David Silver's RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### License Information

**ROB4-S8-Reinforcement_learning** © 2025 by **Vasileios Filippos Skarleas** and **Rami Aridi** is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

This work also includes content that is not the property of **Vasileios Filippos Skarleas** and **Rami Aridi** and is subject to copyright and other licenses from their respective owners

---

## French Version

### 🤖 Aperçu du Cours

Ce dépôt contient les matériaux pour le cours d'Apprentissage par Renforcement (ROB4-S8), se concentrant sur les fondements théoriques et les applications pratiques des algorithmes d'apprentissage par renforcement en robotique et systèmes autonomes.

L'Apprentissage par Renforcement (RL) est un paradigme d'apprentissage automatique où les agents apprennent à prendre des décisions en effectuant des actions dans un environnement pour maximiser les récompenses cumulatives. Contrairement à l'apprentissage supervisé, le RL ne nécessite pas d'exemples étiquetés mais apprend par essais et erreurs, ce qui le rend particulièrement adapté à la robotique, aux jeux et aux systèmes autonomes.

### 📚 Prérequis

Pour tirer le meilleur parti de ce cours, les étudiants devraient avoir:
- Une bonne compréhension de la programmation Python
- Une familiarité avec les concepts de base de l'apprentissage automatique
- Des connaissances en algèbre linéaire et théorie des probabilités
- Une expérience avec les bibliothèques NumPy, Pandas et Matplotlib

### 🧩 Structure du Dépôt

Ce dépôt contient des notebooks Jupyter pour trois sessions pratiques (TPs) qui construisent progressivement la compréhension des concepts d'apprentissage par renforcement:

```
.
├── TP1/
│   └── [Notebooks Jupyter pour la première session pratique]
├── TP2/
│   └── [Notebooks Jupyter pour la deuxième session pratique]
├── TP3/
│   └── [Notebooks Jupyter pour la troisième session pratique]
└── README.md
```

### 🔍 Sessions Pratiques

#### TP1: Fondements de l'Apprentissage par Renforcement

La première session pratique introduit les concepts fondamentaux de l'apprentissage par renforcement:
- Processus de Décision Markoviens (MDPs)
- Fonctions de valeur (fonction de valeur d'état V(s) et fonction de valeur action-état Q(s,a))
- L'équation de Bellman
- Algorithmes d'itération de la valeur et d'itération de la politique
- Implémentation d'un environnement simple de monde en grille

Pendant cette session, les étudiants apprennent à formaliser les problèmes sous forme de MDPs et à les résoudre à l'aide de méthodes de programmation dynamique.

#### TP2: Apprentissage par Différence Temporelle

La deuxième session pratique explore les méthodes d'apprentissage par renforcement sans modèle:
- Méthodes de Monte Carlo
- Apprentissage par Différence Temporelle (TD)
- Q-learning
- Algorithme SARSA (État-Action-Récompense-État-Action)
- Dilemme exploration vs. exploitation
- Politique ε-greedy

Les étudiants implémentent ces algorithmes dans divers environnements et analysent leurs propriétés de convergence et leurs performances.

#### TP3: Techniques Avancées d'Apprentissage par Renforcement

La troisième session pratique couvre des sujets plus avancés:
- Approximation de fonction avec des réseaux de neurones
- Réseaux Q profonds (DQN)
- Méthodes de gradient de politique
- Algorithmes Acteur-Critique
- Apprentissage par renforcement multi-agents
- Applications dans des simulations robotiques

Cette session fait le pont entre la compréhension théorique et les applications pratiques dans des environnements complexes.

### 🛠️ Configuration et Installation

1. Cloner ce dépôt:
```bash
git clone https://github.com/vskarleas/ROB4-S8-Reinforcement_learning.git
cd ROB4-S8-Reinforcement_learning
```

2. Créer un environnement virtuel (recommandé):
```bash
python -m venv rl_env
source rl_env/bin/activate  # Sur Windows: rl_env\Scripts\activate
```

3. Installer les packages requis:
```bash
pip install -r requirements.txt
```

4. Lancer Jupyter Notebook:
```bash
jupyter notebook
```

### 📖 Ressources Supplémentaires

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) par Richard S. Sutton et Andrew G. Barto
- [Cours d'Apprentissage par Renforcement Profond](https://huggingface.co/deep-rl-course/unit0/introduction) par Hugging Face
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Cours de RL de David Silver](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### License Information

**ROB4-S8-Reinforcement_learning** © 2025 by **Vasileios Filippos Skarleas** and **Rami Aridi** is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

This work also includes content that is not the property of **Vasileios Filippos Skarleas** and **Rami Aridi** and is subject to copyright and other licenses from their respective owners
