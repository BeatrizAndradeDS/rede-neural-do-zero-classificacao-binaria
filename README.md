# 🧠 Rede Neural Artificial do Zero para Detecção de Fraude

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.1-orange.svg)](https://numpy.org/)
[![Neural Network](https://img.shields.io/badge/Neural%20Network-From%20Scratch-red.svg)](https://github.com/biasandrade/rede-neural-deteccao-fraude)

Implementei uma rede neural artificial **do zero** usando apenas NumPy para detectar fraudes em transações financeiras. O objetivo aqui foi entender **como uma rede neural funciona por dentro**, sem usar TensorFlow, Keras ou PyTorch.

---

🎯 Objetivo
Construir uma rede neural artificial sem usar bibliotecas prontas (TensorFlow, Keras, PyTorch) para:

•	Entender a matemática por trás das redes neurais
•	Implementar forward e backward propagation manualmente
•	Detectar padrões de fraude em transações financeiras

O problema escolhido foi detecção de fraude em transações financeiras - um caso clássico de classificação binária. Mas essa mesma arquitetura pode ser aplicada a diversos outros problemas, como:

- 📚 **Prever aprovação de alunos** baseado em horas de estudo
- 🛒 **Identificar propensão a compra** a partir do comportamento no site
- 🏥 **Apoiar decisões médicas** sobre necessidade de exames adicionais (ex: idade + IMC)
- 💳 **Aprovar ou negar crédito** com base em histórico financeiro
- 📧 **Classificar spam** em emails
- 🎯 **Prever churn** de clientes

Qualquer problema de classificação binária (sim/não, 0/1) pode usar essa estrutura.

---

## 🎬 Demonstração

![Sistema em Ação](demo.gif)

<details>
<summary>📊 <b>Ver resultados</b></summary>

### Dados de Teste

| Entrada (Transação) | Valor Real | Previsão | Resultado |
|---------------------|------------|----------|-----------|
| [1.5, 2] | Normal (0) | 0 | ✅ Acertou |
| [4, 5.5] | Fraude (1) | 1 | ✅ Acertou |

### Novos Dados (Deploy)

| Entrada | Previsão | Interpretação |
|---------|----------|---------------|
| [1, 2] | 0 | Transação Normal |
| [4, 5] | 1 | ⚠️ Transação Suspeita |

**Acurácia:** 2/2 corretos nos dados de teste (dataset pequeno, mas o foco aqui é didático).

</details>

---

<details>
<summary>🏗️ <b>Arquitetura da Rede</b></summary>

A rede é bem simples - um perceptron de camada única:

```
Entrada (2 features) 
    ↓
Soma Ponderada: Peso(W) · entradas(X) + viés/bias(b)
    ↓
Função Sigmoid
    ↓
Saída (probabilidade de 0 a 1)
    ↓
Classificação (0 ou 1)
```

### Processo de Treinamento

**Forward Pass:**
```python
previsão = sigmoid(W·X + b)
```

**Cálculo do Erro:**
```python
erro = previsão - y_real
```

**Backward Pass:**
```python
∂W = (1/n) · X^T · erro
∂b = (1/n) · Σ(erro)
```

**Atualização:**
```python
W = W - taxa_aprendizado · ∂W
b = b - taxa_aprendizado · ∂b
```

A cada iteração, os pesos e o bias são ajustados para minimizar o erro.

</details>

---

<details>
<summary>🔬 <b>Entendendo o processo de treinamento</b></summary>

### O que acontece a cada iteração?

1. **Forward Pass:** Modelo faz previsões
2. **Cálculo do Erro:** Compara com valores reais
3. **Backward Pass:** Calcula gradientes
4. **Atualização:** Ajusta pesos e bias

Após 1000 iterações, os pesos convergem de [0, 0] para [1.26, -0.47], 
minimizando o erro e aprendendo a separar as classes.

</details>

---

<details>
<summary>🔧 <b>Componentes Implementados</b></summary>

### 1. Função de Ativação Sigmoid
Converte qualquer valor em uma probabilidade entre 0 e 1:
```python
def func_activation_sigmoid(self, pred):
    return 1 / (1 + np.exp(-pred))
```

### 2. Forward Pass
Calcula a saída da rede dado um input:
```python
previsao = np.dot(X, self.pesos) + self.bias
previsao_final = self.func_activation_sigmoid(previsao)
```

### 3. Backward Pass
Ajusta os parâmetros usando gradiente descendente:
```python
dw = (1 / num_registros) * np.dot(X.T, erro)
db = (1 / num_registros) * np.sum(erro)

self.pesos -= self.taxa_aprendizado * dw
self.bias -= self.taxa_aprendizado * db
```

É aqui que a "mágica" acontece - o modelo aprende ajustando pesos e bias iterativamente.

</details>

---

## 📊 Dataset

Criei um dataset sintético simples para focar no algoritmo:

- **Features:** número de compras e valor médio
- **Classes:** 0 (transação normal) ou 1 (suspeita de fraude)
- **Divisão:** 6 para treino, 2 para teste, 2 para deploy

É um dataset pequeno de propósito - o objetivo é didático, não performance em produção.

---

## 🚀 Como Executar

<details>
<summary><b>📥 Instruções de instalação</b></summary>

### Pré-requisitos
- Python 3.8+
- NumPy
- Jupyter Notebook

### Passo a passo

```bash
# Clone o repositório
git clone https://github.com/biasandrade/rede-neural-do-zero-classificacao-binaria.git

# Entre na pasta
cd rede_neural_class_binaria_fraude

# Instale as dependências
pip install -r requirements.txt

# Abra o notebook
jupyter notebook rede_neural_class_binaria_fraude.ipynb
```

### Executando

O notebook está dividido em 5 partes:
1. Implementação do algoritmo (a classe da rede neural)
2. Preparação dos dados
3. Treinamento (1000 iterações)
4. Avaliação com dados de teste
5. Deploy com dados novos

Execute célula por célula pra acompanhar o processo de treinamento.

</details>

---

## 🧠 Conteúdo Utilizado

Conceitos aplicados:
- **Álgebra Linear:** multiplicação de matrizes, transposição, operações vetoriais
- **Cálculo Diferencial:** derivadas parciais, gradiente
- **Otimização:** gradiente descendente, taxa de aprendizado
- **Machine Learning:** forward propagation, backward propagation, função de ativação sigmoid

O mais interessante foi ver na prática como pequenos ajustes nos pesos, repetidos mil vezes, fazem o modelo convergir para a solução.

---

## 🔮 Próximos Passos

Ideias para evoluir o projeto futuramente:
- [ ] Adicionar camadas ocultas (transformar em deep neural network)
- [ ] Testar outras funções de ativação (ReLU, Tanh)
- [ ] Implementar regularização (L1/L2)
- [ ] Usar dataset real de fraudes (Kaggle)
- [ ] Adicionar métricas (precision, recall, F1)
- [ ] Criar visualização da convergência

---

## 👩‍💻 Autora

**Beatriz Andrade**  
18 anos trabalhando com dados e agora com Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Beatriz%20Andrade-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/andrade-beatriz/)
[![GitHub](https://img.shields.io/badge/GitHub-biasandrade-black?style=flat&logo=github)](https://github.com/biasandrade)
[![Email](https://img.shields.io/badge/Email-biasandrade%40gmail.com-red?style=flat&logo=gmail)](mailto:biasandrade@gmail.com)

---

Este projeto foi realizado por mim e faz parte do curso "Matemática e Estatística Aplicada Para Data Science, Machine Learning e IA" do curso de Pós-graduação em Data Science da Data Science Academy.

## 📄 Licença

MIT License.

---

⭐ Se esse projeto te ajudou a entender redes neurais, considera dar uma estrela!
