# 🧠 Rede Neural do Zero para Detecção de Fraude

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.1-orange.svg)](https://numpy.org/)

Implementei uma rede neural artificial usando apenas NumPy para detectar fraudes em transações financeiras. O objetivo aqui foi entender **como uma rede neural funciona por dentro**, sem usar TensorFlow, Keras ou PyTorch.

---

## 🎯 Sobre o Projeto

Sempre quis entender de verdade o que acontece dentro de uma rede neural. Tive a oportunidade de aprender e construir do zero, implementando cada etapa manualmente: forward propagation, backward propagation, função de ativação, gradiente descendente... Tudo com operações de matriz usando NumPy.

O problema escolhido foi detecção de fraude em transações financeiras - um caso clássico de classificação binária.

---

## 🎬 Demonstração

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

**Acurácia:** 100% nos dados de teste (dataset pequeno, claro, mas o foco aqui é didático)

</details>

---

<details>
<summary>🏗️ <b>Arquitetura da Rede</b></summary>

A rede é bem simples - um perceptron de camada única:

```
Entrada (2 features) 
    ↓
Soma Ponderada (W·X + b)
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
git clone https://github.com/biasandrade/rede-neural-deteccao-fraude.git

# Entre na pasta
cd rede-neural-deteccao-fraude

# Instale as dependências
pip install -r requirements.txt

# Abra o notebook
jupyter notebook Projeto2-Final.ipynb
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

## 🧠 O Que Aprendi

Este projeto foi parte do curso de **Matemática e Estatística Aplicada para Data Science** da Data Science Academy. 

Principais conceitos aplicados:
- Álgebra linear (multiplicação de matrizes, transposição)
- Cálculo diferencial (derivadas parciais, gradiente)
- Otimização (gradiente descendente)
- Forward e backward propagation
- Função de ativação sigmoid

O mais legal foi ver na prática como pequenos ajustes nos pesos, repetidos mil vezes, fazem o modelo convergir para a solução.

---

## 🔮 Próximos Passos

Ideias para evoluir o projeto:
- [ ] Adicionar camadas ocultas (transformar em deep neural network)
- [ ] Testar outras funções de ativação (ReLU, Tanh)
- [ ] Implementar regularização (L1/L2)
- [ ] Usar dataset real de fraudes (Kaggle)
- [ ] Adicionar métricas (precision, recall, F1)
- [ ] Criar visualização da convergência

---

## 👩‍💻 Autora

**Beatriz Andrade**  
18 anos com dados | 2024: Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Beatriz%20Andrade-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/andrade-beatriz/)
[![GitHub](https://img.shields.io/badge/GitHub-biasandrade-black?style=flat&logo=github)](https://github.com/biasandrade)
[![Email](https://img.shields.io/badge/Email-biasandrade%40gmail.com-red?style=flat&logo=gmail)](mailto:biasandrade@gmail.com)

---

## 📄 Licença

MIT License - use à vontade, modifique, aprenda.

---

⭐ Se esse projeto te ajudou a entender redes neurais, considera dar uma estrela!
