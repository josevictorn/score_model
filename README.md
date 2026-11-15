# Modelo de Previsão de Pontuação — Plataforma de Educação Online

Modelo de **regressão linear** que prevê a **pontuação no teste final** de um aluno a partir da **quantidade de horas de estudo**. O objetivo é apoiar uma plataforma de educação online: estimar o desempenho esperado e dar ao aluno uma referência de quanto estudo é necessário para atingir uma meta de nota.

## Estrutura do projeto

```
.
├── datasets/
│   └── test_score.csv     # 101 registros: horas_estudo, pontuacao_teste
├── score_model.ipynb      # Notebook com todo o pipeline (EDA → treino → validação)
├── Pipfile / Pipfile.lock # Dependências (pipenv)
└── README.md
```

## Dados

| Coluna            | Tipo    | Descrição                         | Faixa         |
|-------------------|---------|-----------------------------------|---------------|
| `horas_estudo`    | float   | Horas de estudo do aluno          | 1,1 – 46,9    |
| `pontuacao_teste` | int     | Pontuação obtida no teste final   | 30 – 800      |

101 registros, sem valores nulos.

## Pipeline

| # | Etapa                              | Status       |
|---|------------------------------------|--------------|
| 1 | Carregar os dados                  | ✅ Concluído |
| 2 | Análise exploratória (EDA)         | ✅ Concluído |
| 3 | Preparar datasets de treino e teste| ✅ Concluído |
| 4 | Treinar o modelo                   | ✅ Concluído |
| 5 | Validar o modelo (métricas + resíduos) | ✅ Concluído — ver ressalvas abaixo |
| 6 | Salvar o modelo                    | ⏳ Pendente  |
| 7 | Entregar o modelo via API          | ⏳ Pendente  |

### 1. Carregar os dados
Leitura de `datasets/test_score.csv` com pandas.

### 2. EDA
- Estatísticas descritivas (`describe`).
- Dispersão `horas_estudo × pontuacao_teste`: relação claramente crescente.
- Boxplots das duas variáveis: sem outliers aparentes.
- Correlação de **Pearson ≈ 0,99** e de **Spearman = 1,00**.
- Histogramas das distribuições.

> Spearman = 1 indica que a relação é perfeitamente monotônica; Pearson um pouco abaixo de 1 sugere uma leve curvatura que a reta não captura totalmente.

### 3. Treino e teste
Divisão 70/30 com `train_test_split(test_size=0.30, random_state=50)` → 70 amostras de treino e 31 de teste.

### 4. Treinamento
`sklearn.linear_model.LinearRegression`. Equação obtida:

```
pontuacao_teste ≈ 8,48 + 15,64 × horas_estudo
```

Cada hora adicional de estudo está associada a cerca de **15,6 pontos** a mais no teste.

### 5. Validação

#### Métricas (conjunto de teste)

| Métrica | Valor   |
|---------|---------|
| R²      | 0,983   |
| MAE     | 22,96   |
| MSE     | 767,00  |
| RMSE    | 27,69   |

O modelo explica ~98% da variância da pontuação.

#### Análise de resíduos
- Resíduos padronizados (z-score) × valores preditos.
- QQ-Plot dos resíduos padronizados (pingouin, IC 95%).
- Testes de normalidade:

| Teste                         | Aplicado sobre             | Estatística | p-valor     | Conclusão (α = 0,05) |
|-------------------------------|----------------------------|-------------|-------------|----------------------|
| Shapiro-Wilk                  | resíduos brutos            | 0,950       | 0,157       | Não rejeita normalidade |
| Kolmogorov-Smirnov (notebook) | resíduos brutos vs N(0, 1) | 0,484       | 3,4 × 10⁻⁷  | Rejeita normalidade  |
| Kolmogorov-Smirnov (corrigido)| resíduos padronizados vs N(0, 1) | 0,127 | 0,657   | Não rejeita normalidade |

### 6. Salvar o modelo — pendente
Sugestão: serializar com `joblib.dump(reg_model, 'models/score_model.pkl')`.

### 7. Entregar via API — pendente
As dependências já estão no Pipfile (`fastapi`, `uvicorn`, `pydantic`). Sugestão de contrato:

```
POST /predict
{ "horas_estudo": 12.5 }
→ { "pontuacao_prevista": 204.0 }
```

Validar a entrada para aceitar apenas horas dentro da faixa observada no treino (≈ 1 a 47 h), já que fora dela o modelo extrapola.

## O modelo atende ao domínio?

As métricas são boas e, corrigido o teste KS, os resíduos são compatíveis com normalidade. A questão que fica é de negócio: **um erro médio de ~23 pontos é aceitável para a plataforma?**

Para contextualizar o MAE de 22,96:

| Referência                         | Erro relativo |
|------------------------------------|---------------|
| Nota máxima (800)                  | ≈ 2,9%        |
| Amplitude das notas (30–800)       | ≈ 3,0%        |
| Nota média (455)                   | ≈ 5,0%        |
| Maior erro absoluto no teste       | ≈ 58 pontos   |
| Previsões com erro ≤ 25 pontos     | 58% (18 de 31) |
| Previsões com erro ≤ 50 pontos     | 94% (29 de 31) |

Em termos de horas de estudo, 23 pontos equivalem a **~1,5 h** (23 ÷ 15,6).

## Como executar

Requer Python 3.15 e [pipenv](https://pipenv.pypa.io/).

```bash
pipenv install
pipenv shell
```

Depois, abra `score_model.ipynb` no VS Code e selecione o kernel do `.venv`. O Pipfile instala só o `ipykernel`; para usar o Jupyter fora do VS Code, rode `pipenv install notebook` antes.

## Tecnologias
pandas · scikit-learn · scipy · statsmodels · pingouin · seaborn · matplotlib · FastAPI · uvicorn · pydantic
