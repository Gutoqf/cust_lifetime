# CUSTOMER LIFETIME VALUE WITH MACHINE LEARNING ----
# **** ----

# LIBRARIES ----

# Machine Learning
library(tidymodels)
# Visualizar importância das variáveis em ML
library(vip)
# Manipulação, visualização e análise de dados
library(tidyverse)
# Séries Temporais
library(timetk)
# Manipular data/hora
library(lubridate)

# 1.0 Preparação dos dados

# Importação dos dados
cdnow_raw_tbl <- vroom::vroom(
    file = "data/CDNOW_master.txt", # Faz a leitura do arquivo txt
    delim = " ", # Define o delimitador (no nosso caso é o "espaço")
    col_names = FALSE # Não tem nomes de colunas, R vai criar automaticamente
)

# Limpeza dos Dados
cdnow_tbl <- cdnow_raw_tbl %>%
    # Seleciona as colunas
    select(X2, X3, X5, X8) %>%
    # Renomeia as colunas
    set_names(
        c("customer_id", "date", "quantity", "price")
    ) %>%
    # Altera para tipo "Data"
    mutate(date = ymd(as.character(date))) %>%
    # Remove linhas que tenham valores nulos
    drop_na()

# 2.0 Análise de Coorte
# Apenas os clientes que se inscreveram no dia útil específico

# Pega variedade de compras iniciais
cdnow_first_purchase_tbl <- cdnow_tbl %>%
    # Agrupa por clientes
    group_by(customer_id) %>%
    # Pega o mínimo (primeira data) do agrupamento por clientes
    slice_min(date) %>%
    ungroup()

# Nos ajuda a entender o intervalo de datas presente na nossa base
cdnow_first_purchase_tbl %>%
    pull(date) %>%
    range()

# Como a base é grande vamos definir um intervalo de Coorte
# o período que iremos trabalhar é entre 01-01-1997 e 31-03-1997
ids_in_cohort <- cdnow_first_purchase_tbl %>%
    # Filtra o período que escolhermos
    filter_by_time(
        .start_date = "1997-01",
        .end_date = "1997-03"
    ) %>%
    # Traz os clientes distintos que estão nesse período
    distinct(customer_id) %>%
    # Cria um vetor somente do número do cliente
    pull(customer_id)

# Filtra apenas os clientes que pertencem à coorte definida
cdnow_cohort_tbl <- cdnow_tbl %>%
    filter(customer_id %in% ids_in_cohort)

# * Visualização: Compras Totais da Coorte ----
cdnow_cohort_tbl %>%
    # Sumariza os dados no tempo + calcula agregado por mês
    summarize_by_time(
        total_price = sum(price, na.rm = TRUE),
        .by = "month"
    ) %>%
    # Plota nosso gráfico de série temporal
    plot_time_series(date, total_price, .y_intercept=0)
    #.value -> Valores reais
    #.value_smooth -> curva suavizada, destaca tendências gerais ao longo do tempo, eliminando variações curtas ou ruídos

# Compras individuais de clientes
n <- 1:10
ids <- unique(cdnow_cohort_tbl$customer_id)[n]


# * Visualize: Individual Customer Purchases ----
n    <- 1:10 # cria um vetor de 1 até 10
ids  <- unique(cdnow_cohort_tbl$customer_id)[n]  # Seleciona os 10 primeiros id's

cdnow_cohort_tbl %>%
    filter(customer_id %in% ids) %>%
    group_by(customer_id) %>%
    plot_time_series(
        date, price,
        .y_intercept = 0,
        .smooth      = FALSE,
        .facet_ncol  = 2,
        .interactive = FALSE,
        .title = "Customer Purchase Behavior"
    ) +
    geom_point(color = "#2c3e50")


# 3.0 MACHINE LEARNING ----
#
#  - Quais clientes gastaram mais nos últimos 90 dias? (Regressão)
#  - Qual a probabilidade de um cliente realizar uma compra nos próximos 90 dias? (Classificação)


# 3.1 Divisão (2-Fases) ----

# ** Fase 1: Divisão aleatória por ID do cliente ----
# Gera números aleatórios (Se tentarem rodar este código, são os mesmos números aleatórios gerados)
set.seed(123)

# Cria uma amostra de 80% dos clientes, representando o conjunto de treinamento de forma ordenada
ids_train <- cdnow_cohort_tbl %>%
    pull(customer_id) %>%
    unique() %>%
    sample(size = round(0.8*length(.))) %>%
    sort()

# Treino
split_1_train_tbl <- cdnow_cohort_tbl %>%
    filter(customer_id %in% ids_train)

# Teste
split_1_test_tbl  <- cdnow_cohort_tbl %>%
    filter(!customer_id %in% ids_train)

# ** Fase 2: Divisão de Tempo (Time Splitting)
splits_2_train <- time_series_split(
    split_1_train_tbl,
    assess = "90 days", # os últimos 90 dias nos dados de entrada serão separados como o conjunto de avaliação (teste)
    cumulative = TRUE # o conjunto de treinamento começa pequeno e cresce ao longo do tempo.
)

# Cross Validação
# -- Visualização: Facilita a compreensão de como os dados foram particionados em cada iteração
# -- Verificar a consistência: Nos ajuda a certificar que os períodos de treinamento e teste estão sendo corretamente separados
# -- Comunica a Metodologia: É útil para explicar aos stakeholders como a validação cruzada respeita o contexto temporal, garantindo uma avaliação mais realista

splits_2_train %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, price)

splits_2_test <- time_series_split(
    split_1_test_tbl,
    assess     = "90 days",
    cumulative = TRUE
)

splits_2_test %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, price)

# 3.2 FEATURE ENGINEERING (RFM) ----
# - Parte mais desafiadora
# - Processos de 2 estágios
# - Precisa enquadrar o problema
# - Pensar quais recursos (features) incluir

# ** Cria alvos dentro da amostra a partir dos dados de treinamento
targets_train_tbl <- testing(splits_2_train) %>% # Obtém o conjunto de dados de teste (avaliação) do objeto splits_2_train, gerado anteriormente com time_series_split
    group_by(customer_id) %>%
    summarise(
        spend_90_total = sum(price), # Soma o valor total das compras (price) realizadas pelo cliente no período de teste (90 dias)
        spend_90_flag = 1 # Atribui o valor 1 para cada cliente, indicando que esse cliente está incluído no conjunto de teste para o modelo
    )

# ** Cria alvos da amostragem externa a partir de testes (splits_2)

# Faz o mesmo que o código anterior só que com a base de teste
targets_test_tbl <- testing(splits_2_test) %>%
    group_by(customer_id) %>%
    summarise(
        spend_90_total = sum(price),
        spend_90_flag    = 1
    )

# ** Make Training Data ----
# ** Cria dados de treinamentos
# - Quais recursos (features) incluir?
# - RFV: Recência, Frequência e valor?

# Pega a última data contida na base
max_date_train <- training(splits_2_train) %>%
    pull(date) %>%
    max()

# Cria tabela de treino com comportamento do cliente (RFV)
train_tbl <- training(splits_2_train) %>%
    group_by(customer_id) %>%
    summarise(
        recency   = (max(date) - max_date_train) / ddays(1),
        frequency = n(),
        price_sum   = sum(price, na.rm = TRUE),
        price_mean  = mean(price, na.rm = TRUE)
    ) %>%
    # Junta com a tabela de alvos (targets_train_tbl)
    left_join(
        targets_train_tbl
    ) %>%
    # Trata valores ausentes (NA)
    replace_na(replace = list(
        spend_90_total = 0,
        spend_90_flag  = 0
        )
    ) %>%
    # Converte o campo spend_90_flag em um fator, necessário para usar a variável como variável categórica em modelos de machine learning
    mutate(spend_90_flag = as.factor(spend_90_flag))

# **  Cria tabela de teste com comportamento do cliente (RFV)
#    - Mesmo processo do código acima
#    - Precisa do histórico completo do cliente: divisões de treinamento (splits) 1 e 2
test_tbl <- training(splits_2_test) %>%
    group_by(customer_id) %>%
    summarise(
        recency     = (max(date) - max_date_train) / ddays(1),
        frequency   = n(),
        price_sum   = sum(price, na.rm = TRUE),
        price_mean  = mean(price, na.rm = TRUE)
    ) %>%
    left_join(
        targets_test_tbl
    ) %>%
    replace_na(replace = list(
        spend_90_total = 0,
        spend_90_flag  = 0
    )
    ) %>%
    mutate(spend_90_flag = as.factor(spend_90_flag))

# 3.3 Receitas (RECIPES)
# É uma forma de especificar como os dados devem ser preparados antes de serem utilizados em um modelo de machine learning.

# ** Modelo 1: Previsão de gastos de 90 dias
# spend_90_total -> variável resposta
# ~ -> relação entre a variável resposta e as variáveis preditoras (ou independentes)
# . -> todas as outras colunas da tabela train_tbl (exceto a variável resposta) serão usadas como preditoras
recipe_spend_total <- recipe(spend_90_total ~ ., data = train_tbl) %>%
    step_rm(spend_90_flag, customer_id) # Remove colunas desnecessárias

# ** Modelo 2: Probabilidade de gasto em 90 dias
recipe_spend_prob <- recipe(spend_90_flag ~ ., data = train_tbl) %>%
    step_rm(spend_90_total, customer_id)

# prep() -> gera um objeto com as transformações preparadas, mas ainda não aplica diretamente aos dados (não transforma o conjunto de dados em si, apenas prepara)
# juice() -> pega o objeto preparado (criado pelo prep()) e o aplica ao conjunto de dados
# glimpse() -> visão rápida da estrutura do dataframe resultante
recipe_spend_prob %>% prep() %>% juice() %>% glimpse()

#  Visualiza um resumão das transformações
summary(recipe_spend_prob)

# 3.4 MODELOS ----

# ** Modelo 1: Previsão de gastos para 90 dias

# XGBoost (uma árvore de decisão baseada em boosting)
# Objetivo dessa parte do código: cria um workflow, adiciona um modelo de regressão usando XGBoost, adiciona um recipe para transformar os dados de entrada e, finalmente, ajusta o modelo com os dados de treino.

# workflow() ->
# add_model() ->
# boost_tree(mode = "regression") ->  Define um modelo de árvore de decisão baseado em boosting para regressão (previsão de uma variável numérica)
# set_engine("xgboost")
wflw_spend_total_xgb <- workflow() %>% #cria um "esqueleto" para o seu pipeline de modelagem (Recipe em um único objeto)
    # adiciona o modelo ao workflow
    add_model(
    # Define um modelo de árvore de decisão baseado em boosting para regressão (previsão de uma variável numérica)
        boost_tree(
            mode = "regression"
        ) %>%
    # Especifica que a implementação do modelo de árvore de boosting será feita pelo pacote xgboost
            set_engine("xgboost")
    ) %>%
    # adiciona o recipe ao workflow (Etapas de remoção de variáveis, normalização e criação de variáveis dummy/interações)
    add_recipe(recipe_spend_total) %>%
    # treina o modelo com os dados de treino
    fit(train_tbl)

# ** Modelo 2: Probabilidade de gasto em 90 dias

wflw_spend_prob_xgb <- workflow() %>%
    add_model(
        boost_tree(
            mode = "classification" # Probabilidade de um evento ocorrer em uma das categorias (gastar ou não dentro de 90 dias)
        ) %>%
            set_engine("xgboost")
    ) %>%
    add_recipe(recipe_spend_prob) %>%
    fit(train_tbl)

# 3.5 AVALIAÇÃO DOS CONJUNTOS DOS TESTE (TEST SET EVALUATION)

# * Gerar as previsões de Teste dos dois modelos de XGBoost (um para regressão e outro para classificação) e combinar essas previsões com os dados do conjunto de teste (test_tbl)
predictions_test_tbl <-  bind_cols(
    # Previsões no conjunto de teste (test_tbl).
    predict(wflw_spend_total_xgb, test_tbl) %>%
        rename(.pred_total = .pred),
    # Prever a probabilidade do evento ocorrer (cliente gastar ou não gastar)
    predict(wflw_spend_prob_xgb, test_tbl, type = "prob") %>%
        select(.pred_1) %>%
        rename(.pred_prob = .pred_1)
) %>%
    # Combinação das Previsões com os Dados Originais
    bind_cols(test_tbl) %>%
    # Organiza o output da tabela
    select(starts_with(".pred"), starts_with("spend_"), everything())

# * Precisão do teste do modelo (Model Test Accuracy)

# MAE -> Erro Absoluto Médio (Quanto menor melhor é o modelo em termos de previsão do total de gastos)
# Em média, o modelo errou por aproximadamente 8.78 unidades em suas previsões
# Está ok, visto que nossos valores variam de R$0 e maior que R$ 400
predictions_test_tbl %>%
    yardstick::mae(spend_90_total, .pred_total)

# AUC -> Área sob a Curva (Métrica de desempenho utilizada para modelos de classificação binária)
# Mede a capacidade do modelo de distinguir entre as duas classes (gasto ou não gasto)
# Quanto mais próxima de 1, melhor o modelo em termos de discriminação entre as classes
# event_level = "second" -> especifica que o interesse está na probabilidade da classe "1" (clientes que gastaram, por exemplo)
predictions_test_tbl %>%
    yardstick::roc_auc(spend_90_flag, .pred_prob, event_level = "second")

# Taxa de verdadeiros positivos (TPR) contra a taxa de falsos positivos (FPR) para diferentes limiares de decisão
predictions_test_tbl %>%
    yardstick::roc_curve(spend_90_flag, .pred_prob, event_level = "second")%>%
    autoplot()

# 3.6 IMPORTÂNCIA DO RECURSO (FEATURE IMPORTANCE)

# * Modelo de Probabilidade
# Mostra a importância das variáveis em um modelo de aprendizado de máquina
# Cada barra no gráfico representa uma variável, e o comprimento da barra indica a importância dessa variável no modelo. Variáveis com barras mais longas são mais importantes para as previsões do modelo.
# Útil para entender como o modelo toma suas decisões e quais características dos dados estão sendo mais influentes nas previsões
vip(wflw_spend_prob_xgb$fit$fit)

# * Modelo de Gastos
vip(wflw_spend_total_xgb$fit$fit)

# 3.7 SALVA OS MODELOS CRIADOS ----

#fs::dir_create("artifacts")

#wflw_spend_prob_xgb %>% write_rds("artifacts/model_prob.rds")
#wflw_spend_total_xgb %>% write_rds("artifacts/model_spend.rds")

#vi_model(wflw_spend_prob_xgb$fit$fit) %>% write_rds("artifacts/vi_prob.rds")
#vi_model(wflw_spend_total_xgb$fit$fit) %>% write_rds("artifacts/vi_spend.rds")

# Centraliza os dados de treinamento e teste em um único dataframe
all_tbl <- bind_rows(train_tbl, test_tbl)

#  Dados originais + previsões dos modelos
predictions_all_tbl <- bind_cols(
    predict(wflw_spend_total_xgb, all_tbl) %>%
        rename(.pred_total = .pred),
    predict(wflw_spend_prob_xgb, all_tbl, type = "prob") %>%
        select(.pred_1) %>%
        rename(.pred_prob = .pred_1)
) %>%
    bind_cols(all_tbl) %>%
    select(starts_with(".pred"), starts_with("spend_"), everything())

# Salva Dataframe
#predictions_all_tbl %>% write_rds("artifacts/predictions_all_tbl.rds")

# 4.0 Como podemos utilizar essas informações?

# ** Quais clientes tem a maior probabilidade de compra nos próximos 90 dias?
# - Alvo para novos produtos semelhantes ao que eles compraram no passado
predictions_test_tbl %>%
    arrange(desc(.pred_prob))

# ** Quais clientes compraram recentemente, mas provavelmente não comprarão?
#    - Incentivar ações para aumentar a probabilidade
#    - Oferecer descontos, encorajar a indicação de um amigo, nutrir, deixando-os saber o que está por vir
predictions_test_tbl %>%
    filter(
        recency    > -90,
        .pred_prob < 0.2
    ) %>%
    arrange(.pred_prob)

# **
# ** Oportunidades perdidas: potenciais clientes (gastadores) que poderiam ser "desbloqueados"
# - Ofertas de pacotes incentivando compras em volumes
# - Foco em oportunidades perdidas
predictions_test_tbl %>%
    arrange(desc(.pred_total)) %>%
    filter(
        spend_90_total == 0
    )

# 5.0 PRÓXIMOS PASSOS
# - Evoluir Manipulação (Data Wrangling), modelagem e Visualização
# # - Melhoria do modelo:
# - AutoML (201)
# - Ajuste de hiperparâmetros (Hyper Parameter Tuning)
# - Conjunto (Ensembling)
# - Previsão: quando os clientes comprarão? (Forecasting)
# - Rotatividade: quais clientes provavelmente sairão? (Churn)
# - Shiny Web Applications & Production
