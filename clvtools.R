# BUSINESS SCIENCE LEARNING LABS ----
# LAB 58: CUSTOMER LIFETIME VALUE ----
# BONUS: CLVTOOLS FOR CLASSICAL CLV ----
# **** ----

# LIBRARIES ----

library(CLVTools)
library(plotly)
library(tidyquant)
library(tidyverse)
library(timetk)
library(lubridate)

# 1.0 DATA PREPARATION ----

# * Data Import ----
cdnow_raw_tbl <- vroom::vroom(
    file = "data/CDNOW_master.txt",
    delim = " ",
    col_names = FALSE
)

# * Data Cleanup -----
cdnow_tbl <- cdnow_raw_tbl %>%
    select(X2, X3, X5, X8) %>%
    set_names(
        c("customer_id", "date", "quantity", "price")
    ) %>%
    mutate(date = ymd(as.character(date))) %>%
    drop_na()


# 2.0 COHORT ANALYSIS ----
# - Only the customers that have joined at the specific business day

# * Get Range of Initial Purchases ----
cdnow_first_purchase_tbl <- cdnow_tbl %>%
    group_by(customer_id) %>%
    slice_min(date) %>%
    ungroup()

cdnow_first_purchase_tbl %>%
    pull(date) %>%
    range()

# "1997-01-01" "1998-06-26"

# * Set Cohort Span ----
#   - Set initial purchase: 1997-01-01 1997-03-31
ids_in_cohort <- cdnow_first_purchase_tbl %>%
    filter_by_time(
        .start_date = "1997-01",
        .end_date   = "1997-03"
    ) %>%
    distinct(customer_id) %>%
    pull(customer_id)

cdnow_cohort_tbl <- cdnow_tbl %>%
    filter(customer_id %in% ids_in_cohort)

# * Visualize: Total Cohort Purchases ----

cdnow_cohort_tbl %>%
    summarize_by_time(
        total_price = sum(price, na.rm = TRUE),
        .by   = "month"
    ) %>%
    plot_time_series(date, total_price, .y_intercept = 0)


# * Visualize: Individual Customer Purchases ----
n    <- 1:10
ids  <- unique(cdnow_cohort_tbl$customer_id)[n]

cdnow_cohort_tbl %>%
    filter(customer_id %in% ids) %>%
    group_by(customer_id) %>%
    plot_time_series(
        date, price,
        .y_intercept = 0,
        .smooth      = FALSE,
        .facet_ncol  = 2,
        .interactive = FALSE
    ) +
    geom_point(color = "#2c3e50")

# 3.0 CLVTOOLS MODELING ----

# * CLV Data ----
cdnow_cohort_clv <- clvdata(
    data.transactions = cdnow_cohort_tbl,
    date.format       = "ymd",
    time.unit         = "day",
    estimation.split  = "1998-01-01",
    name.id           = "customer_id",
    name.date         = "date",
    name.price        = "price"
)

cdnow_cohort_clv

summary(cdnow_cohort_clv)

# * PNBD METHOD ----

?CLVTools::pnbd

model_pnbd <- pnbd(cdnow_cohort_clv)

summary(model_pnbd)

coef(model_pnbd)


# 4.0 PREDICTION ----

predict(model_pnbd, cdnow_cohort_clv) %>%
    as_tibble()

g <- plot(model_pnbd)


gg <- g$data %>%
    ggplot(aes(period.until, value, color = variable)) +
    geom_line() +
    scale_color_tq() +
    theme_minimal()

ggplotly(gg)

# 5.0 PREDICT CUSTOMER SPENDING ----

model_gg <- gg(
    clv.data                 = cdnow_cohort_clv,
    remove.first.transaction = FALSE
)

model_gg

predict(model_gg)

plot(model_gg)


# REFERENCES ----

# R Examples:
#   https://bookdown.org/mike/marketing_research/customer-lifetime-value-clv.html
#   https://www.clvtools.com/index.html
#   https://github.com/bachmannpatrick/CLVTools

# Python Examples:
#    https://github.com/CamDavidsonPilon/lifetimes
#    https://lifetimes.readthedocs.io/en/latest/

# Case Studies:
#   https://hbr.org/2007/10/how-valuable-is-word-of-mouth


# Marketing Mix Modeling
#    Python Tutorial: https://towardsdatascience.com/marketing-channel-attribution-with-markov-chains-in-python-part-2-the-complete-walkthrough-733c65b23323
#    Basic Marketing Mix Model in Sklearn: https://practicaldatascience.co.uk/machine-learning/how-to-create-a-basic-marketing-mix-model-in-scikit-learn


