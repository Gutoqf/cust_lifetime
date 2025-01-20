# LL PRO BONUS: R SHINY APPLICATION ----
# BUSINESS SCIENCE LEARNING LABS ----
# LAB 58: CUSTOMER LIFETIME VALUE WITH MACHINE LEARNING ----
# **** ----

library(shiny)
library(bslib)
library(crosstalk)
library(plotly)
library(DT)
library(tidyverse)

color_primary <- "#78c2ad"
# color_primary <- "#2196f3"

ui <- shiny::navbarPage(
    footer = tagList(
        "Made by Business Science"
    ),

    title = 'Customer Lifetime Value',
    theme = bs_theme(version = 4, bootswatch = 'lux'),

    tabPanel(
        title = "Machine Learning",
        # SIDEBAR ----
        sidebarLayout(
            sidebarPanel(
                width = 3,
                h1("Scoring Your Customers Future Spending"),
                h3("by Business Science.") %>% tags$a(href = 'https://www.business-science.io/', target="_blank"),
                p("With Machine Learning, we can predict the probability and amount that customers will spend withing the next 90-days."),
                br(),
                hr(),
                h3("Visualization Controls"),
                sliderInput("sample_prop",
                            "Proportion of Data Shown:",
                            min = 0, max = 1,
                            value = 0.05, step = 0.05),
                hr(),

            ),
            mainPanel(
                width = 9,

                tabsetPanel(
                    tabPanel(
                        "Customer Exploration",
                        h1("Probability of Future Spend", tags$small(" by purchase frequency")),
                        plotlyOutput("probability_plot"),
                        h1("Key Features", tags$small(" for future spend")),
                        plotlyOutput("feature_plot")
                    ),

                    tabPanel(
                        "Customer Actions",
                        h1("Customer Actions"),
                        p("These Customers Should be Spending More."),
                        dataTableOutput("spend_more_table")
                    )

                )



            )
        )
    )
)

server <- function(input, output, session) {

    rv <- reactiveValues()

    observe({

        # IMPORT PREDICTIONS
        rv$prediction_tbl <- read_rds("artifacts/predictions_all_tbl.rds") %>%
            mutate(
                spend_actual_vs_pred = spend_90_total - (.pred_total + 0.0001)
            ) %>%
            mutate(text = str_glue("
                                   Customer ID: {customer_id}
                                   90-Day Spend Prob: {scales::percent(.pred_prob, accuracy = 0.1)}
                                   90-Day Spend Pred: {scales::dollar(.pred_total)}
                                   Actual 90-Day Spend: {scales::dollar(spend_90_total)}
                                   Actual vs Pred: {scales::dollar(spend_actual_vs_pred)}
                                   ---
                                   Recency: {recency}
                                   Spend History (Total): {scales::dollar(price_sum)}
                                   Avg Purchase: {scales::dollar(price_mean)}
                                   "))

        set.seed(123)
        rv$prediction_sample <- rv$prediction_tbl %>%
            sample_frac(size = input$sample_prop)

        rv$shared_predictions <- SharedData$new(rv$prediction_sample, key = ~customer_id, group = "customer_id")

        rv$features_tbl <- rv$prediction_sample %>%
            select(customer_id, .pred_prob, .pred_total, spend_90_total, spend_actual_vs_pred,
                   recency, frequency, price_sum) %>%
            pivot_longer(cols = -c(customer_id, .pred_prob, .pred_total, spend_90_total, spend_actual_vs_pred), names_to = "feature", values_to = "value") %>%
            group_by(feature) %>%
            mutate(value_scaled = scale(value) %>% as.numeric()) %>%
            ungroup() %>%
            mutate(text = str_glue("Customer ID: {customer_id}
                                    90-Day Spend Prob: {scales::percent(.pred_prob, accuracy = 0.1)}
                                    90-Day Spend Pred: {scales::dollar(.pred_total)}
                                    Actual 90-Day Spend: {scales::dollar(spend_90_total)}
                                    Actual vs Pred: {scales::dollar(spend_actual_vs_pred)}
                                    ---
                                    {feature}: {round(value,2)}"))

        rv$shared_features <- SharedData$new(rv$features_tbl, key = ~customer_id, group = "customer_id")

    })

    output$probability_plot <- renderPlotly({

        req(rv$prediction_sample)

        g <- rv$shared_predictions %>%

            ggplot(aes(frequency, .pred_prob, color = spend_actual_vs_pred)) +
            geom_point(aes(text = text), size = 4) +
            geom_smooth(se = F, color = 'black', method = "gam") +
            theme_minimal() +
            scale_color_gradient2(low = "red", mid = color_primary, high = 'black', midpoint = 0) +
            labs(x = "Purchase Frequency", y = "Probability of Future 90-Day Purchase")

        ggplotly(g, tooltip = "text") %>%
            highlight()
    })

    output$feature_plot <- renderPlotly({

        req(rv$prediction_sample)

        g <- rv$shared_features %>%

            ggplot(aes(feature, value_scaled)) +
            geom_violin() +
            geom_jitter(aes(text = text, color = spend_actual_vs_pred), size = 2, alpha = 0.75) +
            coord_flip() +
            theme_minimal() +
            scale_color_gradient2(low = "red", mid = color_primary, high = 'black', midpoint = 0) +
            labs(x = "", y = "Probability of Future 90-Day Purchase")

        ggplotly(g, tooltip = "text") %>%
            highlight()
    })

    output$spend_more_table <- renderDataTable({

        req(rv$prediction_tbl)

        rv$prediction_tbl %>%
            arrange(desc(.pred_total)) %>%
            filter(spend_90_total < 10) %>%
            filter(.pred_total > 10) %>%
            select(spend_actual_vs_pred, spend_90_total, .pred_total, customer_id, recency, frequency, price_sum, price_mean) %>%

            rename(
                "Actual vs Prediction"   = spend_actual_vs_pred,
                "90-Day Spend (Predict)" = .pred_total,
                "90-Day Spend (Actual)"  = spend_90_total,
                "Cust ID"                = customer_id,
                "Monetary (Total)"       = price_sum,
                "Monetary (Avg)"         = price_mean
            ) %>%
            rename_all(
                .funs = ~ str_replace_all(., "_", " ") %>%
                    str_to_title()

            ) %>%

            DT::datatable(
                extensions = 'Buttons',
                options   = list(
                    scrollX = TRUE,
                    dom     = "tBp",
                    buttons = c('copy', 'csv', 'excel')
                )
            ) %>%

            DT::formatCurrency(
              columns = c("Actual Vs Prediction",
                          "90-Day Spend (Predict)", "90-Day Spend (Actual)",
                          "Monetary (Total)", "Monetary (Avg)"),
              digits  = 0
            )

    })


}


shinyApp(ui, server)
