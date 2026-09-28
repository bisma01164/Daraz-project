# Daraz Orders


# 1. load dataset
# 2. choose x and y
# 3. split into train test split
# 4. Train linear Regression
# 5. Evaluate it with MAE and R2
# 6. Predict orders for a new business scenerio
# 7. Build an interactive streamlit interface

# 1. Import libraries

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 2.  Page setting

st.set_page_config(
    page_title="Daraz Sales",
    page_icon="🛒",
    layout="wide"
)

# 3. load the dataset

df = pd.read_csv("daraz_daily_orders_synthetic.csv")

# 4. feature x and target y

feature_columns = [
    "Website_Visitors",
    "Ad_Spend_PKR",
    "Discount_Percent",
    "Weekend"
]

X = df[feature_columns]

y = df['Daily_Orders']


# 5. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=42,)

# 6. Train the linear regression model
model = LinearRegression()

model.fit(X_train,y_train)

# 7. Test the model
test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)

r2 = r2_score(y_test, test_predictions)


# 8. prepare tables

test_results = X_test.copy()
test_results["Actual_Orders"] = y_test
test_results["Predicted_Orders"] = test_predictions.round(1)

test_results["Residual"] = (
    test_results["Actual_Orders"] - test_results["Predicted_Orders"]
).round(1)

coefficient_table = pd.DataFrame(
    {
        "Feature" : feature_columns,
        "Coefficient" : model.coef_.round(4),
     }
)


# 9. streamlit interface

st.title(" 🛒 Daraz Sales")
st.write("Daraz 30-Day Sales Predictor: Forecast daily orders from business inputs.")
st.info("Model: Random Forest Regressor | Target: Daily_Orders")


# 10. Color Theme

st.markdown("""
<style>
/* Pura background light */
.stApp {
    background: linear-gradient(180deg, #FFF9F2 0%, #FFFFFF 100%);
}

/* Top 3 boxes */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #FFE0B2;
    border-left: 7px solid #FF6A00;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 6px 20px rgba(255, 106, 0, 0.08);
}
div[data-testid="stMetricLabel"] {
    color: #8D6E63 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
div[data-testid="stMetricValue"] {
    color: #1A1A1A !important;
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* Blue info bar */
div[data-testid="stAlert"] {
    background-color: #E3F2FD !important;
    border-radius: 12px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# 11. Quick project summary
metric_1 , metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Data Rows", len(df))
metric_2.metric("Training Rows", len(X_train))
metric_3.metric("Test MAE", f"{mae:.2f} orders")
metric_4.metric("Test R2", f"{r2:.3f}")

st.caption(f"Model Performance: MAE is {mae:.2f} orders and R2 score is {r2:.3f}. Lower MAE is better.")
# 12. create the main tabs

predict_tab, insights_tab, data_tab = st.tabs(
    [
        "Predict Orders",
        "Model Insights",
        "Explore Data"
    ]
)

# 13. Tab 1 "Predict Orders"

with predict_tab:
    st.subheader("Create a new Business Scenerio")
    st.write("Choose the values for a new day, then press the prediction button")

    with st.form("predictio_form_1"):
        left_col, right_col = st.columns(2)

        with left_col:
            website_visitors = st.slider(
                "Website Visitors",
                min_value= int(df["Website_Visitors"].min()),
                max_value= int(df["Website_Visitors"].max()),
                value=3500,
                step=50,
                help="How Many people visited the website today?",
            )

            ad_spend = st.slider(
                            "Ad Spend (PKR)",
                            min_value= int(df["Ad_Spend_PKR"].min()),
                            max_value= int(df["Ad_Spend_PKR"].max()),
                            value=15000,
                            step=500,
                            help="How much money was spent on advertising today?",
                        )
        with right_col:
            discount_percent = st.slider(
                "Discount (%)",
                min_value= int(df["Discount_Percent"].min()),
                max_value= int(df["Discount_Percent"].max()),
                    value=15,
                    step=1,
                help="What discount percentage is being offered?",
            )
            day_type = st.radio(
                "Day Type",
                options=["Weekday","Weekend"],
                horizontal=True
            )

        predict_button = st.form_submit_button(
            "Predict Daily Orders",
            type="primary",
            use_container_width=True
        )

    if predict_button:

        weekend = 1 if day_type == "Wweekend" else 0

        new_day = pd.DataFrame(
            {
                "Website_Visitors": [website_visitors],
                "Ad_Spend_PKR" : [ad_spend],
                "Discount_Percent" : [discount_percent],
                "Weekend": [weekend],
            }
        )
        # predict() return an array because a model can predict many rows.
        predicted_orders = model.predict(new_day)[0]

        st.success("Prediction Completed Successfully")

        st.metric(
            "Estimated Daily Orders",
            f"{round(predicted_orders):,} orders",
        )

        st.caption(
            f"Raw model estimate: {predicted_orders:.2f} orders"
        )

        with st.expander("see the exact input sent to the model"):
            st.dataframe(
                new_day,
                use_container_width=True,
                hide_index=True
            )

        st.warning(
            "This is a model estimate based on synthetic classroom data."
            "It is not a guaarenteed future sales number"
        )

# 14. tab 2 : model Insights
with insights_tab:
    st.subheader("How well did the model perform")

    score_1, score_2 = st.columns(2)

    score_1.metric("Mean Absolute Error", f"{mae:.2f} orders")
    score_2.metric(" R2 Score", f"{r2:.3f}")

    st.write(
        "MAE Prediction are about"
        f"{mae:.2f} orders away from the actual value on average."
    )

    st.write(
        "R2 : about"
        f"{r2 * 100:.1f}% of the variation in test orders is explained"
    )

    st.divider()

    # 15. Intercept and coefficients
    st.subheader("What did linear regression learn?")

    st.metric("Intercept", f"{model.intercept_:.2f}")

    st.dataframe(
        coefficient_table,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        coefficient_table.set_index("Feature")
    )

    st.caption(
        "A coefficient describes the fitted relationship whil the other model"
        "features are held constant. It does not automatically prove causation"
    )

    st.divider()

    st.subheader("Actual vs Predicted Orders")

    st.scatter_chart(
        test_results, 
        x ="Actual_Orders",
        y="Predicted_Orders",
    )

    with st.expander("Open the test-set prediction table"):
        st.dataframe(
            test_results,
            use_container_width=True
        )

# 16. Explore the dataset
with data_tab:
    st.subheader("Explore the Synthetic Dataset")

    st.write(
        "Each row represents one historical day. Daily_Orders is our target"
    )

    selected_feature = st.selectbox(
        "Choose a feature to compare with Daily Orders",
        options = feature_columns
    )

    st.scatter_chart(
        df,
        x= selected_feature,
        y="Daily_Orders"
    )

    show_full_data = st.checkbox("Show the complete dataset")

    if show_full_data:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View Summary Statistics"):
        st.dataframe(
            df.describe(),
            use_container_width=True
        )

    page_icon="🛒",
    layout="wide"

    # Intercept and coefficients
    st.subheader("What did linear regression learn?")

    st.metric("Intercept", f"{model.intercept_:.2f}")

    st.dataframe(
        coefficient_table,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        coefficient_table.set_index("Feature")
    )

    st.caption(
        "A coefficient describes the fitted relationship whil the other model"
        "features are held constant. It does not automatically prove causation"
    )

    st.divider()

    st.subheader("Actual vs Predicted Orders")

    st.scatter_chart(
        test_results, 
        x ="Actual_Orders",
        y="Predicted_Orders",
    )

    with st.expander("Open the test-set prediction table"):
        st.dataframe(
            test_results,
            use_container_width=True
        )

# Explore the dataset
with data_tab:
    st.subheader("Explore the Synthetic Dataset")

    st.write(
        "Each row represents one historical day. Daily_Orders is our target"
    )

    selected_feature = st.selectbox(
        "Choose a feature",
        feature_columns,
 key="unique_feature_selector")
    

    st.scatter_chart(
        df,
        x= selected_feature,
        y="Daily_Orders"
    )

    show_full_data = st.checkbox("Show the complete dataset",
    key="show_dataset_unique_1")

    if show_full_data:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View Summary Statistics"):
        st.dataframe(
            df.describe(),
            use_container_width=True
        )