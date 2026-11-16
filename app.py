import altair as alt
import joblib
import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits

from diamonds import CATEGORIES, FEATURES, MODEL_PATH

st.set_page_config(page_title="Цены бриллиантов", layout="wide")
st.title("Оценка цены бриллианта")


@st.cache_resource
def load_bundle(path, modified_at):
    return joblib.load(path)


if not MODEL_PATH.is_file():
    st.error("Откройте Diamonds.ipynb и выполните ячейку сохранения модели.")
    st.stop()

bundle = load_bundle(MODEL_PATH, MODEL_PATH.stat().st_mtime_ns)
report = bundle["report"]
estimate, benchmark = st.tabs(["Оценка цены", "Эксперименты"])

with estimate:
    model_name = report["selected_model"]

    with st.form("diamond"):
        grade, measurements = st.columns(2)
        with grade:
            carat = st.number_input("Вес, карат", 0.20, 5.01, 0.70, 0.01)
            cut = st.selectbox("Огранка", CATEGORIES["cut"], index=4)
        with measurements:
            color = st.selectbox("Цвет", CATEGORIES["color"], index=3)
            clarity = st.selectbox("Чистота", CATEGORIES["clarity"], index=3)
        values = {"carat": carat, "cut": cut, "color": color, "clarity": clarity}
        st.subheader("Измерения")
        proportions, size = st.columns(2)
        with proportions:
            values["depth"] = st.number_input("Глубина, %", 43.0, 79.0, 62.0, 0.1)
            values["table"] = st.number_input("Площадка, %", 43.0, 95.0, 57.0, 0.1)
        with size:
            values["x"] = st.number_input("Длина, мм", min_value=0.01, value=5.72, step=0.01)
            values["y"] = st.number_input("Ширина, мм", min_value=0.01, value=5.75, step=0.01)
            values["z"] = st.number_input("Высота, мм", min_value=0.01, value=3.56, step=0.01)
        submitted = st.form_submit_button("Оценить цену", type="primary")

    if submitted:
        features = pd.DataFrame([values], columns=FEATURES)
        with threadpool_limits(limits=2):
            prediction = bundle["model"].predict(features)[0]
        st.metric("Оценка цены", f"${prediction:,.0f}")
        mae = report["models"][model_name]["mae_usd"]
        st.caption(f"Средняя абсолютная ошибка этой модели на тесте — ${mae:,.0f}.")
    st.caption("Оценка основана на исторических ценах круглых бриллиантов.")

with benchmark:
    st.subheader("Сравнение моделей")
    st.caption(
        f"Выбор по 5-фолдовой кросс-валидации — {report['selected_model']}. "
        f"Отложенная выборка — {report['test_rows']:,} записей."
    )
    comparison = pd.DataFrame.from_dict(report["models"], orient="index")
    comparison.index.name = "Модель"
    chart_data = comparison.reset_index()
    chart = alt.Chart(chart_data).mark_bar(color="#316c89").encode(
        x=alt.X("mae_usd:Q", title="Средняя абсолютная ошибка на тесте, USD"),
        y=alt.Y("Модель:N", sort="x", title=None),
        tooltip=[alt.Tooltip("Модель:N"), alt.Tooltip("mae_usd:Q", title="MAE, USD", format=",.2f")],
    )
    st.altair_chart(chart, width="stretch")
    comparison = comparison.rename(columns={
        "cv_mae_usd": "MAE на CV, $",
        "cv_mae_std_usd": "Разброс MAE на CV, $",
        "mae_usd": "MAE на тесте, $",
        "rmse_usd": "RMSE на тесте, $",
        "r2": "R² на тесте",
    })
    st.dataframe(comparison.style.format({
        column: "{:.3f}" if column == "R² на тесте" else "{:.2f}"
        for column in comparison.columns
    }), width="stretch")
    st.subheader("Ошибка по диапазонам фактических цен")
    bands = pd.DataFrame(report["price_slices"]).rename(columns={
        "actual_price_band": "Диапазон",
        "rows": "Записей",
        "mae_usd": "MAE, $",
        "bias_usd": "Среднее смещение, $",
    })
    st.dataframe(bands.style.format({"MAE, $": "{:.2f}", "Среднее смещение, $": "{:.2f}"}),
                 hide_index=True, width="stretch")
