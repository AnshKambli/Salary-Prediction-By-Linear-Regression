import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor | Linear Regression",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3147; }

    .metric-card {
        background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label { color: #8892b0; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { color: #64ffda; font-size: 26px; font-weight: 700; }
    .metric-sub   { color: #8892b0; font-size: 11px; margin-top: 4px; }

    .result-box {
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin-top: 12px;
    }
    .result-box-salary {
        background: linear-gradient(135deg, #0d2137 0%, #0a2744 100%);
        border: 2px solid #64ffda;
    }
    .result-box-exp {
        background: linear-gradient(135deg, #1a0d37 0%, #220a44 100%);
        border: 2px solid #a78bfa;
    }
    .result-label { color: #8892b0; font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
    .result-value-teal  { color: #64ffda; font-size: 46px; font-weight: 800; margin: 8px 0; }
    .result-value-purple{ color: #a78bfa; font-size: 46px; font-weight: 800; margin: 8px 0; }
    .result-range { color: #a8b2d8; font-size: 13px; margin-top: 4px; }

    .mode-active   { background: #64ffda !important; color: #0f1117 !important; font-weight: 700; border-radius: 8px; }
    .section-header { color: #ccd6f6; font-size: 17px; font-weight: 700; margin-bottom: 14px; padding-bottom: 7px; border-bottom: 2px solid #3a3f5c; }
    div[data-testid="stNumberInput"] input { background: #1a1d27; color: #ccd6f6; border: 1px solid #3a3f5c; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Dataset ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.DataFrame({
        "YearsExperience": [1.2,1.4,1.6,2.1,2.3,3.0,3.1,3.3,3.3,3.8,
                             4.0,4.1,4.1,4.2,4.6,5.0,5.2,5.4,6.0,6.1,
                             6.9,7.2,8.0,8.3,8.8,9.1,9.6,9.7,10.4,10.6],
        "Salary":          [39344,46206,37732,43526,39892,56643,60151,54446,64446,57190,
                             63219,55795,56958,57082,61112,67939,66030,83089,81364,93941,
                             91739,98274,101303,113813,109432,105583,116970,112636,122392,121873],
    })

@st.cache_resource
def train_model(df):
    X = df[["YearsExperience"]].values
    y = df["Salary"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, {
        "r2":        round(r2_score(y_test, y_pred), 4),
        "mae":       round(mean_absolute_error(y_test, y_pred), 2),
        "rmse":      round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "coef":      round(model.coef_[0], 2),
        "intercept": round(model.intercept_, 2),
        "y_test":    y_test,
        "y_pred":    y_pred,
    }

df      = load_data()
model, metrics = train_model(df)

# Inverse prediction helper: experience = (salary - intercept) / coef
def predict_experience(salary):
    return (salary - metrics["intercept"]) / metrics["coef"]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Salary Predictor")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("**Algorithm:** Linear Regression  \n**Feature:** Years of Experience  \n**Target:** Annual Salary (USD)  \n**Dataset:** 30 samples · 80/20 split")
    st.markdown("---")
    st.markdown("### 🔢 Model Equation")
    st.latex(r"\hat{y} = \beta_0 + \beta_1 \cdot X")
    st.markdown(f"- **β₀ (Intercept):** `${metrics['intercept']:,.0f}`\n- **β₁ (Slope):** `${metrics['coef']:,.0f} / year`")
    st.markdown("---")
    st.markdown("### 🔁 Inverse Formula")
    st.latex(r"\hat{X} = \frac{y - \beta_0}{\beta_1}")
    st.markdown("---")
    st.markdown("**Built by [Ansh Kambli](https://anshkambli.github.io/Portfolio)**  \nData Analyst | BFSI")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 💰 Salary Prediction App")
st.markdown("**Linear Regression** · Bidirectional — predict salary from experience, or experience from salary")
st.markdown("---")

# ── Metrics Row ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
for col, label, value, sub in [
    (c1, "R² Score",  f"{metrics['r2']}",           "Variance Explained"),
    (c2, "MAE",       f"${metrics['mae']:,.0f}",     "Mean Absolute Error"),
    (c3, "RMSE",      f"${metrics['rmse']:,.0f}",    "Root Mean Sq. Error"),
    (c4, "Slope",     f"${metrics['coef']:,.0f}",    "Per Year of Experience"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Mode Toggle ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Make a Prediction</div>', unsafe_allow_html=True)

mode = st.radio(
    "Prediction Mode",
    options=["📈 Experience → Salary", "💵 Salary → Experience"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

input_col, chart_col = st.columns([1, 2], gap="large")

# ── Mode A: Experience → Salary ────────────────────────────────────────────────
if mode == "📈 Experience → Salary":
    with input_col:
        st.markdown("#### Enter Years of Experience")
        years_exp = st.number_input(
            "Years of Experience",
            min_value=0.0, max_value=50.0,
            value=5.0, step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )
        st.slider("", min_value=0.0, max_value=20.0, value=years_exp,
                  step=0.1, key="exp_slider", disabled=True)

        predicted_salary = model.predict([[years_exp]])[0]
        low  = max(0, predicted_salary - metrics["mae"])
        high = predicted_salary + metrics["mae"]

        st.markdown(f"""<div class="result-box result-box-salary">
            <div class="result-label">Predicted Annual Salary</div>
            <div class="result-value-teal">${predicted_salary:,.0f}</div>
            <div class="result-range">± ${metrics['mae']:,.0f} confidence (MAE)</div>
            <div class="result-range" style="margin-top:6px;">Range: <b>${low:,.0f}</b> – <b>${high:,.0f}</b></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Formula:** `${metrics['intercept']:,.0f} + ${metrics['coef']:,.0f} × {years_exp:.1f}` = **${predicted_salary:,.0f}**")

    with chart_col:
        highlight_x, highlight_y = years_exp, predicted_salary
        point_color = "#64ffda"
        axis_label  = "Years of Experience"

# ── Mode B: Salary → Experience ───────────────────────────────────────────────
else:
    with input_col:
        st.markdown("#### Enter Target Salary (USD)")
        target_salary = st.number_input(
            "Target Salary",
            min_value=0, max_value=500_000,
            value=75_000, step=1_000,
            format="%d",
            label_visibility="collapsed",
        )

        predicted_exp = predict_experience(target_salary)
        low_exp  = max(0.0, predict_experience(target_salary - metrics["mae"]))
        high_exp = predict_experience(target_salary + metrics["mae"])

        if predicted_exp < 0:
            st.warning(f"⚠️ Salary ${target_salary:,} is below the model's base intercept (${metrics['intercept']:,.0f}). Try a higher value.")
            predicted_exp = 0.0

        st.markdown(f"""<div class="result-box result-box-exp">
            <div class="result-label">Estimated Years of Experience</div>
            <div class="result-value-purple">{predicted_exp:.1f} yrs</div>
            <div class="result-range">± {(high_exp - low_exp)/2:.1f} yr confidence (MAE)</div>
            <div class="result-range" style="margin-top:6px;">Range: <b>{low_exp:.1f}</b> – <b>{high_exp:.1f} years</b></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Formula:** `(${target_salary:,} − ${metrics['intercept']:,.0f}) ÷ ${metrics['coef']:,.0f}` = **{predicted_exp:.1f} years**")

    with chart_col:
        highlight_x, highlight_y = predicted_exp, target_salary
        point_color = "#a78bfa"
        axis_label  = "Years of Experience"

# ── Shared Chart ───────────────────────────────────────────────────────────────
with chart_col:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    x_line = np.linspace(0, max(15, highlight_x + 2), 300)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color="#64ffda", linewidth=2.5, label="Regression Line", zorder=3)
    ax.fill_between(x_line, y_line - metrics["mae"], y_line + metrics["mae"],
                    alpha=0.12, color="#64ffda", label=f"±MAE Band")

    ax.scatter(df["YearsExperience"], df["Salary"],
               color="#a78bfa", s=65, zorder=5, edgecolors="#ffffff22", linewidths=0.5, label="Training Data")

    ax.scatter([highlight_x], [highlight_y],
               color=point_color, s=200, zorder=7, marker="*",
               label=f"Prediction ({highlight_x:.1f} yrs → ${highlight_y:,.0f})")
    ax.axvline(x=highlight_x, color=point_color, linestyle="--", alpha=0.35, linewidth=1.2)
    ax.axhline(y=highlight_y, color=point_color, linestyle="--", alpha=0.35, linewidth=1.2)

    ax.set_xlabel("Years of Experience", color="#8892b0", fontsize=12)
    ax.set_ylabel("Salary (USD)",        color="#8892b0", fontsize=12)
    ax.set_title("Salary vs. Experience — Linear Regression", color="#ccd6f6", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8892b0")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.spines[:].set_color("#2d3147")
    ax.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8", fontsize=9)
    ax.grid(True, color="#2d3147", linestyle="--", alpha=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Diagnostics ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 Model Diagnostics</div>', unsafe_allow_html=True)

diag1, diag2 = st.columns(2)

with diag1:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#1a1d27")
    ax2.scatter(metrics["y_test"], metrics["y_pred"], color="#a78bfa", s=80, edgecolors="#ffffff22", zorder=3)
    mn = min(metrics["y_test"].min(), metrics["y_pred"].min()) - 3000
    mx = max(metrics["y_test"].max(), metrics["y_pred"].max()) + 3000
    ax2.plot([mn, mx], [mn, mx], color="#64ffda", linestyle="--", linewidth=1.5, label="Perfect Fit")
    ax2.set_xlabel("Actual Salary",    color="#8892b0", fontsize=11)
    ax2.set_ylabel("Predicted Salary", color="#8892b0", fontsize=11)
    ax2.set_title("Actual vs. Predicted", color="#ccd6f6", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="#8892b0")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax2.spines[:].set_color("#2d3147")
    ax2.legend(facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8")
    ax2.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with diag2:
    residuals = metrics["y_test"] - metrics["y_pred"]
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    fig3.patch.set_facecolor("#0f1117"); ax3.set_facecolor("#1a1d27")
    ax3.scatter(metrics["y_pred"], residuals, color="#ff9f43", s=80, edgecolors="#ffffff22", zorder=3)
    ax3.axhline(0, color="#64ffda", linewidth=1.5, linestyle="--")
    ax3.fill_between([metrics["y_pred"].min()-3000, metrics["y_pred"].max()+3000],
                     -metrics["rmse"], metrics["rmse"],
                     alpha=0.1, color="#64ffda", label=f"±RMSE")
    ax3.set_xlabel("Predicted Salary", color="#8892b0", fontsize=11)
    ax3.set_ylabel("Residuals",        color="#8892b0", fontsize=11)
    ax3.set_title("Residual Plot", color="#ccd6f6", fontsize=13, fontweight="bold")
    ax3.tick_params(colors="#8892b0")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax3.spines[:].set_color("#2d3147")
    ax3.legend(facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8")
    ax3.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


# ── Dataset ────────────────────────────────────────────────────────────────────
with st.expander("📋 View Full Dataset (30 samples)", expanded=False):
    disp = df.copy()
    disp["Predicted Salary ($)"] = model.predict(df[["YearsExperience"]].values).round(0).astype(int)
    disp["Error ($)"] = (disp["Salary"] - disp["Predicted Salary ($)"]).round(0).astype(int)
    disp["Salary"] = disp["Salary"].astype(int)
    disp.columns = ["Years of Experience", "Actual Salary ($)", "Predicted Salary ($)", "Error ($)"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8892b0;font-size:13px;'>"
    "Built by <a href='https://anshkambli.github.io/Portfolio' style='color:#64ffda;'>Ansh Kambli</a> · "
    "<a href='https://github.com/AnshKambli' style='color:#64ffda;'>GitHub</a> · "
    "Salary Prediction · Linear Regression"
    "</div>",
    unsafe_allow_html=True,
)
