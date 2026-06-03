import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── Inline CSS ─────────────────────────────────────────────────────────────────
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
    .metric-label { color: #8892b0; font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { color: #64ffda; font-size: 28px; font-weight: 700; }
    .metric-sub   { color: #8892b0; font-size: 12px; margin-top: 4px; }
    .prediction-box {
        background: linear-gradient(135deg, #0d2137 0%, #0a2744 100%);
        border: 2px solid #64ffda;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 16px;
    }
    .prediction-label { color: #8892b0; font-size: 15px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
    .prediction-value { color: #64ffda; font-size: 48px; font-weight: 800; margin: 8px 0; }
    .prediction-range { color: #a8b2d8; font-size: 14px; }
    .section-header { color: #ccd6f6; font-size: 18px; font-weight: 700; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #3a3f5c; }
    .stSlider [data-baseweb="slider"] div { background: #64ffda !important; }
</style>
""", unsafe_allow_html=True)


# ── Dataset (embedded — matches your notebook exactly) ─────────────────────────
@st.cache_data
def load_data():
    data = {
        "YearsExperience": [1.2,1.4,1.6,2.1,2.3,3.0,3.1,3.3,3.3,3.8,
                             4.0,4.1,4.1,4.2,4.6,5.0,5.2,5.4,6.0,6.1,
                             6.9,7.2,8.0,8.3,8.8,9.1,9.6,9.7,10.4,10.6],
        "Salary":          [39344,46206,37732,43526,39892,56643,60151,54446,64446,57190,
                             63219,55795,56958,57082,61112,67939,66030,83089,81364,93941,
                             91739,98274,101303,113813,109432,105583,116970,112636,122392,121873],
    }
    return pd.DataFrame(data)


@st.cache_resource
def train_model(df):
    X = df[["YearsExperience"]].values
    y = df["Salary"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "mae":  round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "coef": round(model.coef_[0], 2),
        "intercept": round(model.intercept_, 2),
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return model, metrics


# ── Load ───────────────────────────────────────────────────────────────────────
df = load_data()
model, metrics = train_model(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Salary Predictor")
    st.markdown("---")
    st.markdown("### 📊 About the Model")
    st.markdown("""
    **Algorithm:** Linear Regression  
    **Feature:** Years of Experience  
    **Target:** Annual Salary (USD)  
    **Dataset:** 30 samples  
    **Train/Test Split:** 80/20  
    """)
    st.markdown("---")
    st.markdown("### 🔢 Model Equation")
    st.latex(r"\hat{y} = \beta_0 + \beta_1 \cdot X")
    st.markdown(f"""
    - **β₀ (Intercept):** `${metrics['intercept']:,.0f}`  
    - **β₁ (Slope):** `${metrics['coef']:,.0f} / year`
    """)
    st.markdown("---")
    st.markdown("### 👤 Built by")
    st.markdown("[**Ansh Kambli**](https://anshkambli.github.io/Portfolio)  \nData Analyst | BFSI")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 💰 Salary Prediction App")
st.markdown("**Linear Regression** · Predicts annual salary based on years of professional experience")
st.markdown("---")

# ── Model Metrics Row ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{metrics['r2']}</div>
        <div class="metric-sub">Variance Explained</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">${metrics['mae']:,.0f}</div>
        <div class="metric-sub">Mean Absolute Error</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">${metrics['rmse']:,.0f}</div>
        <div class="metric-sub">Root Mean Sq. Error</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Slope</div>
        <div class="metric-value">${metrics['coef']:,.0f}</div>
        <div class="metric-sub">Per Year of Experience</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction Section ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Make a Prediction</div>', unsafe_allow_html=True)

pred_col, chart_col = st.columns([1, 2], gap="large")

with pred_col:
    years_exp = st.slider(
        "Years of Experience",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Slide to select years of professional experience",
    )

    predicted_salary = model.predict([[years_exp]])[0]
    margin = metrics['mae']  # use MAE as ±confidence range

    st.markdown(f"""<div class="prediction-box">
        <div class="prediction-label">Predicted Salary</div>
        <div class="prediction-value">${predicted_salary:,.0f}</div>
        <div class="prediction-range">± ${margin:,.0f} (MAE confidence range)</div>
        <div class="prediction-range" style="margin-top:8px;">
            Range: <b>${max(0, predicted_salary - margin):,.0f}</b> – <b>${predicted_salary + margin:,.0f}</b>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**{years_exp:.1f} years** of experience → **${predicted_salary:,.0f}/yr**\n\n"
            f"Formula: `{metrics['intercept']:,.0f} + {metrics['coef']:,.0f} × {years_exp:.1f}`")


with chart_col:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Regression line
    x_line = np.linspace(0, 12, 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color="#64ffda", linewidth=2.5, label="Regression Line", zorder=3)

    # Confidence band (±MAE)
    ax.fill_between(x_line, y_line - metrics['mae'], y_line + metrics['mae'],
                    alpha=0.15, color="#64ffda", label=f"±MAE Band (${metrics['mae']:,.0f})")

    # Scatter — training data
    ax.scatter(df["YearsExperience"], df["Salary"],
               color="#a78bfa", s=70, zorder=5, edgecolors="#ffffff22", linewidths=0.5,
               label="Training Data")

    # Highlight current prediction
    ax.scatter([years_exp], [predicted_salary],
               color="#ff6b6b", s=180, zorder=6, marker="*", label=f"Your Prediction (${predicted_salary:,.0f})")
    ax.axvline(x=years_exp, color="#ff6b6b", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=predicted_salary, color="#ff6b6b", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Years of Experience", color="#8892b0", fontsize=12)
    ax.set_ylabel("Salary (USD)", color="#8892b0", fontsize=12)
    ax.set_title("Salary vs. Experience — Linear Regression Fit", color="#ccd6f6", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8892b0")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.spines[:].set_color("#2d3147")
    ax.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8", fontsize=10)
    ax.grid(True, color="#2d3147", linestyle="--", alpha=0.5)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Charts Row ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 Model Diagnostics</div>', unsafe_allow_html=True)

diag1, diag2 = st.columns(2)

with diag1:
    # Actual vs Predicted
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#0f1117")
    ax2.set_facecolor("#1a1d27")

    ax2.scatter(metrics["y_test"], metrics["y_pred"], color="#a78bfa", s=80,
                edgecolors="#ffffff22", zorder=3)
    min_val = min(metrics["y_test"].min(), metrics["y_pred"].min()) - 2000
    max_val = max(metrics["y_test"].max(), metrics["y_pred"].max()) + 2000
    ax2.plot([min_val, max_val], [min_val, max_val], color="#64ffda", linestyle="--", linewidth=1.5, label="Perfect Fit")
    ax2.set_xlabel("Actual Salary", color="#8892b0", fontsize=11)
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
    # Residuals
    residuals = metrics["y_test"] - metrics["y_pred"]
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    fig3.patch.set_facecolor("#0f1117")
    ax3.set_facecolor("#1a1d27")

    ax3.scatter(metrics["y_pred"], residuals, color="#ff9f43", s=80,
                edgecolors="#ffffff22", zorder=3)
    ax3.axhline(0, color="#64ffda", linewidth=1.5, linestyle="--")
    ax3.fill_between([metrics["y_pred"].min()-2000, metrics["y_pred"].max()+2000],
                     -metrics['rmse'], metrics['rmse'],
                     alpha=0.1, color="#64ffda", label=f"±RMSE (${metrics['rmse']:,.0f})")
    ax3.set_xlabel("Predicted Salary", color="#8892b0", fontsize=11)
    ax3.set_ylabel("Residuals", color="#8892b0", fontsize=11)
    ax3.set_title("Residual Plot", color="#ccd6f6", fontsize=13, fontweight="bold")
    ax3.tick_params(colors="#8892b0")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax3.spines[:].set_color("#2d3147")
    ax3.legend(facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8")
    ax3.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


# ── Dataset Table ──────────────────────────────────────────────────────────────
with st.expander("📋 View Full Dataset (30 samples)", expanded=False):
    display_df = df.copy()
    display_df["Predicted Salary"] = model.predict(df[["YearsExperience"]].values).round(0).astype(int)
    display_df["Error"] = (display_df["Salary"] - display_df["Predicted Salary"]).round(0).astype(int)
    display_df["Salary"] = display_df["Salary"].astype(int)
    display_df.columns = ["Years of Experience", "Actual Salary ($)", "Predicted Salary ($)", "Error ($)"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8892b0; font-size:13px;'>"
    "Built by <a href='https://anshkambli.github.io/Portfolio' style='color:#64ffda;'>Ansh Kambli</a> · "
    "<a href='https://github.com/AnshKambli' style='color:#64ffda;'>GitHub</a> · "
    "Salary Prediction · Linear Regression Demo"
    "</div>",
    unsafe_allow_html=True,
)
