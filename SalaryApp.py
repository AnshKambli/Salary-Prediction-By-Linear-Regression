import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Constants ──────────────────────────────────────────────────────────────────
USD_TO_INR = 83.5
LPA_DIVISOR = 100_000  # 1 LPA = ₹1,00,000

def usd_to_inr(usd): return usd * USD_TO_INR
def usd_to_lpa(usd): return round(usd_to_inr(usd) / LPA_DIVISOR, 2)
def lpa_to_usd(lpa): return (lpa * LPA_DIVISOR) / USD_TO_INR
def fmt_inr(amount_inr):
    """Format INR in Indian numbering: ₹X,XX,XX,XXX"""
    s = str(int(round(amount_inr)))
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    return "₹" + ",".join(reversed(parts)) + "," + last3

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
    .metric-value { color: #64ffda; font-size: 24px; font-weight: 700; }
    .metric-sub   { color: #8892b0; font-size: 11px; margin-top: 4px; }
    .result-box { border-radius: 16px; padding: 28px 32px; text-align: center; margin-top: 12px; }
    .result-box-salary { background: linear-gradient(135deg, #0d2137 0%, #0a2744 100%); border: 2px solid #64ffda; }
    .result-box-exp    { background: linear-gradient(135deg, #1a0d37 0%, #220a44 100%); border: 2px solid #a78bfa; }
    .result-label { color: #8892b0; font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
    .result-value-teal   { color: #64ffda; font-size: 42px; font-weight: 800; margin: 8px 0; }
    .result-value-purple { color: #a78bfa; font-size: 42px; font-weight: 800; margin: 8px 0; }
    .result-lpa  { color: #ffffff99; font-size: 17px; margin-top: 2px; }
    .result-range { color: #a8b2d8; font-size: 13px; margin-top: 6px; }
    .section-header { color: #ccd6f6; font-size: 17px; font-weight: 700; margin-bottom: 14px; padding-bottom: 7px; border-bottom: 2px solid #3a3f5c; }
</style>
""", unsafe_allow_html=True)


# ── Dataset (USD internally, convert for display) ──────────────────────────────
@st.cache_data
def load_data():
    return pd.DataFrame({
        "YearsExperience": [1.2,1.4,1.6,2.1,2.3,3.0,3.1,3.3,3.3,3.8,
                             4.0,4.1,4.1,4.2,4.6,5.0,5.2,5.4,6.0,6.1,
                             6.9,7.2,8.0,8.3,8.8,9.1,9.6,9.7,10.4,10.6],
        "Salary_USD":      [39344,46206,37732,43526,39892,56643,60151,54446,64446,57190,
                             63219,55795,56958,57082,61112,67939,66030,83089,81364,93941,
                             91739,98274,101303,113813,109432,105583,116970,112636,122392,121873],
    })

@st.cache_resource
def train_model(df):
    X = df[["YearsExperience"]].values
    y = df["Salary_USD"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, {
        "r2":        round(r2_score(y_test, y_pred), 4),
        "mae_usd":   mean_absolute_error(y_test, y_pred),
        "rmse_usd":  np.sqrt(mean_squared_error(y_test, y_pred)),
        "coef":      model.coef_[0],
        "intercept": model.intercept_,
        "y_test":    y_test,
        "y_pred":    y_pred,
    }

df = load_data()
model, m = train_model(df)

mae_inr  = usd_to_inr(m["mae_usd"])
rmse_inr = usd_to_inr(m["rmse_usd"])
mae_lpa  = usd_to_lpa(m["mae_usd"])

def predict_salary_inr(years):
    usd = model.predict([[years]])[0]
    return usd_to_inr(usd), usd_to_lpa(usd)

def predict_experience(lpa):
    usd = lpa_to_usd(lpa)
    exp = (usd - m["intercept"]) / m["coef"]
    return exp


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Salary Predictor")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"""
**Algorithm:** Linear Regression  
**Feature:** Years of Experience  
**Target:** Annual Salary (INR)  
**Dataset:** 30 samples · 80/20 split  
**Exchange Rate:** ₹{USD_TO_INR}/USD
    """)
    st.markdown("---")
    st.markdown("### 🔢 Model Equation")
    st.latex(r"\hat{y}_{INR} = (\beta_0 + \beta_1 \cdot X) \times 83.5")
    slope_inr = usd_to_lpa(m["coef"])
    intercept_lpa = usd_to_lpa(m["intercept"])
    st.markdown(f"- **Intercept:** `{intercept_lpa:.2f} LPA`\n- **Slope:** `{slope_inr:.2f} LPA / year`")
    st.markdown("---")
    st.markdown("### 🔁 Inverse Formula")
    st.latex(r"\hat{X} = \frac{(LPA \div 83.5 \times 10^5) - \beta_0}{\beta_1}")
    st.markdown("---")
    st.markdown("**Built by [Ansh Kambli](https://anshkambli.github.io/Portfolio)**  \nData Analyst | BFSI")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 💰 Salary Prediction App")
st.markdown("**Linear Regression** · Predict salary in **₹ INR / LPA** from experience, or vice versa")
st.markdown("---")

# ── Metrics Row ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

slope_lpa = usd_to_lpa(m["coef"])

for col, label, value, sub in [
    (c1, "R² Score",   f"{m['r2']}",                      "Variance Explained"),
    (c2, "MAE",        f"{fmt_inr(mae_inr)}",              f"≈ {mae_lpa:.2f} LPA"),
    (c3, "RMSE",       f"{fmt_inr(rmse_inr)}",             f"≈ {usd_to_lpa(m['rmse_usd']):.2f} LPA"),
    (c4, "Slope",      f"{slope_lpa:.2f} LPA/yr",          "Per Year of Experience"),
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
    options=["📈 Experience → Salary (₹)", "💵 Salary (LPA) → Experience"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)
input_col, chart_col = st.columns([1, 2], gap="large")

# ── Mode A: Experience → Salary ────────────────────────────────────────────────
if mode == "📈 Experience → Salary (₹)":
    with input_col:
        st.markdown("#### 📅 Enter Years of Experience")
        years_exp = st.number_input(
            "Years of Experience",
            min_value=0.0, max_value=50.0,
            value=3.0, step=0.1, format="%.1f",
            label_visibility="collapsed",
        )
        st.slider("Slide to adjust", min_value=0.0, max_value=20.0,
                  value=float(years_exp), step=0.1, disabled=True, label_visibility="collapsed")

        if years_exp == 0.0:
            st.markdown(f"""<div class="result-box" style="background:linear-gradient(135deg,#1a1a0d,#2a2a0a);border:2px solid #f0c040;border-radius:16px;padding:28px 32px;text-align:center;margin-top:12px;">
                <div class="result-label">⚠️ No Experience Entered</div>
                <div style="color:#f0c040;font-size:22px;font-weight:700;margin:12px 0;">Enter experience > 0</div>
                <div style="color:#a8b2d8;font-size:13px;">The model predicts salary based on<br>actual years of professional experience.</div>
                <div style="color:#8892b0;font-size:12px;margin-top:10px;">Dataset range: <b>1.2 – 10.6 years</b></div>
            </div>""", unsafe_allow_html=True)
            highlight_x, highlight_y_inr = 0.0, 0.0
            point_color = "#64ffda"
        else:
            pred_inr, pred_lpa = predict_salary_inr(years_exp)
            low_inr  = max(0, pred_inr - mae_inr)
            high_inr = pred_inr + mae_inr
            low_lpa  = max(0, pred_lpa - mae_lpa)
            high_lpa = pred_lpa + mae_lpa

            # Warn if outside training range
            if years_exp < 1.2 or years_exp > 10.6:
                st.warning(f"⚠️ {years_exp:.1f} yrs is outside the training range (1.2–10.6 yrs). Prediction is an extrapolation.")

            st.markdown(f"""<div class="result-box result-box-salary">
                <div class="result-label">Predicted Annual Salary</div>
                <div class="result-value-teal">{pred_lpa:.2f} LPA</div>
                <div class="result-lpa">{fmt_inr(pred_inr)} / year</div>
                <div class="result-range">± {mae_lpa:.2f} LPA confidence (MAE)</div>
                <div class="result-range">Range: <b>{low_lpa:.2f}</b> – <b>{high_lpa:.2f} LPA</b></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**{years_exp:.1f} yrs experience** → **{pred_lpa:.2f} LPA** ({fmt_inr(pred_inr)})")

        if years_exp > 0.0:
            highlight_x, highlight_y_inr = years_exp, pred_inr
        point_color = "#64ffda"

# ── Mode B: Salary → Experience ───────────────────────────────────────────────
else:
    with input_col:
        st.markdown("#### 💰 Enter Target Salary (LPA)")
        target_lpa = st.number_input(
            "Target Salary in LPA",
            min_value=0.0, max_value=500.0,
            value=10.0, step=0.5, format="%.1f",
            label_visibility="collapsed",
        )
        target_inr = target_lpa * LPA_DIVISOR

        pred_exp = predict_experience(target_lpa)
        low_exp  = max(0.0, predict_experience(max(0.1, target_lpa - mae_lpa)))
        high_exp = predict_experience(target_lpa + mae_lpa)

        if pred_exp < 0:
            intercept_lpa_val = usd_to_lpa(m["intercept"])
            st.warning(f"⚠️ {target_lpa} LPA is below the model's base salary ({intercept_lpa_val:.2f} LPA). Enter a higher value.")
            pred_exp = 0.0

        st.markdown(f"""<div class="result-box result-box-exp">
            <div class="result-label">Estimated Years of Experience</div>
            <div class="result-value-purple">{pred_exp:.1f} yrs</div>
            <div class="result-lpa">for a salary of {fmt_inr(target_inr)} / yr</div>
            <div class="result-range">± {abs(high_exp - low_exp)/2:.1f} yr confidence (MAE)</div>
            <div class="result-range">Range: <b>{low_exp:.1f}</b> – <b>{high_exp:.1f} years</b></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**{target_lpa} LPA** ({fmt_inr(target_inr)}) → **{pred_exp:.1f} years** of experience needed")

    highlight_x, highlight_y_inr = pred_exp, target_inr
    point_color = "#a78bfa"


# ── Shared Chart ───────────────────────────────────────────────────────────────
with chart_col:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    x_line = np.linspace(0, max(15, highlight_x + 2), 300)
    y_line_inr = usd_to_inr(model.predict(x_line.reshape(-1, 1)))

    ax.plot(x_line, y_line_inr, color="#64ffda", linewidth=2.5, label="Regression Line", zorder=3)
    ax.fill_between(x_line, y_line_inr - mae_inr, y_line_inr + mae_inr,
                    alpha=0.12, color="#64ffda", label="±MAE Band")

    salary_inr_all = usd_to_inr(df["Salary_USD"].values)
    ax.scatter(df["YearsExperience"], salary_inr_all,
               color="#a78bfa", s=65, zorder=5, edgecolors="#ffffff22", linewidths=0.5, label="Training Data")

    lpa_label = highlight_y_inr / LPA_DIVISOR
    ax.scatter([highlight_x], [highlight_y_inr],
               color=point_color, s=200, zorder=7, marker="*",
               label=f"Prediction ({highlight_x:.1f} yrs → {lpa_label:.1f} LPA)")
    ax.axvline(x=highlight_x,    color=point_color, linestyle="--", alpha=0.35, linewidth=1.2)
    ax.axhline(y=highlight_y_inr, color=point_color, linestyle="--", alpha=0.35, linewidth=1.2)

    ax.set_xlabel("Years of Experience", color="#8892b0", fontsize=12)
    ax.set_ylabel("Salary (INR)",        color="#8892b0", fontsize=12)
    ax.set_title("Salary (₹) vs. Experience — Linear Regression", color="#ccd6f6", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8892b0")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/LPA_DIVISOR:.1f}L"))
    ax.spines[:].set_color("#2d3147")
    ax.legend(loc="upper left", facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8", fontsize=9)
    ax.grid(True, color="#2d3147", linestyle="--", alpha=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Diagnostics ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 Model Diagnostics</div>', unsafe_allow_html=True)

diag1, diag2 = st.columns(2)

y_test_inr = usd_to_inr(m["y_test"])
y_pred_inr = usd_to_inr(m["y_pred"])

with diag1:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#1a1d27")
    ax2.scatter(y_test_inr, y_pred_inr, color="#a78bfa", s=80, edgecolors="#ffffff22", zorder=3)
    mn = min(y_test_inr.min(), y_pred_inr.min()) - 50000
    mx = max(y_test_inr.max(), y_pred_inr.max()) + 50000
    ax2.plot([mn, mx], [mn, mx], color="#64ffda", linestyle="--", linewidth=1.5, label="Perfect Fit")
    ax2.set_xlabel("Actual Salary (₹)",    color="#8892b0", fontsize=11)
    ax2.set_ylabel("Predicted Salary (₹)", color="#8892b0", fontsize=11)
    ax2.set_title("Actual vs. Predicted", color="#ccd6f6", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="#8892b0")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/LPA_DIVISOR:.1f}L"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/LPA_DIVISOR:.1f}L"))
    ax2.spines[:].set_color("#2d3147")
    ax2.legend(facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8")
    ax2.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with diag2:
    residuals_inr = y_test_inr - y_pred_inr
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    fig3.patch.set_facecolor("#0f1117"); ax3.set_facecolor("#1a1d27")
    ax3.scatter(y_pred_inr, residuals_inr, color="#ff9f43", s=80, edgecolors="#ffffff22", zorder=3)
    ax3.axhline(0, color="#64ffda", linewidth=1.5, linestyle="--")
    ax3.fill_between([y_pred_inr.min()-50000, y_pred_inr.max()+50000],
                     -rmse_inr, rmse_inr, alpha=0.1, color="#64ffda", label="±RMSE")
    ax3.set_xlabel("Predicted Salary (₹)", color="#8892b0", fontsize=11)
    ax3.set_ylabel("Residuals (₹)",        color="#8892b0", fontsize=11)
    ax3.set_title("Residual Plot", color="#ccd6f6", fontsize=13, fontweight="bold")
    ax3.tick_params(colors="#8892b0")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/LPA_DIVISOR:.1f}L"))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/LPA_DIVISOR:.0f}"))
    ax3.spines[:].set_color("#2d3147")
    ax3.legend(facecolor="#1a1d27", edgecolor="#3a3f5c", labelcolor="#a8b2d8")
    ax3.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


# ── Dataset Table ──────────────────────────────────────────────────────────────
with st.expander("📋 View Full Dataset (30 samples)", expanded=False):
    disp = df.copy()
    disp["Actual Salary (LPA)"]    = disp["Salary_USD"].apply(usd_to_lpa)
    disp["Predicted Salary (LPA)"] = [usd_to_lpa(v) for v in model.predict(df[["YearsExperience"]].values)]
    disp["Error (LPA)"]            = (disp["Actual Salary (LPA)"] - disp["Predicted Salary (LPA)"]).round(2)
    disp = disp[["YearsExperience", "Actual Salary (LPA)", "Predicted Salary (LPA)", "Error (LPA)"]]
    disp.columns = ["Years of Experience", "Actual Salary (LPA)", "Predicted Salary (LPA)", "Error (LPA)"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8892b0;font-size:13px;'>"
    "Built by <a href='https://anshkambli.github.io/Portfolio' style='color:#64ffda;'>Ansh Kambli</a> · "
    "<a href='https://github.com/AnshKambli' style='color:#64ffda;'>GitHub</a> · "
    f"Exchange Rate: ₹{USD_TO_INR}/USD"
    "</div>",
    unsafe_allow_html=True,
)
