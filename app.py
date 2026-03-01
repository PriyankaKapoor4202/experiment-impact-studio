import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Experiment Impact Studio", page_icon="📊", layout="wide")

# ---------- Helpers ----------
def make_sample_ab(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n = 20000
    groups = rng.choice(["control", "variant"], size=n, p=[0.5, 0.5])

    # baseline conversion + uplift
    base = 0.085
    uplift = 0.012  # absolute uplift
    p = np.where(groups == "variant", base + uplift, base)

    converted = rng.binomial(1, p, size=n)

    # guardrail metric: average revenue per user (ARPU-ish)
    revenue = rng.gamma(shape=1.8, scale=25, size=n) * converted

    # segmentation columns
    channel = rng.choice(["Paid Search", "Organic", "Social", "Referral"], size=n, p=[0.3, 0.35, 0.2, 0.15])
    device = rng.choice(["Mobile", "Desktop"], size=n, p=[0.72, 0.28])

    df = pd.DataFrame(
        {
            "user_id": [f"U{idx:06d}" for idx in range(n)],
            "group": groups,
            "converted": converted,
            "revenue": revenue.round(2),
            "channel": channel,
            "device": device,
        }
    )
    return df


def two_prop_ztest(success_a, n_a, success_b, n_b):
    # pooled
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return np.nan, np.nan
    z = ((success_b / n_b) - (success_a / n_a)) / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    return z, pval


def ci_diff_prop(success_a, n_a, success_b, n_b, alpha=0.05):
    pa = success_a / n_a
    pb = success_b / n_b
    diff = pb - pa

    # unpooled SE
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = norm.ppf(1 - alpha / 2)
    lo = diff - z * se
    hi = diff + z * se
    return diff, lo, hi


def fmt_pct(x):
    return f"{x*100:.2f}%"


# ---------- UI ----------
st.title("Experiment Impact Studio")
st.caption("Upload experiment results and generate an executive-ready decision summary with uplift, confidence, and guardrails.")

with st.sidebar:
    st.header("1) Load data")
    mode = st.radio("Choose input", ["Use sample dataset", "Upload CSV"], index=0)

    if mode == "Upload CSV":
        file = st.file_uploader("Upload a CSV", type=["csv"])
        if file is None:
            st.info("Upload a CSV to begin.")
            st.stop()
        df = pd.read_csv(file)
    else:
        df = make_sample_ab()

    st.divider()
    st.header("2) Map columns")

    cols = df.columns.tolist()
    group_col = st.selectbox("Group column", cols, index=cols.index("group") if "group" in cols else 0)
    outcome_col = st.selectbox("Binary outcome column (0/1)", cols, index=cols.index("converted") if "converted" in cols else 0)

    metric_cols = [c for c in cols if c not in [group_col, outcome_col]]
    guardrail_col = st.selectbox("Guardrail metric (optional)", ["None"] + metric_cols, index=0)

    st.divider()
    st.header("3) Filters (optional)")
    filter_cols = [c for c in cols if c not in [group_col, outcome_col, guardrail_col]]
    seg_col = st.selectbox("Segment column", ["None"] + filter_cols, index=0)
    seg_val = None
    if seg_col != "None":
        options = sorted(df[seg_col].dropna().astype(str).unique().tolist())
        seg_val = st.selectbox("Segment value", options)

# apply filter
df_work = df.copy()
if seg_col != "None" and seg_val is not None:
    df_work = df_work[df_work[seg_col].astype(str) == str(seg_val)]

# Clean
df_work = df_work.dropna(subset=[group_col, outcome_col]).copy()
df_work[outcome_col] = pd.to_numeric(df_work[outcome_col], errors="coerce")
df_work = df_work[df_work[outcome_col].isin([0, 1])]

if df_work.empty:
    st.warning("No valid rows after applying filters / mapping columns.")
    st.stop()

# Identify groups
groups = sorted(df_work[group_col].astype(str).unique().tolist())
if len(groups) < 2:
    st.warning("Need at least 2 groups (e.g., control and variant).")
    st.stop()

# Choose control + variant
colA, colB = st.columns(2)
with colA:
    control_name = st.selectbox("Control group", groups, index=0)
with colB:
    variant_name = st.selectbox("Variant group", groups, index=1 if len(groups) > 1 else 0)

control = df_work[df_work[group_col].astype(str) == str(control_name)]
variant = df_work[df_work[group_col].astype(str) == str(variant_name)]

n_a, n_b = len(control), len(variant)
s_a, s_b = int(control[outcome_col].sum()), int(variant[outcome_col].sum())

p_a = s_a / n_a
p_b = s_b / n_b

abs_uplift = p_b - p_a
rel_uplift = (abs_uplift / p_a) if p_a > 0 else np.nan

z, pval = two_prop_ztest(s_a, n_a, s_b, n_b)
diff, lo, hi = ci_diff_prop(s_a, n_a, s_b, n_b)

# guardrail
guardrail_summary = None
if guardrail_col != "None":
    gA = pd.to_numeric(control[guardrail_col], errors="coerce")
    gB = pd.to_numeric(variant[guardrail_col], errors="coerce")
    guardrail_summary = {
        "control_mean": float(np.nanmean(gA)),
        "variant_mean": float(np.nanmean(gB)),
        "delta": float(np.nanmean(gB) - np.nanmean(gA)),
    }

# Decision summary
alpha = 0.05
significant = (pval is not None) and (not np.isnan(pval)) and (pval < alpha)

decision = "Ship" if (significant and abs_uplift > 0) else "Do not ship yet"
reason = []
reason.append(f"Control conversion {fmt_pct(p_a)} vs variant {fmt_pct(p_b)}")
reason.append(f"Absolute uplift {abs_uplift*100:.2f} pp (CI {lo*100:.2f} to {hi*100:.2f})")
if not np.isnan(pval):
    reason.append(f"P-value {pval:.4f} at alpha {alpha:.2f}")
else:
    reason.append("P-value unavailable (edge case)")

if guardrail_summary:
    reason.append(f"Guardrail mean delta: {guardrail_col} {guardrail_summary['delta']:.2f}")

# ---------- Layout ----------
topL, topR = st.columns([1.2, 0.8], gap="large")

with topL:
    st.subheader("Key Results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Control Conversion", fmt_pct(p_a))
    k2.metric("Variant Conversion", fmt_pct(p_b))
    k3.metric("Abs Uplift", f"{abs_uplift*100:.2f} pp")
    k4.metric("P-value", "NA" if np.isnan(pval) else f"{pval:.4f}")

    chart_df = pd.DataFrame({
        "Group": ["Control", "Variant"],
        "Conversion": [p_a, p_b]
    })
    fig = px.bar(chart_df, x="Group", y="Conversion", text=chart_df["Conversion"].map(lambda x: f"{x*100:.2f}%"))
    fig.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

with topR:
    st.subheader("Decision Summary")
    if decision == "Ship":
        st.success(decision)
    else:
        st.warning(decision)

    st.write("**Why:**")
    for r in reason:
        st.write(f"- {r}")

    st.write("")
    st.write("**Suggested next step:**")
    if decision == "Ship":
        st.write("Roll out incrementally, monitor guardrails, and validate impact across key segments.")
    else:
        st.write("Increase sample size, validate instrumentation, and check segment-level consistency before rollout.")

    if guardrail_summary:
        st.subheader("Guardrail Check")
        st.write(f"Control mean: {guardrail_summary['control_mean']:.2f}")
        st.write(f"Variant mean: {guardrail_summary['variant_mean']:.2f}")
        st.write(f"Delta: {guardrail_summary['delta']:.2f}")

st.subheader("Data Preview")
st.dataframe(df_work.head(50), use_container_width=True, height=320)

st.caption("Tip: Your CSV should have a group column (control/variant) and a binary outcome column (0/1). Optional guardrail metrics strengthen decisioning.")
