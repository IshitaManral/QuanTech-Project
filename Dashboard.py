"""
Interactive Workforce Scheduling Dashboard  (Streamlit + Plotly + PuLP/CBC)

Install : pip install streamlit plotly matplotlib "pulp==2.9.0" pandas openpyxl
Run     : streamlit run dashboard_app.py

Put your Excel file in the same folder (or upload it from the sidebar).
Sheets expected:
  - 'Day_to_Day' : columns ID, Name, then one column per day (1..31) with eaches handled
  - 'Outbound'   : monthly Plan in cell D2
"""

import glob
import inspect
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
import streamlit as st

st.set_page_config(page_title="Workforce Scheduling Dashboard", page_icon="📊", layout="wide")

# Make KPI cards readable: smaller value text, no "..." truncation
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricDelta"] > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BLUE, ORANGE, GREEN, RED, GREY = "#2563EB", "#F97316", "#16A34A", "#DC2626", "#64748B"
TEMPLATE = "plotly_white"

# Streamlit-version compatibility helpers
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_METRIC_HAS_BORDER = "border" in inspect.signature(st.metric).parameters


def show(fig):
    """Render a Plotly figure full-width on any Streamlit version."""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def metric(col, label, value, delta=None, **kw):
    """st.metric with a card border where the installed Streamlit supports it."""
    if _METRIC_HAS_BORDER:
        kw["border"] = True
    col.metric(label, value, delta, **kw)


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def find_local_excel():
    here = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(here, "Productivity*.xlsx")) or glob.glob(os.path.join(here, "*.xlsx"))
    files = [f for f in files if "Optimized_Roster" not in f and not os.path.basename(f).startswith("~$")]
    return files[0] if files else None


@st.cache_data(show_spinner="Reading Excel file...")
def load_data(source, file_key):
    src = io.BytesIO(source) if isinstance(source, bytes) else source
    xl = pd.ExcelFile(src)
    df = xl.parse("Day_to_Day").dropna(subset=["ID"])
    day_cols = [c for c in df.columns if str(c).isdigit()]
    days = [int(c) for c in day_cols]
    df = df.rename(columns={c: int(c) for c in day_cols})

    ids = df["ID"].tolist()
    # unique display labels (duplicate names get a (2), (3)... suffix)
    seen, labels = {}, []
    for nm in df["Name"].astype(str):
        seen[nm] = seen.get(nm, 0) + 1
        labels.append(nm if seen[nm] == 1 else f"{nm} ({seen[nm]})")

    eaches = df[days].fillna(0).values.astype(float)                  # employees x days
    worked = eaches > 0
    cap = np.array([eaches[i][worked[i]].mean() if worked[i].any() else 0.0
                    for i in range(len(ids))])                        # eaches per shift

    out = xl.parse("Outbound", header=None)
    plan = float(out.iloc[1, 3])                                      # D2
    if np.isnan(plan) or eaches.sum() <= 0:
        raise ValueError("Could not read the monthly Plan (Outbound!D2) or the daily eaches.")
    return dict(ids=ids, labels=labels, days=days, cap=cap, plan=plan,
                actual_by_day=eaches.sum(axis=0), hist=worked.astype(int))


# ------------------------------------------------------------------
# OPTIMISER (same model as the original script)
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Solving the integer program (up to the time limit)...")
def solve(_data, file_key, p):
    cap, days = _data["cap"], _data["days"]
    n, D = len(cap), len(days)
    demand = p["buffer"] * _data["plan"] * _data["actual_by_day"] / _data["actual_by_day"].sum()

    if n * p["max_shifts"] < D * p["min_staff"]:
        return dict(status="Infeasible (max shifts per person too low for the minimum staff per day)", params=p)

    model = pulp.LpProblem("Retail_Workforce_Scheduling", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(D)] for i in range(n)]

    # Objective: minimise total labour cost
    model += pulp.lpSum(p["wage"] * p["shift_hours"] * x[i][j] for i in range(n) for j in range(D))

    for j in range(D):
        # Constraint 1: daily workload covered by productivity of the people on duty
        model += pulp.lpSum(cap[i] * x[i][j] for i in range(n)) >= demand[j]
        # Constraint 2: minimum headcount per day
        model += pulp.lpSum(x[i][j] for i in range(n)) >= p["min_staff"]

    for i in range(n):
        # Constraint 3: monthly shift limits (fairness)
        model += pulp.lpSum(x[i]) >= p["min_shifts"]
        model += pulp.lpSum(x[i]) <= p["max_shifts"]
        # Constraint 4: weekly rest - max working days in any 7-day window
        for s in range(D - 6):
            model += pulp.lpSum(x[i][s:s + 7]) <= p["max_days_in_7"]
        # Constraint 5: max consecutive working days
        for s in range(D - p["max_consec"]):
            model += pulp.lpSum(x[i][s:s + p["max_consec"] + 1]) <= p["max_consec"]

    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=p["time_limit"]))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        return dict(status=status, params=p)

    # CBC reports "Optimal" even when it stopped at the time limit with a feasible solution
    proven = getattr(model, "sol_status", 1) == getattr(pulp, "LpSolutionOptimal", 1)
    M = np.array([[int(round(x[i][j].value() or 0)) for j in range(D)] for i in range(n)])
    return dict(status=status, proven=bool(proven), params=p, M=M, demand=demand)


# ------------------------------------------------------------------
# STATIC PNG DASHBOARD (same layout as the original matplotlib script)
# ------------------------------------------------------------------
def build_dashboard_png(coverage, M, labels, days, P, plan, cost_opt, cost_hist, utilisation):
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch

    n_emp, n_days = M.shape
    saving = cost_hist - cost_opt
    saving_pct = 100 * saving / cost_hist if cost_hist else 0
    sv_color = GREEN if saving >= 0 else RED
    kpis = [
        ("Optimised cost", f"Rs {cost_opt:,}", BLUE),
        ("Historical cost", f"Rs {cost_hist:,}", GREY),
        ("Savings", f"Rs {saving:,}", sv_color),
        ("Savings %", f"{saving_pct:.1f}%", sv_color),
        ("Avg staff / day", f"{coverage['Staff on duty'].mean():.1f}", ORANGE),
        ("Capacity used", f"{utilisation:.1f}%", BLUE),
    ]

    fig = Figure(figsize=(18, 14), facecolor="white")
    gs = fig.add_gridspec(4, 6, height_ratios=[0.8, 2.2, 2.2, 4])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.06, hspace=0.6, wspace=0.45)
    fig.suptitle("Workforce Scheduling Dashboard", fontsize=22, fontweight="bold", y=0.975)
    fig.text(0.5, 0.935, f"{n_emp} employees  |  {n_days} days  |  monthly plan {plan:,.0f} eaches",
             ha="center", fontsize=12, color=GREY)

    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.01",
                                    facecolor=color, alpha=0.12, edgecolor=color, transform=ax.transAxes))
        ax.text(0.5, 0.60, value, ha="center", va="center", fontsize=20, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=11, color=GREY, transform=ax.transAxes)

    ax1 = fig.add_subplot(gs[1, :])
    ax1.bar(coverage["Day"], coverage["Capacity scheduled"], color=BLUE, alpha=0.75, label="Capacity scheduled")
    ax1.plot(coverage["Day"], coverage["Workload (eaches)"], color=RED, marker="o", linewidth=2, label="Workload")
    for d, c in zip(coverage["Day"], coverage["Capacity scheduled"]):
        ax1.text(d, c * 0.03, f"{c:,.0f}", ha="center", va="bottom", fontsize=8, rotation=90,
                 color="white", fontweight="bold")
    ax1.set_title("Daily workload vs scheduled capacity", loc="left", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Eaches")
    ax1.set_xticks(days)
    ax1.set_xlim(min(days) - 0.7, max(days) + 0.7)
    ax1.set_ylim(0, max(coverage["Capacity scheduled"].max(), coverage["Workload (eaches)"].max()) * 1.15)
    ax1.legend(loc="upper right", ncol=2, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[2, :3])
    ax2.bar(coverage["Day"], coverage["Staff on duty"], color=ORANGE, alpha=0.85)
    ax2.axhline(P["min_staff"], color=RED, linestyle="--", linewidth=1.2, label=f"Minimum ({P['min_staff']})")
    ax2.axhline(coverage["Staff on duty"].mean(), color=GREY, linestyle=":", linewidth=1.5,
                label=f"Average ({coverage['Staff on duty'].mean():.1f})")
    for d, sv in zip(coverage["Day"], coverage["Staff on duty"]):
        ax2.text(d, sv, str(sv), ha="center", va="bottom", fontsize=8)
    ax2.set_title("Staff on duty per day", loc="left", fontweight="bold", fontsize=13)
    ax2.set_xlabel("Day of month")
    ax2.set_ylabel("Employees")
    ax2.set_xticks(days[::2])
    ax2.set_xlim(min(days) - 0.7, max(days) + 0.7)
    ax2.set_ylim(0, max(coverage["Staff on duty"].max(), 1) * 1.3)
    ax2.legend(frameon=False, loc="upper right", ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[2, 3:])
    totals = M.sum(axis=1)
    counts = pd.Series(totals).value_counts().sort_index()
    ax3.bar(counts.index, counts.values, color=GREEN, alpha=0.85, width=0.8)
    for k, v in counts.items():
        ax3.text(k, v, str(v), ha="center", va="bottom", fontsize=9)
    ax3.axvline(P["min_shifts"], color=GREY, linestyle="--", linewidth=1.2, label=f"Min ({P['min_shifts']})")
    ax3.axvline(P["max_shifts"], color=RED, linestyle="--", linewidth=1.2, label=f"Max ({P['max_shifts']})")
    ax3.set_title("Shifts per employee (fairness)", loc="left", fontweight="bold", fontsize=13)
    ax3.set_xlabel("Shifts worked in the month")
    ax3.set_ylabel("Number of employees")
    ax3.set_xlim(P["min_shifts"] - 1.5, P["max_shifts"] + 1.5)
    ax3.set_ylim(0, counts.max() * 1.15)
    ax3.legend(frameon=False, loc="upper right")
    ax3.spines[["top", "right"]].set_visible(False)

    ax4 = fig.add_subplot(gs[3, :])
    order = totals.argsort(kind="stable")[::-1]
    ax4.imshow(M[order], aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax4.set_title("Roster (green = working, sorted by shifts)", loc="left", fontweight="bold", fontsize=13)
    ax4.set_xlabel("Day of month")
    ax4.set_xticks(range(n_days))
    ax4.set_xticklabels(days, fontsize=9)
    ax4.set_xticks(np.arange(-0.5, n_days, 1), minor=True)
    ax4.grid(which="minor", color="white", linewidth=0.6)
    ax4.tick_params(which="minor", length=0)
    if n_emp <= 35:
        ax4.set_yticks(range(n_emp))
        ax4.set_yticklabels([labels[i] for i in order], fontsize=8)
    else:
        ax4.set_yticks([])
        ax4.set_ylabel(f"{n_emp} employees")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


# ------------------------------------------------------------------
# SIDEBAR: data source + parameters
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Excel file (optional)", type=["xlsx"],
                                    help="If empty, the app uses the .xlsx in this folder.")
if uploaded is not None:
    source, file_key = uploaded.getvalue(), f"upload:{uploaded.name}:{uploaded.size}"
else:
    path = find_local_excel()
    if path is None:
        st.title("📊 Workforce Scheduling Dashboard")
        st.info("No Excel file found. Put it next to this script or upload it from the sidebar.")
        st.stop()
    source, file_key = path, f"{path}:{os.path.getmtime(path)}"
    st.sidebar.caption(f"Using: {os.path.basename(path)}")

try:
    data = load_data(source, file_key)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the Excel file: {e}")
    st.stop()

# ---- Beginner-friendly hints, calculated from the uploaded data ----
_n_emp, _n_days = len(data["ids"]), len(data["days"])
_prod = data["cap"][data["cap"] > 0]
_avg_prod = float(_prod.mean()) if len(_prod) else 1.0
_need_shifts = data["plan"] / _avg_prod            # shifts needed to hit the plan at average productivity
_need_per_day = _need_shifts / _n_days             # average people needed per day
_need_per_person = _need_shifts / _n_emp           # fair share of shifts per person


def weekly_cap(days_in_month, per_week):
    """Most shifts one person can work in the month if at most `per_week` days in any 7 are allowed."""
    return (days_in_month // 7) * per_week + min(days_in_month % 7, per_week)


def explain_conflicts(p):
    """Plain-English reasons why a combination of settings cannot work."""
    msgs = []
    if p["min_shifts"] > p["max_shifts"]:
        msgs.append("**Min shifts per person** is higher than **Max shifts per person**. "
                    "The minimum must be less than or equal to the maximum.")
    wc = weekly_cap(_n_days, p["max_days_in_7"])
    if p["min_shifts"] > wc:
        msgs.append(f"**Min shifts per person ({p['min_shifts']})** cannot be reached: with at most "
                    f"{p['max_days_in_7']} working days in any 7, one person can work about {wc} days in a "
                    f"{_n_days}-day month. Lower the minimum or raise 'Max working days in any 7'.")
    if p["min_staff"] > _n_emp:
        msgs.append(f"**Minimum people per day ({p['min_staff']})** is more than the {_n_emp} employees you have.")
    elif _n_emp * min(p["max_shifts"], wc) < _n_days * p["min_staff"]:
        msgs.append("There are not enough shifts to keep the **minimum people per day** on duty. "
                    "Lower that minimum, or raise the max shifts / max working days in any 7.")
    return msgs


with st.sidebar.expander("ℹ️ What do these settings mean?"):
    st.markdown(
        f"""
**Quick facts from your data**
- {_n_emp} employees, {_n_days} days
- The monthly plan needs about **{_need_per_day:.0f} people per day** on average
  (about **{_need_per_person:.0f} shifts per person** for the month)

**How it works:** the optimiser picks who works on which day so the plan is covered
at the **lowest total cost**, while following every rule below.

**Rules of thumb**
- **Safety cushion:** 1.00 = just enough. 1.10 = 10% spare capacity. Usually 1.00 to 1.15.
- **Minimum people per day:** keep it at or below the average need above.
  Higher values add cost on quiet days.
- **Min / max shifts per person:** keep the minimum at or below the fair share above.
  Raising it forces extra shifts and extra cost.
- **Max days in any 7:** 6 means everyone gets at least one day off a week.
- **Max in a row:** 5 or 6 is typical. Lower = more rest, but a harder puzzle.
- **Wage and shift hours** only change the cost numbers, not who is scheduled.
- **Solver time:** 60 s for quick tests, 180 to 300 s for your final answer.
"""
    )

with st.sidebar.form("params"):
    st.subheader("Workload rules")
    buffer = st.slider(
        "Safety cushion (buffer)", 0.80, 1.50, 1.00, 0.05,
        help="How much extra capacity to schedule on top of the monthly plan.\n\n"
             "1.00 = schedule just enough people to hit the plan.\n"
             "1.10 = 10% spare, to absorb absences or busy days.\n"
             "Below 1.00 = deliberately under-staffed.\n\n"
             "Typical range: 1.00 to 1.15. Higher = safer but costs more.")
    min_staff = st.slider(
        "Minimum people per day", 0, 40, 10,
        help=f"The fewest people allowed on the floor on any day, even a quiet one.\n\n"
             f"Your plan needs about {_need_per_day:.0f} people per day on average. "
             f"Keep this at or below that. Going above it forces extra paid shifts on quiet days.\n\n"
             f"Rule of thumb: around half of the average need.")

    st.subheader("Fairness rules (per person)")
    c1, c2 = st.columns(2)
    min_shifts = c1.number_input(
        "Min shifts per person", 0, 31, 12,
        help=f"Everyone works at least this many days in the month, so nobody is left out.\n\n"
             f"A fair share is about {_need_per_person:.0f} shifts per person. "
             f"Setting the minimum above that forces extra shifts and raises cost.")
    max_shifts = c2.number_input(
        "Max shifts per person", 1, 31, 26,
        help=f"Nobody works more than this many days in the month.\n\n"
             f"Lower = workload is spread more evenly but you need more people on duty. "
             f"With the weekly-rest rule below, the practical ceiling is about "
             f"{weekly_cap(_n_days, 6)} days.")

    st.subheader("Rest rules")
    max_days_in_7 = st.slider(
        "Max working days in any 7", 4, 7, 6,
        help="Looking at any 7 days in a row, nobody works more than this many.\n\n"
             "6 = at least one day off every week (common choice).\n"
             "5 = two days off every week.\n"
             "7 = no weekly rest required.")
    max_consec = st.slider(
        "Max days in a row", 3, 10, 6,
        help="The longest streak of consecutive working days allowed.\n\n"
             "5 or 6 is typical. Lower = more rest for staff, but the optimiser "
             "has less freedom and may need more people or more time.")

    st.subheader("Cost")
    c3, c4 = st.columns(2)
    wage = c3.number_input(
        "Wage (Rs/hour)", 10, 1000, 100, 10,
        help="Hourly pay, same for everyone. Only changes the cost figures, not who works when.")
    shift_hours = c4.number_input(
        "Hours per shift", 1, 12, 8,
        help="Length of one shift. Only changes the cost figures, not who works when.")

    st.subheader("Solver")
    time_limit = st.slider(
        "Solver time limit (s)", 10, 300, 60, 10,
        help="How long the optimiser may search for the cheapest schedule.\n\n"
             "60 s is fine for quick tests. If you see 'time limit reached', "
             "try 180 to 300 s for a better or proven-best result.")
    submitted = st.form_submit_button("▶ Run optimiser", type="primary")

params = dict(buffer=float(buffer), min_staff=int(min_staff), min_shifts=int(min_shifts),
              max_shifts=int(max_shifts), max_days_in_7=int(max_days_in_7), max_consec=int(max_consec),
              wage=int(wage), shift_hours=int(shift_hours), time_limit=int(time_limit))

if "runs" not in st.session_state:
    st.session_state.runs = []

need_first_run = "result" not in st.session_state or st.session_state.get("file_key") != file_key
if need_first_run or submitted:
    problems = explain_conflicts(params)
    if problems:
        for msg in problems:
            st.error(msg)
        if need_first_run:
            st.stop()
    else:
        res = solve(data, file_key, params)
        if res["status"] != "Optimal":
            st.error(f"Solver status: **{res['status']}** - no schedule found. Relax the constraints "
                     "(lower the buffer or min staff, or raise max shifts) and run again.")
            if need_first_run:
                st.stop()
        else:
            if st.session_state.get("file_key") != file_key:
                st.session_state.runs = []
            st.session_state.result, st.session_state.file_key = res, file_key
            st.session_state.pop("png", None)
            M = res["M"]
            hist_cost = int(data["hist"].sum()) * params["wage"] * params["shift_hours"]
            cost = int(M.sum()) * params["wage"] * params["shift_hours"]
            st.session_state.runs.append({
                "Run": len(st.session_state.runs) + 1, "Buffer": params["buffer"],
                "Min staff": params["min_staff"], "Min shifts": params["min_shifts"],
                "Max shifts": params["max_shifts"], "Shifts": int(M.sum()), "Cost (Rs)": cost,
                "Savings (Rs)": hist_cost - cost,
            })

# ------------------------------------------------------------------
# DERIVED RESULTS
# ------------------------------------------------------------------
res = st.session_state.result
P, M, demand = res["params"], res["M"], res["demand"]
days, labels, cap = data["days"], data["labels"], data["cap"]
n, D = M.shape

shift_cost = P["wage"] * P["shift_hours"]
total_shifts, hist_shifts = int(M.sum()), int(data["hist"].sum())
cost_opt, cost_hist = total_shifts * shift_cost, hist_shifts * shift_cost
saving = cost_hist - cost_opt

coverage = pd.DataFrame({
    "Day": days,
    "Workload (eaches)": np.round(demand).astype(int),
    "Capacity scheduled": np.round(cap @ M).astype(int),
    "Staff on duty": M.sum(axis=0),
})
coverage["Surplus"] = coverage["Capacity scheduled"] - coverage["Workload (eaches)"]
coverage["Utilisation %"] = (100 * coverage["Workload (eaches)"]
                             / coverage["Capacity scheduled"].replace(0, np.nan)).round(1)
utilisation = 100 * coverage["Workload (eaches)"].sum() / max(coverage["Capacity scheduled"].sum(), 1)

emp = pd.DataFrame({
    "Employee": labels, "ID": data["ids"],
    "Productivity (eaches/shift)": np.round(cap, 1),
    "Shifts (optimised)": M.sum(axis=1), "Shifts (historical)": data["hist"].sum(axis=1),
})
emp["Change"] = emp["Shifts (optimised)"] - emp["Shifts (historical)"]

# ------------------------------------------------------------------
# HEADER + KPIs
# ------------------------------------------------------------------
st.title("📊 Workforce Scheduling Dashboard")
solver_txt = "Optimal" if res.get("proven", True) else "Best found (time limit reached)"
st.caption(f"{n} employees  |  {D} days  |  monthly plan {data['plan']:,.0f} eaches  |  "
           f"buffer {P['buffer']:.2f}  |  solver: {solver_txt}")
if not res.get("proven", True):
    st.warning("The solver hit the time limit before proving optimality. The schedule is valid, but may cost "
               "slightly more than the true optimum. Raise the solver time limit to tighten it.")

# Two rows of three cards so values and deltas are not truncated
row1 = st.columns(3)
row2 = st.columns(3)
k1, k2, k3 = row1
k4, k5, k6 = row2

metric(k1, "Optimised cost", f"Rs {cost_opt:,}", f"{cost_opt - cost_hist:,} vs historical", delta_color="inverse")
metric(k2, "Historical cost", f"Rs {cost_hist:,}", f"{hist_shifts:,} shifts", delta_color="off")
metric(k3, "Savings", f"Rs {saving:,}", f"{100 * saving / cost_hist:.1f}%" if cost_hist else None)
metric(k4, "Total shifts", f"{total_shifts:,}", f"{total_shifts - hist_shifts:,} vs historical", delta_color="inverse")
metric(k5, "Avg staff / day", f"{coverage['Staff on duty'].mean():.1f}", f"peak {coverage['Staff on duty'].max()}",
       delta_color="off")
metric(k6, "Capacity used", f"{utilisation:.1f}%", "no margin" if utilisation > 98 else None,
       delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Employees", "Roster", "Scenarios", "Data & export"])

# ------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------
with tab1:
    fig = go.Figure()
    fig.add_bar(x=coverage["Day"], y=coverage["Capacity scheduled"], name="Capacity scheduled",
                marker_color=BLUE, opacity=0.75,
                hovertemplate="Day %{x}<br>Capacity: %{y:,}<extra></extra>")
    fig.add_scatter(x=coverage["Day"], y=coverage["Workload (eaches)"], name="Workload", mode="lines+markers",
                    line=dict(color=RED, width=3), hovertemplate="Day %{x}<br>Workload: %{y:,}<extra></extra>")
    fig.update_layout(template=TEMPLATE, title="Daily workload vs scheduled capacity", height=380,
                      xaxis=dict(title="Day of month", dtick=1, rangeslider=dict(visible=True, thickness=0.06)),
                      yaxis_title="Eaches", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                      hovermode="x unified", margin=dict(t=70))
    show(fig)

    fig2 = go.Figure()
    fig2.add_bar(x=coverage["Day"], y=coverage["Staff on duty"], marker_color=ORANGE, name="Staff on duty",
                 text=coverage["Staff on duty"], textposition="outside",
                 hovertemplate="Day %{x}<br>%{y} employees<extra></extra>")
    fig2.add_hline(y=P["min_staff"], line_dash="dash", line_color=RED, annotation_text=f"Minimum ({P['min_staff']})")
    fig2.add_hline(y=coverage["Staff on duty"].mean(), line_dash="dot", line_color=GREY,
                   annotation_text="Average", annotation_position="bottom right")
    fig2.update_layout(template=TEMPLATE, title="Staff on duty per day", height=340,
                       xaxis=dict(title="Day of month", dtick=1), yaxis_title="Employees", showlegend=False,
                       margin=dict(t=60))
    show(fig2)

    st.subheader("Day explorer")
    day = st.select_slider("Pick a day", options=days, value=days[0])
    j = days.index(day)
    row = coverage.iloc[j]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Workload", f"{row['Workload (eaches)']:,}")
    d2.metric("Capacity scheduled", f"{row['Capacity scheduled']:,}")
    d3.metric("Staff on duty", int(row["Staff on duty"]))
    d4.metric("Surplus", f"{row['Surplus']:,}")
    on_duty = pd.DataFrame({"Employee": labels, "Productivity (eaches/shift)": np.round(cap, 1),
                            "Working": M[:, j] == 1})
    on_duty = on_duty[on_duty["Working"]].drop(columns="Working").sort_values(
        "Productivity (eaches/shift)", ascending=False)
    with st.expander(f"Who works on day {day} ({len(on_duty)} people)"):
        st.dataframe(on_duty, hide_index=True)

    with st.expander("How the model works"):
        st.markdown(
            f"""
**Decision variable:** `x[e,d] = 1` if employee *e* works on day *d* (binary), {n} × {D} = {n * D:,} variables.

**Objective:** minimise labour cost = Σ wage × shift hours × `x[e,d]`.

**Constraints**
1. **Workload coverage:** for every day *d*, Σ productivity[e] × `x[e,d]` ≥ workload[d]
   (monthly plan spread over days using the real daily pattern, times the service buffer).
2. **Minimum headcount:** Σ `x[e,d]` ≥ {P['min_staff']} each day.
3. **Fair monthly load:** {P['min_shifts']} ≤ Σ `x[e,d]` ≤ {P['max_shifts']} for every employee.
4. **Weekly rest:** at most {P['max_days_in_7']} working days in any 7 consecutive days.
5. **Consecutive days:** at most {P['max_consec']} working days in a row.

**Inputs from Excel:** productivity = average eaches per shift on the days the employee worked
(`Day_to_Day`); a day with 0 eaches counts as "not worked" in the historical baseline.
"""
        )

# ------------------------------------------------------------------
# TAB 2: EMPLOYEES
# ------------------------------------------------------------------
with tab2:
    left, right = st.columns([1, 1])
    counts = emp["Shifts (optimised)"].value_counts().sort_index()
    figh = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=GREEN, text=counts.values,
                            textposition="outside", hovertemplate="%{y} employees worked %{x} shifts<extra></extra>"))
    figh.add_vline(x=P["min_shifts"], line_dash="dash", line_color=GREY, annotation_text="Min")
    figh.add_vline(x=P["max_shifts"], line_dash="dash", line_color=RED, annotation_text="Max")
    figh.update_layout(template=TEMPLATE, title="Shifts per employee (fairness)", height=360,
                       xaxis_title="Shifts in the month", yaxis_title="Employees", margin=dict(t=60))
    with left:
        show(figh)

    figs = go.Figure(go.Scatter(
        x=emp["Productivity (eaches/shift)"], y=emp["Shifts (optimised)"], mode="markers",
        text=emp["Employee"], marker=dict(size=10, color=BLUE, opacity=0.7),
        hovertemplate="%{text}<br>%{x} eaches/shift<br>%{y} shifts<extra></extra>"))
    figs.update_layout(template=TEMPLATE, title="Productivity vs shifts assigned", height=360,
                       xaxis_title="Eaches per shift", yaxis_title="Shifts assigned", margin=dict(t=60))
    with right:
        show(figs)

    q = st.text_input("Search employee", placeholder="type part of a name...")
    table = emp[emp["Employee"].str.contains(q, case=False, na=False, regex=False)] if q else emp
    st.dataframe(
        table.sort_values("Shifts (optimised)", ascending=False), hide_index=True,
        column_config={"Shifts (optimised)": st.column_config.ProgressColumn(
            "Shifts (optimised)", min_value=0, max_value=int(P["max_shifts"]), format="%d")})

    pick = st.selectbox("Employee detail", ["-"] + labels)
    if pick != "-":
        i = labels.index(pick)
        worked_days = [d for d, v in zip(days, M[i]) if v]
        e1, e2, e3 = st.columns(3)
        e1.metric("Shifts (optimised)", int(M[i].sum()), int(M[i].sum() - data["hist"][i].sum()), delta_color="off")
        e2.metric("Productivity", f"{cap[i]:,.0f} eaches/shift")
        e3.metric("Cost", f"Rs {int(M[i].sum()) * shift_cost:,}")
        st.write("**Days worked:** " + (", ".join(map(str, worked_days)) or "none"))

# ------------------------------------------------------------------
# TAB 3: ROSTER
# ------------------------------------------------------------------
with tab3:
    r1, r2, r3 = st.columns([1, 1, 2])
    view = r1.radio("View", ["Optimised", "Historical"], horizontal=True)
    sort_by = r2.radio("Sort by", ["Shifts", "Name"], horizontal=True)
    chosen = r3.multiselect("Filter employees (empty = all)", labels)

    mat = M if view == "Optimised" else data["hist"]
    idx = list(range(n))
    if chosen:
        idx = [labels.index(c) for c in chosen]
    idx = sorted(idx, key=(lambda i: -mat[i].sum()) if sort_by == "Shifts" else (lambda i: labels[i].lower()))

    z = mat[idx]
    txt = np.where(z == 1, "Working", "Off")
    figr = go.Figure(go.Heatmap(
        z=z, x=days, y=[labels[i] for i in idx], text=txt, zmin=0, zmax=1, xgap=1, ygap=1,
        colorscale=[[0, "#EEF2F7"], [1, GREEN]], showscale=False,
        hovertemplate="%{y}<br>Day %{x}: %{text}<extra></extra>"))
    figr.update_layout(template=TEMPLATE, title=f"{view} roster (green = working)",
                       height=max(420, 16 * len(idx) + 120), xaxis=dict(dtick=1, side="top"),
                       yaxis=dict(autorange="reversed", tickfont=dict(size=10)), margin=dict(t=90))
    show(figr)

# ------------------------------------------------------------------
# TAB 4: SCENARIOS
# ------------------------------------------------------------------
with tab4:
    st.write("Change the sliders in the sidebar and press **Run optimiser**. Every run is saved here "
             "so you can compare scenarios side by side.")
    runs = pd.DataFrame(st.session_state.runs)
    if runs.empty:
        st.info("No runs yet.")
    else:
        figc = go.Figure(go.Bar(x=[f"Run {r}" for r in runs["Run"]], y=runs["Cost (Rs)"], marker_color=BLUE,
                                text=[f"Rs {v:,}" for v in runs["Cost (Rs)"]], textposition="outside",
                                hovertemplate="%{x}<br>Rs %{y:,}<extra></extra>"))
        figc.add_hline(y=cost_hist, line_dash="dash", line_color=GREY, annotation_text="Historical cost")
        figc.update_layout(template=TEMPLATE, title="Labour cost by scenario", height=360,
                           yaxis_title="Rs", margin=dict(t=60))
        show(figc)
        st.dataframe(runs, hide_index=True)
        if st.button("Clear scenario history"):
            st.session_state.runs = []
            st.rerun()

# ------------------------------------------------------------------
# TAB 5: DATA & EXPORT
# ------------------------------------------------------------------
with tab5:
    roster_df = pd.DataFrame(M, columns=days, index=labels)
    roster_df["Total Shifts"] = roster_df.sum(axis=1)

    st.subheader("Daily coverage")
    st.dataframe(coverage, hide_index=True)

    summary = pd.DataFrame({
        "Metric": ["Employees", "Days", "Monthly plan (eaches)", "Optimised shifts", "Historical shifts",
                   "Optimised cost (Rs)", "Historical cost (Rs)", "Savings (Rs)", "Capacity used (%)"],
        "Value": [n, D, round(data["plan"]), total_shifts, hist_shifts, cost_opt, cost_hist, saving,
                  round(utilisation, 1)],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        roster_df.to_excel(writer, sheet_name="Roster")
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        emp.to_excel(writer, sheet_name="Employees", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    st.subheader("Export")
    b1, b2, b3 = st.columns(3)
    b1.download_button("⬇ Excel (roster + coverage + employees + summary)", buf.getvalue(),
                       file_name="Optimized_Roster.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    b2.download_button("⬇ Roster CSV", roster_df.to_csv().encode("utf-8"),
                       file_name="Optimized_Roster.csv", mime="text/csv")
    if b3.button("🖼 Prepare dashboard PNG"):
        with st.spinner("Drawing the dashboard image..."):
            st.session_state.png = build_dashboard_png(
                coverage, M, labels, days, P, data["plan"], cost_opt, cost_hist, utilisation)
    if "png" in st.session_state:
        b3.download_button("⬇ Download roster_chart.png", st.session_state.png,
                           file_name="roster_chart.png", mime="image/png")