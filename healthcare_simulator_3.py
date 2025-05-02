import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("CGS Three-Layer Monte Carlo Simulator for Healthcare Expenditures")

st.markdown("""
This simulator models healthcare costs using a log-normal distribution, including:

1. A base Monte Carlo of expenditures per person for population defined by distribution characteristics (for now - we will later use the actual raw data)
2. Subsampling  to simulate an employer group with truncation for the specific deductible and the excess layer, then summed group costs  
3. Final sampling from those group sums to derive total claims against a book of premium based on number of policies written, giving the distribution and therefore what we should charge for the stop loss based on the original distribution.  

It calculates per-person and per-person-per-month breakdowns.
""")

# Input: distribution parameters
mu = st.number_input("Log-Mean (μ)", value=8.8, step=0.1)
sigma = st.number_input("Log-Std Dev (σ)", value=1.3, step=0.1)
n_samples = st.number_input("Initial Monte Carlo Samples", value=100000, min_value=1000, step=1000)

# Subsampling and truncation settings
st.markdown("### First-Level Subsampling (From Base Distribution)")
subsample_size = st.number_input("Subsample Size", value=150, min_value=1)
n_subsample_runs = st.number_input("Subsample Monte Carlo Runs", value=1000, min_value=100)
lower_thresh = st.number_input("Specific Deductible ($)", value=125000.0, step=100.0)
upper_thresh = st.number_input("Excess Layer Level ($)", value=1000000.0, step=100.0)

# Final Monte Carlo sampling from truncated sums
st.markdown("### Second-Level Subsampling (Total Book Stop Loss Policy Claims)")
final_sample_size = st.number_input("Final Subsample Size (How many policies)", value=50, min_value=1)
n_final_runs = st.number_input("Final Monte Carlo Runs", value=1000, min_value=100)

# Bootstrap for original mean
n_bootstrap = st.number_input("Bootstrap Samples for CI (Original Mean)", value=1000, min_value=100, step=100)

if st.button("Run Full 3-Layer Simulation"):
    # Step 1: Full base sample
    full_sample = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)

    # Summary
    mean_val = np.mean(full_sample)
    median_val = np.median(full_sample)
    percentile_95 = np.percentile(full_sample, 95)
    percentile_99 = np.percentile(full_sample, 99)

    # Bootstrap CI for mean
    means = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(full_sample, size=n_samples, replace=True)
        means.append(np.mean(resample))
    ci_lower = np.percentile(means, 2.5)
    ci_upper = np.percentile(means, 97.5)

    # Step 2: Truncated subsample Monte Carlo
    truncated_sums = []
    for _ in range(n_subsample_runs):
        subsample = np.random.choice(full_sample, size=subsample_size, replace=False)
        truncated = subsample[(subsample >= lower_thresh) & (subsample <= upper_thresh)]
        truncated_sums.append(np.sum(truncated))
    truncated_sums = np.array(truncated_sums)

    # Step 3: Final-level Monte Carlo from truncated sums
    final_sums = []
    for _ in range(n_final_runs):
        final_subsample = np.random.choice(truncated_sums, size=final_sample_size, replace=True)
        final_sums.append(np.sum(final_subsample))
    final_sums = np.array(final_sums)

    # Display summaries
    st.subheader("Full Sample Statistics")
    st.write(f"Mean: ${mean_val:,.2f}  (95% CI: ${ci_lower:,.2f} – ${ci_upper:,.2f})")
    st.write(f"Median: ${median_val:,.2f}")
    st.write(f"95th Percentile: ${percentile_95:,.2f}")
    st.write(f"99th Percentile: ${percentile_99:,.2f}")

    st.subheader("First-Level (Specific Stop Loss Claims) Monte Carlo")
    st.write(f"Mean of Specific Claims Per Employer: ${np.mean(truncated_sums):,.2f}")
    st.write(f"Std. Dev of Specific Claims Per Employer: ${np.std(truncated_sums):,.2f}")

    fig1, ax1 = plt.subplots()
    ax1.hist(truncated_sums, bins=50, color='orange', edgecolor='black', alpha=0.8)
    ax1.set_title("Distribution of Truncated Subsample Sums")
    ax1.set_xlabel("Truncated Subsample Sum ($)")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    st.subheader("Final Monte Carlo (For a Book)")
    st.write(f"Mean of Final Sums: ${np.mean(final_sums):,.2f}")
    st.write(f"Std. Dev: ${np.std(final_sums):,.2f}")
    st.write(f"95% Range: ${np.percentile(final_sums, 2.5):,.2f} – ${np.percentile(final_sums, 97.5):,.2f}")

    fig2, ax2 = plt.subplots()
    ax2.hist(final_sums, bins=50, color='green', edgecolor='black', alpha=0.8)
    ax2.set_title("Distribution of Final Monte Carlo (Sum of Sums)")
    ax2.set_xlabel("Total Sum ($)")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # Padding for export and table
    max_len = max(len(full_sample), len(truncated_sums), len(final_sums))
    original_padded = np.append(full_sample, [np.nan] * (max_len - len(full_sample)))
    truncated_padded = np.append(truncated_sums, [np.nan] * (max_len - len(truncated_sums)))
    final_padded = np.append(final_sums, [np.nan] * (max_len - len(final_sums)))

    # Per person / per month metrics
    truncated_per_person = truncated_padded / subsample_size
    final_per_person = final_padded / (final_sample_size * subsample_size)
    truncated_monthly = truncated_per_person / 12
    final_monthly = final_per_person / 12

    df_export = pd.DataFrame({
        "OriginalSample": original_padded,
        "TruncatedSums": truncated_padded,
        "TruncatedSums_per_person": truncated_per_person,
        "TruncatedSums_per_person_per_month": truncated_monthly,
        "FinalSums": final_padded,
        "FinalSums_per_person": final_per_person,
        "FinalSums_per_person_per_month": final_monthly
    })

    st.subheader("Preview of Simulation Data with Monthly Metrics")
    st.dataframe(df_export.head(100))

    # Per-person-per-month stats
    st.subheader("Final Per-Person-Per-Month Summary Statistics")
    final_pppm = final_monthly[~np.isnan(final_monthly)]
    avg_pppm = np.mean(final_pppm)
    std_pppm = np.std(final_pppm)
    p5 = np.percentile(final_pppm, 5)
    p95 = np.percentile(final_pppm, 95)

    st.write(f"**Average Per Person Per Month**: ${avg_pppm:,.2f}")
    st.write(f"**Standard Deviation**: ${std_pppm:,.2f}")
    st.write(f"**5th–95th Percentile Range**: ${p5:,.2f} – ${p95:,.2f}")

    fig3, ax3 = plt.subplots()
    ax3.hist(final_pppm, bins=50, color='purple', edgecolor='black', alpha=0.7)
    ax3.set_title("Final Per-Person-Per-Month Distribution")
    ax3.set_xlabel("Monthly Cost ($)")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

    csv = df_export.to_csv(index=False)
    st.download_button("Download CSV of All Monte Carlo Levels", csv, file_name="3layer_mc_results.csv", mime="text/csv")

    st.success("Full 3-layer simulation completed successfully.")
