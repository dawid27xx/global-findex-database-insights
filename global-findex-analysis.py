import pandas as pd
import matplotlib.pyplot as plt

def importDataset():
    columns = [
        "countrynewwb", "year", "group",
        "account_t_d", "fiaccount_t_d", "mobileaccount_t_d", "g20_any",
        "borrow_any_t_d", "fin22a_22a1_22g_d", "fin22b", "fin22c", "fin22f"
    ]
    return pd.read_csv("GlobalFindexDatabase2025.csv", usecols=columns)

def cleanData(df):
    return df.dropna(subset=["year", "group"])

def addDigitalGap(df):
    df = df.copy()
    df["digital_gap"] = df["account_t_d"] - df["g20_any"]
    return df

def topDigitalGap2024(df, n=5):
    df = addDigitalGap(df)
    df_2024 = df[(df["year"] == 2024) & (df["group"] == "all")]
    return (df_2024
            .sort_values("digital_gap", ascending=False)
            [["countrynewwb", "account_t_d", "g20_any", "digital_gap"]]
            .head(n))

def topDigitalGapHistorical(df, years=(2011, 2014, 2017, 2021, 2024), n=5):
    df = addDigitalGap(df)
    top_countries = topDigitalGap2024(df, n)["countrynewwb"].unique()
    return (df[(df["year"].isin(years)) &
               (df["group"] == "all") &
               (df["countrynewwb"].isin(top_countries))]
            [["countrynewwb", "year", "account_t_d", "g20_any", "digital_gap"]]
            .sort_values(["countrynewwb", "year"]))

def plotHistoricalGaps(hist):
    countries = hist["countrynewwb"].unique()
    for country in countries:
        subset = hist[hist["countrynewwb"] == country]
        plt.figure(figsize=(8, 5))
        plt.plot(subset["year"], subset["account_t_d"], marker="o", label="Account Ownership")
        plt.plot(subset["year"], subset["g20_any"], marker="o", label="Digital Payments")
        plt.plot(subset["year"], subset["digital_gap"], marker="o", label="Digital Gap", linewidth=2.5, color="red")
        plt.title(f"Digital Banking Underutilisation in {country}")
        plt.xlabel("Year")
        plt.ylabel("Proportion of Adults (0–1)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

def addBorrowingMetrics(df, include_store_credit=False):
    df = df.copy()
    df["formal_share"] = df["fin22a_22a1_22g_d"]
    cols = ["fin22b", "fin22c"]
    if include_store_credit and "fin22f" in df.columns:
        cols.append("fin22f")
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df[cols] = df[cols].fillna(0)
    df["informal_share"] = df[cols].sum(axis=1)
    df["borrow_any"] = df["borrow_any_t_d"]
    df["formalisation_gap"] = df["informal_share"] - df["formal_share"]
    df["formal_headroom"] = df["borrow_any"] - df["formal_share"]
    return df

def topUnderFormalised2024(df, n=5):
    d = addBorrowingMetrics(df)
    d = d[(d["group"] == "all") & (d["year"] == 2024)]
    d = d[~d["countrynewwb"].str.contains("Asia|Europe|Africa|America|income|world", case=False, na=False)]
    return (d.sort_values("formalisation_gap", ascending=False)
              [["countrynewwb", "borrow_any", "formal_share", "informal_share",
                "formalisation_gap", "formal_headroom"]]
              .head(n))

def borrowingHistorical(df, countries, years=(2011, 2014, 2017, 2021, 2024)):
    d = addBorrowingMetrics(df)
    hist = d[(d["group"] == "all") &
             (d["countrynewwb"].isin(countries)) &
             (d["year"].isin(years))]
    return hist[["countrynewwb", "year", "borrow_any", "formal_share", "informal_share",
                 "formalisation_gap", "formal_headroom"]].sort_values(["countrynewwb","year"])

def plotBorrowingSmallMultiples(hist):
    countries = hist["countrynewwb"].unique()
    n = len(countries)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows), sharex=False, sharey=True)
    axes = axes.flatten()
    for i, c in enumerate(countries):
        sub = hist[hist["countrynewwb"] == c].sort_values("year")
        ax = axes[i]
        ax.plot(sub["year"], sub["formal_share"], marker="o", label="Formal")
        ax.plot(sub["year"], sub["informal_share"], marker="o", label="Informal")
        ax.set_title(c)
        ax.set_xlabel("Year")
        ax.set_ylabel("Share (0–1)")
        ax.grid(True, linestyle="--", alpha=0.5)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def main():
    df = importDataset()
    df_clean = cleanData(df)
    hist = topDigitalGapHistorical(df_clean, years=(2011, 2014, 2017, 2021, 2024), n=5)
    plotHistoricalGaps(hist)
    top2024 = topUnderFormalised2024(df_clean, n=5)
    countries = top2024["countrynewwb"].tolist()
    bhist = borrowingHistorical(df_clean, countries, years=(2011, 2014, 2017, 2021, 2024))
    print(top2024)
    plotBorrowingSmallMultiples(bhist)

if __name__ == "__main__":
    main()
