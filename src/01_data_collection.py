"""
QM640 Capstone: Notebook 01: Data Collection & Merge
Run this script to produce data/processed/merged_dataset_2023.csv
"""

import pandas as pd
import numpy as np
import os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# ── ISO3 country code reference (to standardise all datasets) ─────────────
# We use World Bank country metadata as the master ISO3 list
META_CSV = glob.glob(os.path.join(RAW, "gdp_raw", "Metadata_Country_*.csv"))[0]
meta = pd.read_csv(META_CSV, skiprows=0)
meta = meta[["Country Code", "Region"]].rename(columns={"Country Code": "iso3"})
meta = meta[meta["Region"].notna()]  # drop aggregates (World, regions, etc.)
print(f"World Bank country list: {len(meta)} economies")

# ═══════════════════════════════════════════════════════════════════
# 1. Basel AML Index 2023
# ═══════════════════════════════════════════════════════════════════
import xlrd

wb     = xlrd.open_workbook(os.path.join(RAW, "basel_aml_2023.xls"))
sh     = wb.sheet_by_name("2023")
rows   = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(3, sh.nrows)]
basel  = pd.DataFrame(rows, columns=["country_name", "basel_score", "basel_rank"])
basel  = basel[basel["country_name"] != ""].copy()
basel["basel_score"] = pd.to_numeric(basel["basel_score"], errors="coerce")
print(f"Basel AML 2023: {len(basel)} countries")

# Manual ISO3 mapping for Basel country names (uppercase)
BASEL_ISO3 = {
    "HAITI": "HTI", "CHAD": "TCD", "MYANMAR": "MMR", "CONGO, DEM. REP.": "COD",
    "MALI": "MLI", "MOZAMBIQUE": "MOZ", "TANZANIA": "TZA", "LAOS": "LAO",
    "ANGOLA": "AGO", "NIGERIA": "NGA", "CAMBODIA": "KHM", "KENYA": "KEN",
    "MADAGASCAR": "MDG", "GUINEA": "GIN", "NIGER": "NER", "SENEGAL": "SEN",
    "CAMEROON": "CMR", "BOLIVIA": "BOL", "PERU": "PER", "GHANA": "GHA",
    "ECUADOR": "ECU", "ZAMBIA": "ZMB", "ETHIOPIA": "ETH", "IRAQ": "IRQ",
    "BANGLADESH": "BGD", "PAKISTAN": "PAK", "PHILIPPINES": "PHL",
    "INDONESIA": "IDN", "INDIA": "IND", "VIETNAM": "VNM", "SRI LANKA": "LKA",
    "RUSSIA": "RUS", "TURKEY": "TUR", "UKRAINE": "UKR", "IRAN": "IRN",
    "VENEZUELA": "VEN", "ZIMBABWE": "ZWE", "SUDAN": "SDN", "LIBYA": "LBY",
    "ALGERIA": "DZA", "EGYPT": "EGY", "MOROCCO": "MAR", "TUNISIA": "TUN",
    "SOUTH AFRICA": "ZAF", "BRAZIL": "BRA", "ARGENTINA": "ARG",
    "COLOMBIA": "COL", "MEXICO": "MEX", "CHINA": "CHN", "THAILAND": "THA",
    "MALAYSIA": "MYS", "SOUTH KOREA": "KOR", "JAPAN": "JPN",
    "AUSTRALIA": "AUS", "CANADA": "CAN", "UNITED STATES": "USA",
    "UNITED KINGDOM": "GBR", "GERMANY": "DEU", "FRANCE": "FRA",
    "ITALY": "ITA", "SPAIN": "ESP", "NETHERLANDS": "NLD",
    "SWEDEN": "SWE", "NORWAY": "NOR", "DENMARK": "DNK", "FINLAND": "FIN",
    "SWITZERLAND": "CHE", "AUSTRIA": "AUT", "BELGIUM": "BEL",
    "PORTUGAL": "PRT", "GREECE": "GRC", "POLAND": "POL",
    "CZECH REPUBLIC": "CZE", "HUNGARY": "HUN", "ROMANIA": "ROU",
    "BULGARIA": "BGR", "SLOVAKIA": "SVK", "SLOVENIA": "SVN",
    "CROATIA": "HRV", "ESTONIA": "EST", "LATVIA": "LVA", "LITHUANIA": "LTU",
    "LUXEMBOURG": "LUX", "IRELAND": "IRL", "ICELAND": "ISL",
    "NEW ZEALAND": "NZL", "SINGAPORE": "SGP", "HONG KONG": "HKG",
    "ISRAEL": "ISR", "SAUDI ARABIA": "SAU", "UAE": "ARE",
    "UNITED ARAB EMIRATES": "ARE", "QATAR": "QAT", "KUWAIT": "KWT",
    "BAHRAIN": "BHR", "OMAN": "OMN", "JORDAN": "JOR", "LEBANON": "LBN",
    "SYRIA": "SYR", "AFGHANISTAN": "AFG", "NEPAL": "NPL", "BHUTAN": "BTN",
    "MYANMAR/BURMA": "MMR", "INDONESIA ": "IDN",
    "TRINIDAD AND TOBAGO": "TTO", "JAMAICA": "JAM", "CUBA": "CUB",
    "DOMINICAN REPUBLIC": "DOM", "COSTA RICA": "CRI", "PANAMA": "PAN",
    "HONDURAS": "HND", "GUATEMALA": "GTM", "EL SALVADOR": "SLV",
    "NICARAGUA": "NIC", "PARAGUAY": "PRY", "URUGUAY": "URY", "CHILE": "CHL",
    "RWANDA": "RWA", "BURUNDI": "BDI", "UGANDA": "UGA", "SOMALIA": "SOM",
    "ERITREA": "ERI", "DJIBOUTI": "DJI", "SOUTH SUDAN": "SSD",
    "CENTRAL AFRICAN REPUBLIC": "CAF", "GABON": "GAB", "REPUBLIC OF CONGO": "COG",
    "EQUATORIAL GUINEA": "GNQ", "SAO TOME AND PRINCIPE": "STP",
    "CAPE VERDE": "CPV", "GAMBIA": "GMB", "GUINEA-BISSAU": "GNB",
    "SIERRA LEONE": "SLE", "LIBERIA": "LBR", "IVORY COAST": "CIV",
    "COTE D'IVOIRE": "CIV", "BURKINA FASO": "BFA", "TOGO": "TGO",
    "BENIN": "BEN", "MAURITANIA": "MRT", "MALAWI": "MWI",
    "LESOTHO": "LSO", "ESWATINI": "SWZ", "SWAZILAND": "SWZ",
    "NAMIBIA": "NAM", "BOTSWANA": "BWA",
    # Additional mappings for Basel-specific names
    "THE DEMOCRATIC REPUBLIC OF THE CONGO": "COD",
    "SURINAME": "SUR", "SOLOMON ISLANDS": "SLB", "TONGA": "TON",
    "SAINT KITTS AND NEVIS": "KNA", "MACAO SAR, CHINA": "MAC", "PALAU": "PLW",
    "TÜRKIYE": "TUR", "BAHAMAS": "BHS", "VANUATU": "VUT", "BARBADOS": "BRB",
    "SAINT LUCIA": "LCA", "SEYCHELLES": "SYC", "MONGOLIA": "MNG",
    "GRENADA": "GRD", "SAMOA": "WSM", "HONG KONG SAR, CHINA": "HKG",
    "ANTIGUA AND BARBUDA": "ATG", "ARUBA": "ABW", "MAURITIUS": "MUS",
    "CYPRUS": "CYP", "MALTA": "MLT", "LIECHTENSTEIN": "LIE",
    "KOREA, SOUTH": "KOR", "DOMINICA": "DMA", "BRUNEI DARUSSALAM": "BRN",
    "MACEDONIA NORTH": "MKD", "TAIWAN": "TWN", "SAN MARINO": "SMR",
    "ANDORRA": "AND",
    "GEORGIA": "GEO", "ARMENIA": "ARM", "AZERBAIJAN": "AZE",
    "KAZAKHSTAN": "KAZ", "UZBEKISTAN": "UZB", "TURKMENISTAN": "TKM",
    "KYRGYZSTAN": "KGZ", "TAJIKISTAN": "TJK",
    "MOLDOVA": "MDA", "BELARUS": "BLR", "NORTH MACEDONIA": "MKD",
    "ALBANIA": "ALB", "BOSNIA AND HERZEGOVINA": "BIH", "SERBIA": "SRB",
    "MONTENEGRO": "MNE", "KOSOVO": "XKX",
    "PAPUA NEW GUINEA": "PNG", "FIJI": "FJI",
}

basel["iso3"] = basel["country_name"].str.strip().str.upper().map(BASEL_ISO3)
# For any remaining, try pycountry fallback
try:
    import pycountry
    def name_to_iso3(name):
        try:
            c = pycountry.countries.lookup(name.title())
            return c.alpha_3
        except:
            return None
    mask = basel["iso3"].isna()
    basel.loc[mask, "iso3"] = basel.loc[mask, "country_name"].apply(name_to_iso3)
except ImportError:
    pass

mapped   = basel["iso3"].notna().sum()
unmapped = basel["iso3"].isna().sum()
print(f"Basel ISO3 mapped: {mapped}, unmapped: {unmapped}")
if unmapped > 0:
    print("  Unmapped:", list(basel.loc[basel["iso3"].isna(), "country_name"]))

# ═══════════════════════════════════════════════════════════════════
# 2. World Bank WGI (4 indicators, 2022 values)
# ═══════════════════════════════════════════════════════════════════
def load_wb_indicator(folder_glob, col_name, year="2022"):
    csv_path = glob.glob(folder_glob)[0]
    df = pd.read_csv(csv_path, skiprows=4)
    df = df[["Country Code", year]].rename(
        columns={"Country Code": "iso3", year: col_name}
    )
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df.dropna(subset=["iso3"])

wgi_cc = load_wb_indicator(os.path.join(RAW, "wgi_CC.EST_raw", "API_*.csv"), "corruption_control")
wgi_rl = load_wb_indicator(os.path.join(RAW, "wgi_RL.EST_raw", "API_*.csv"), "rule_of_law")
wgi_ge = load_wb_indicator(os.path.join(RAW, "wgi_GE.EST_raw", "API_*.csv"), "govt_effectiveness")
wgi_rq = load_wb_indicator(os.path.join(RAW, "wgi_RQ.EST_raw", "API_*.csv"), "regulatory_quality")

wgi = wgi_cc.merge(wgi_rl, on="iso3").merge(wgi_ge, on="iso3").merge(wgi_rq, on="iso3")
print(f"WGI 2022: {len(wgi)} countries with all 4 indicators")

# ═══════════════════════════════════════════════════════════════════
# 3. TI CPI 2023
# ═══════════════════════════════════════════════════════════════════
try:
    import openpyxl
    cpi_raw = pd.read_excel(os.path.join(RAW, "cpi_2023.xlsx"), sheet_name="CPI 2023", header=3)
except ImportError:
    import subprocess
    subprocess.run(["pip3", "install", "openpyxl", "-q"], capture_output=True)
    import openpyxl
    cpi_raw = pd.read_excel(os.path.join(RAW, "cpi_2023.xlsx"), sheet_name="CPI 2023", header=3)

print("CPI raw columns:", list(cpi_raw.columns[:8]))
# Header row 3 gives: Country/Territory, ISO3, Region, CPI score 2023, ...
iso_col   = "ISO3"
score_col = "CPI score 2023"
cpi = cpi_raw[[iso_col, score_col]].rename(columns={iso_col: "iso3", score_col: "cpi_score"})
cpi["cpi_score"] = pd.to_numeric(cpi["cpi_score"], errors="coerce")
cpi = cpi.dropna(subset=["iso3", "cpi_score"])
print(f"CPI 2023: {len(cpi)} countries")

# ═══════════════════════════════════════════════════════════════════
# 4. World Bank Bank Branches per 100k adults (sourced from IMF FAS)
# ═══════════════════════════════════════════════════════════════════
branches = load_wb_indicator(
    os.path.join(RAW, "branches_raw", "API_*.csv"), "bank_branches_per100k"
)
# If 2022 has too many NAs, fall back to 2021
branches_2021 = load_wb_indicator(
    os.path.join(RAW, "branches_raw", "API_*.csv"), "bank_branches_2021", year="2021"
)
branches = branches.merge(branches_2021, on="iso3", how="left")
# Use 2022 value; fill with 2021 if 2022 is NA
branches["bank_branches_per100k"] = branches["bank_branches_per100k"].fillna(
    branches["bank_branches_2021"]
)
branches = branches[["iso3", "bank_branches_per100k"]]
print(f"Bank branches: {branches['bank_branches_per100k'].notna().sum()} countries with values")

# ═══════════════════════════════════════════════════════════════════
# 5. Mobile money / financial account ownership (% adults)
# ═══════════════════════════════════════════════════════════════════
# FX.OWN.TOTL.MA.ZS = Account at a financial institution (male, age 15+)
# Using FX.OWN.TOTL.ZS = All adults as proxy for financial inclusion
account = load_wb_indicator(
    os.path.join(RAW, "account_raw", "API_*.csv"), "financial_account_pct"
)
# Fill with 2021 values if 2022 missing (Global Findex waves)
try:
    acct_2021 = load_wb_indicator(
        os.path.join(RAW, "account_raw", "API_*.csv"), "acct_2021", year="2021"
    )
    account = account.merge(acct_2021, on="iso3", how="left")
    account["financial_account_pct"] = account["financial_account_pct"].fillna(account["acct_2021"])
    account = account[["iso3", "financial_account_pct"]]
except:
    pass
print(f"Financial account: {account['financial_account_pct'].notna().sum()} countries with values")

# ═══════════════════════════════════════════════════════════════════
# 6. GDP per capita (log)
# ═══════════════════════════════════════════════════════════════════
gdp = load_wb_indicator(
    os.path.join(RAW, "gdp_raw", "API_*.csv"), "gdp_per_capita"
)
gdp_2021 = load_wb_indicator(
    os.path.join(RAW, "gdp_raw", "API_*.csv"), "gdp_2021", year="2021"
)
gdp = gdp.merge(gdp_2021, on="iso3", how="left")
gdp["gdp_per_capita"] = gdp["gdp_per_capita"].fillna(gdp["gdp_2021"])
gdp = gdp[["iso3", "gdp_per_capita"]]
gdp["gdp_log"] = np.log(gdp["gdp_per_capita"].replace(0, np.nan))
print(f"GDP per capita: {gdp['gdp_per_capita'].notna().sum()} countries with values")

# ═══════════════════════════════════════════════════════════════════
# MERGE: join all datasets on ISO3
# ═══════════════════════════════════════════════════════════════════
# Start with Basel (outcome) and merge predictors
df = (
    basel[["iso3", "country_name", "basel_score"]]
    .merge(wgi,      on="iso3", how="left")
    .merge(cpi,      on="iso3", how="left")
    .merge(branches, on="iso3", how="left")
    .merge(account,  on="iso3", how="left")
    .merge(gdp[["iso3", "gdp_per_capita", "gdp_log"]], on="iso3", how="left")
)

print(f"\n=== MERGE RESULTS ===")
print(f"Total countries (from Basel): {len(df)}")
print(f"\nMissing values per variable:")
print(df.isnull().sum())

# Drop rows missing >20% of predictors (more than 1 of 8)
pred_cols = ["corruption_control","rule_of_law","govt_effectiveness",
             "regulatory_quality","cpi_score","bank_branches_per100k",
             "financial_account_pct","gdp_log"]
df["missing_pct"] = df[pred_cols].isna().mean(axis=1)
df_clean = df[df["missing_pct"] <= 0.20].copy()
print(f"\nAfter dropping >20% missing: {len(df_clean)} countries")

# Median imputation for remaining NAs
for col in pred_cols:
    median_val = df_clean[col].median()
    na_count   = df_clean[col].isna().sum()
    if na_count > 0:
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"  Imputed {na_count} values in {col} with median={median_val:.3f}")

# Drop rows still missing Basel score
df_clean = df_clean.dropna(subset=["basel_score"])
print(f"\nFinal analytical sample: n = {len(df_clean)}")

# Save
out_path = os.path.join(PROC, "merged_dataset_2023.csv")
df_clean.drop(columns=["missing_pct"]).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\nSample (first 5 rows):")
print(df_clean[["country_name","basel_score","corruption_control","cpi_score","gdp_log"]].head())
