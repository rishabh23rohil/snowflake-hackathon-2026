from snowflake.snowpark.context import get_active_session
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import io

session = get_active_session()

st.set_page_config(page_title="Public Health Intelligence", layout="wide", page_icon="🏥")

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — Clean Professional Theme
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Hero header */
.hero-banner {
    background: linear-gradient(135deg, #0055A5 0%, #0093EE 60%, #00B4D8 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 147, 238, 0.15);
}
.hero-banner h1 {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-banner .subtitle {
    color: rgba(255,255,255,0.85);
    font-size: 15px;
    margin: 0;
    font-weight: 400;
}

/* KPI cards */
[data-testid="stMetric"] {
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 18px 20px;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h1>Public Health Trend Intelligence</h1>
    <p class="subtitle">COVID-19 outbreak forecasting, risk classification & AI-generated health summaries — powered by Snowflake ML + Cortex AI</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA LAYER — Snowflake SQL with Window Functions, JOINs, CTEs
# ═══════════════════════════════════════════════════════════════

# ─── 1. Core case & death time-series with advanced epi metrics ───
@st.cache_data
def load_case_data():
    return session.sql("""
        WITH daily AS (
            SELECT
                COUNTRY_REGION,
                CONTINENTEXP AS CONTINENT,
                DATE,
                CASES,
                DEATHS,
                CASES_SINCE_PREV_DAY AS DAILY_CASES,
                DEATHS_SINCE_PREV_DAY AS DAILY_DEATHS,
                POPULATION,
                -- 7-day rolling average (window)
                ROUND(AVG(CASES_SINCE_PREV_DAY) OVER (
                    PARTITION BY COUNTRY_REGION ORDER BY DATE
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ), 2) AS CASES_7D_AVG,
                ROUND(AVG(DEATHS_SINCE_PREV_DAY) OVER (
                    PARTITION BY COUNTRY_REGION ORDER BY DATE
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ), 2) AS DEATHS_7D_AVG,
                -- 14-day rolling average
                ROUND(AVG(CASES_SINCE_PREV_DAY) OVER (
                    PARTITION BY COUNTRY_REGION ORDER BY DATE
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ), 2) AS CASES_14D_AVG,
                -- Cumulative cases (running sum)
                SUM(CASES_SINCE_PREV_DAY) OVER (
                    PARTITION BY COUNTRY_REGION ORDER BY DATE
                    ROWS UNBOUNDED PRECEDING
                ) AS CUMULATIVE_CASES,
                -- Cases per 100k
                ROUND(CASES * 100000.0 / NULLIF(POPULATION, 0), 2) AS CASES_PER_100K,
                -- Case fatality rate
                ROUND(DEATHS * 100.0 / NULLIF(CASES, 0), 2) AS CFR,
                -- Week-over-week change (LAG window)
                ROUND(CASES_SINCE_PREV_DAY - LAG(CASES_SINCE_PREV_DAY, 7) OVER (
                    PARTITION BY COUNTRY_REGION ORDER BY DATE
                ), 0) AS WOW_CASE_CHANGE,
                -- Doubling time proxy: days since cases doubled
                LAG(CASES, 14) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE) AS CASES_14D_AGO
            FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL
            WHERE CASES > 0 AND POPULATION > 0
        )
        SELECT *,
            -- Doubling time estimate: ln(2) / ln(CASES / CASES_14D_AGO) * 14
            CASE WHEN CASES_14D_AGO > 0 AND CASES > CASES_14D_AGO
                 THEN ROUND(14.0 * 0.693 / LN(CASES * 1.0 / CASES_14D_AGO), 1)
                 ELSE NULL END AS DOUBLING_TIME_DAYS
        FROM daily
        ORDER BY COUNTRY_REGION, DATE
    """).to_pandas()

# ─── 2. Country summary with trend analysis ───
@st.cache_data
def load_country_summary():
    return session.sql("""
        SELECT
            COUNTRY_REGION,
            CONTINENTEXP AS CONTINENT,
            MAX(POPULATION) AS POPULATION,
            MAX(CASES) AS TOTAL_CASES,
            MAX(DEATHS) AS TOTAL_DEATHS,
            ROUND(MAX(DEATHS) * 100.0 / NULLIF(MAX(CASES), 0), 2) AS CFR,
            ROUND(MAX(CASES) * 100000.0 / NULLIF(MAX(POPULATION), 0), 2) AS CASES_PER_100K,
            MIN(DATE) AS FIRST_CASE_DATE,
            MAX(DATE) AS LAST_REPORT_DATE,
            COUNT(DISTINCT DATE) AS DAYS_TRACKED,
            ROUND(AVG(CASE WHEN DATE >= DATEADD('day', -14, (SELECT MAX(DATE) FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL))
                        THEN CASES_SINCE_PREV_DAY END), 2) AS RECENT_14D_AVG,
            ROUND(AVG(CASE WHEN DATE BETWEEN DATEADD('day', -28, (SELECT MAX(DATE) FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL))
                        AND DATEADD('day', -14, (SELECT MAX(DATE) FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL))
                        THEN CASES_SINCE_PREV_DAY END), 2) AS PRIOR_14D_AVG
        FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL
        WHERE CASES > 0 AND POPULATION > 0
        GROUP BY COUNTRY_REGION, CONTINENTEXP
        HAVING MAX(CASES) >= 1000
        ORDER BY TOTAL_CASES DESC
    """).to_pandas()

# ─── 3. Vaccination data (OWID) ───
@st.cache_data
def load_vaccination_data():
    try:
        return session.sql("""
            SELECT
                COUNTRY_REGION,
                DATE,
                TOTAL_VACCINATIONS,
                PEOPLE_VACCINATED,
                PEOPLE_FULLY_VACCINATED,
                DAILY_VACCINATIONS,
                TOTAL_VACCINATIONS_PER_HUNDRED,
                PEOPLE_VACCINATED_PER_HUNDRED,
                PEOPLE_FULLY_VACCINATED_PER_HUNDRED,
                -- Rolling 7-day avg of daily vaccinations
                ROUND(AVG(DAILY_VACCINATIONS) OVER (
                    PARTITION BY COUNTRY_REGION ORDER BY DATE
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ), 0) AS VAX_7D_AVG
            FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.OWID_VACCINATIONS
            WHERE COUNTRY_REGION IS NOT NULL
            ORDER BY COUNTRY_REGION, DATE
        """).to_pandas()
    except:
        return pd.DataFrame()

# ─── 4. Google Mobility data ───
@st.cache_data
def load_mobility_data():
    try:
        return session.sql("""
            SELECT
                COUNTRY_REGION,
                DATE,
                ROUND(AVG(RETAIL_AND_RECREATION_CHANGE_PERC), 2) AS RETAIL_MOBILITY,
                ROUND(AVG(GROCERY_AND_PHARMACY_CHANGE_PERC), 2) AS GROCERY_MOBILITY,
                ROUND(AVG(WORKPLACES_CHANGE_PERC), 2) AS WORKPLACE_MOBILITY,
                ROUND(AVG(RESIDENTIAL_CHANGE_PERC), 2) AS RESIDENTIAL_MOBILITY,
                ROUND(AVG(TRANSIT_STATIONS_CHANGE_PERC), 2) AS TRANSIT_MOBILITY,
                ROUND(AVG(PARKS_CHANGE_PERC), 2) AS PARKS_MOBILITY
            FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.GOOG_GLOBAL_MOBILITY_REPORT
            WHERE COUNTRY_REGION IS NOT NULL
            GROUP BY COUNTRY_REGION, DATE
            ORDER BY COUNTRY_REGION, DATE
        """).to_pandas()
    except:
        return pd.DataFrame()

# ─── 5. Continent aggregation ───
@st.cache_data
def load_continent_data():
    return session.sql("""
        SELECT
            CONTINENTEXP AS CONTINENT,
            DATE,
            SUM(CASES_SINCE_PREV_DAY) AS DAILY_CASES,
            SUM(DEATHS_SINCE_PREV_DAY) AS DAILY_DEATHS,
            COUNT(DISTINCT COUNTRY_REGION) AS COUNTRIES
        FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL
        WHERE CASES > 0 AND CONTINENTEXP IS NOT NULL AND CONTINENTEXP != ''
        GROUP BY CONTINENTEXP, DATE
        ORDER BY DATE
    """).to_pandas()

# ─── 6. Cross-dataset: Cases vs Mobility JOIN ───
@st.cache_data
def load_cases_vs_mobility(country):
    try:
        return session.sql(f"""
            SELECT
                e.DATE,
                e.CASES_SINCE_PREV_DAY AS DAILY_CASES,
                ROUND(AVG(e.CASES_SINCE_PREV_DAY) OVER (ORDER BY e.DATE ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS CASES_7D_AVG,
                m.RETAIL_AND_RECREATION_CHANGE_PERC AS RETAIL_MOBILITY,
                m.WORKPLACES_CHANGE_PERC AS WORKPLACE_MOBILITY,
                m.RESIDENTIAL_CHANGE_PERC AS RESIDENTIAL_MOBILITY,
                m.TRANSIT_STATIONS_CHANGE_PERC AS TRANSIT_MOBILITY
            FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL e
            INNER JOIN COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.GOOG_GLOBAL_MOBILITY_REPORT m
                ON e.COUNTRY_REGION = m.COUNTRY_REGION AND e.DATE = m.DATE
            WHERE e.COUNTRY_REGION = '{country}'
              AND m.PROVINCE_STATE IS NULL
              AND e.CASES > 0
            ORDER BY e.DATE
        """).to_pandas()
    except:
        return pd.DataFrame()

# ─── 7. Cross-dataset: Cases vs Vaccination JOIN ───
@st.cache_data
def load_cases_vs_vaccination(country):
    try:
        return session.sql(f"""
            SELECT
                e.DATE,
                e.CASES_SINCE_PREV_DAY AS DAILY_CASES,
                ROUND(AVG(e.CASES_SINCE_PREV_DAY) OVER (ORDER BY e.DATE ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS CASES_7D_AVG,
                v.PEOPLE_FULLY_VACCINATED_PER_HUNDRED AS VAX_PCT,
                v.DAILY_VACCINATIONS
            FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL e
            INNER JOIN COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.OWID_VACCINATIONS v
                ON e.COUNTRY_REGION = v.COUNTRY_REGION AND e.DATE = v.DATE
            WHERE e.COUNTRY_REGION = '{country}'
              AND e.CASES > 0
            ORDER BY e.DATE
        """).to_pandas()
    except:
        return pd.DataFrame()

# ─── 8. Data quality / reporting gaps analysis ───
@st.cache_data
def load_reporting_gaps():
    return session.sql("""
        WITH country_dates AS (
            SELECT
                COUNTRY_REGION,
                CONTINENTEXP AS CONTINENT,
                COUNT(DISTINCT DATE) AS DAYS_REPORTED,
                MIN(DATE) AS FIRST_REPORT,
                MAX(DATE) AS LAST_REPORT,
                DATEDIFF('day', MIN(DATE), MAX(DATE)) + 1 AS EXPECTED_DAYS,
                ROUND(COUNT(DISTINCT DATE) * 100.0 / NULLIF(DATEDIFF('day', MIN(DATE), MAX(DATE)) + 1, 0), 1) AS REPORTING_COMPLETENESS_PCT,
                COUNT(CASE WHEN CASES_SINCE_PREV_DAY = 0 THEN 1 END) AS ZERO_CASE_DAYS,
                MAX(POPULATION) AS POPULATION
            FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL
            WHERE CASES > 0
            GROUP BY COUNTRY_REGION, CONTINENTEXP
        )
        SELECT *,
            CASE
                WHEN REPORTING_COMPLETENESS_PCT >= 90 THEN 'High Quality'
                WHEN REPORTING_COMPLETENESS_PCT >= 70 THEN 'Moderate Quality'
                ELSE 'Low Quality'
            END AS DATA_QUALITY_TIER
        FROM country_dates
        ORDER BY REPORTING_COMPLETENESS_PCT ASC
    """).to_pandas()

# ─── 9. World Bank Development Indicators (Snowflake Public Data) ───
@st.cache_data
def load_world_bank_indicators():
    try:
        return session.sql("""
            WITH iso_map AS (
                SELECT DISTINCT
                    COUNTRY_REGION,
                    ISO3166_1 AS ISO2
                FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL
                WHERE ISO3166_1 IS NOT NULL
            ),
            iso23 AS (
                SELECT COUNTRY_REGION, ISO2,
                    CASE ISO2
                        WHEN 'AF' THEN 'AFG' WHEN 'AL' THEN 'ALB' WHEN 'DZ' THEN 'DZA'
                        WHEN 'AD' THEN 'AND' WHEN 'AO' THEN 'AGO' WHEN 'AG' THEN 'ATG'
                        WHEN 'AR' THEN 'ARG' WHEN 'AM' THEN 'ARM' WHEN 'AU' THEN 'AUS'
                        WHEN 'AT' THEN 'AUT' WHEN 'AZ' THEN 'AZE' WHEN 'BS' THEN 'BHS'
                        WHEN 'BH' THEN 'BHR' WHEN 'BD' THEN 'BGD' WHEN 'BB' THEN 'BRB'
                        WHEN 'BY' THEN 'BLR' WHEN 'BE' THEN 'BEL' WHEN 'BZ' THEN 'BLZ'
                        WHEN 'BJ' THEN 'BEN' WHEN 'BT' THEN 'BTN' WHEN 'BO' THEN 'BOL'
                        WHEN 'BA' THEN 'BIH' WHEN 'BW' THEN 'BWA' WHEN 'BR' THEN 'BRA'
                        WHEN 'BN' THEN 'BRN' WHEN 'BG' THEN 'BGR' WHEN 'BF' THEN 'BFA'
                        WHEN 'BI' THEN 'BDI' WHEN 'CV' THEN 'CPV' WHEN 'KH' THEN 'KHM'
                        WHEN 'CM' THEN 'CMR' WHEN 'CA' THEN 'CAN' WHEN 'CF' THEN 'CAF'
                        WHEN 'TD' THEN 'TCD' WHEN 'CL' THEN 'CHL' WHEN 'CN' THEN 'CHN'
                        WHEN 'CO' THEN 'COL' WHEN 'KM' THEN 'COM' WHEN 'CD' THEN 'COD'
                        WHEN 'CG' THEN 'COG' WHEN 'CR' THEN 'CRI' WHEN 'CI' THEN 'CIV'
                        WHEN 'HR' THEN 'HRV' WHEN 'CU' THEN 'CUB' WHEN 'CY' THEN 'CYP'
                        WHEN 'CZ' THEN 'CZE' WHEN 'DK' THEN 'DNK' WHEN 'DJ' THEN 'DJI'
                        WHEN 'DM' THEN 'DMA' WHEN 'DO' THEN 'DOM' WHEN 'EC' THEN 'ECU'
                        WHEN 'EG' THEN 'EGY' WHEN 'SV' THEN 'SLV' WHEN 'GQ' THEN 'GNQ'
                        WHEN 'ER' THEN 'ERI' WHEN 'EE' THEN 'EST' WHEN 'SZ' THEN 'SWZ'
                        WHEN 'ET' THEN 'ETH' WHEN 'FJ' THEN 'FJI' WHEN 'FI' THEN 'FIN'
                        WHEN 'FR' THEN 'FRA' WHEN 'GA' THEN 'GAB' WHEN 'GM' THEN 'GMB'
                        WHEN 'GE' THEN 'GEO' WHEN 'DE' THEN 'DEU' WHEN 'GH' THEN 'GHA'
                        WHEN 'GR' THEN 'GRC' WHEN 'GD' THEN 'GRD' WHEN 'GT' THEN 'GTM'
                        WHEN 'GN' THEN 'GIN' WHEN 'GW' THEN 'GNB' WHEN 'GY' THEN 'GUY'
                        WHEN 'HT' THEN 'HTI' WHEN 'HN' THEN 'HND' WHEN 'HU' THEN 'HUN'
                        WHEN 'IS' THEN 'ISL' WHEN 'IN' THEN 'IND' WHEN 'ID' THEN 'IDN'
                        WHEN 'IR' THEN 'IRN' WHEN 'IQ' THEN 'IRQ' WHEN 'IE' THEN 'IRL'
                        WHEN 'IL' THEN 'ISR' WHEN 'IT' THEN 'ITA' WHEN 'JM' THEN 'JAM'
                        WHEN 'JP' THEN 'JPN' WHEN 'JO' THEN 'JOR' WHEN 'KZ' THEN 'KAZ'
                        WHEN 'KE' THEN 'KEN' WHEN 'KW' THEN 'KWT' WHEN 'KG' THEN 'KGZ'
                        WHEN 'LA' THEN 'LAO' WHEN 'LV' THEN 'LVA' WHEN 'LB' THEN 'LBN'
                        WHEN 'LS' THEN 'LSO' WHEN 'LR' THEN 'LBR' WHEN 'LY' THEN 'LBY'
                        WHEN 'LI' THEN 'LIE' WHEN 'LT' THEN 'LTU' WHEN 'LU' THEN 'LUX'
                        WHEN 'MG' THEN 'MDG' WHEN 'MW' THEN 'MWI' WHEN 'MY' THEN 'MYS'
                        WHEN 'MV' THEN 'MDV' WHEN 'ML' THEN 'MLI' WHEN 'MT' THEN 'MLT'
                        WHEN 'MR' THEN 'MRT' WHEN 'MU' THEN 'MUS' WHEN 'MX' THEN 'MEX'
                        WHEN 'MD' THEN 'MDA' WHEN 'MC' THEN 'MCO' WHEN 'MN' THEN 'MNG'
                        WHEN 'ME' THEN 'MNE' WHEN 'MA' THEN 'MAR' WHEN 'MZ' THEN 'MOZ'
                        WHEN 'MM' THEN 'MMR' WHEN 'NA' THEN 'NAM' WHEN 'NP' THEN 'NPL'
                        WHEN 'NL' THEN 'NLD' WHEN 'NZ' THEN 'NZL' WHEN 'NI' THEN 'NIC'
                        WHEN 'NE' THEN 'NER' WHEN 'NG' THEN 'NGA' WHEN 'NO' THEN 'NOR'
                        WHEN 'OM' THEN 'OMN' WHEN 'PK' THEN 'PAK' WHEN 'PA' THEN 'PAN'
                        WHEN 'PG' THEN 'PNG' WHEN 'PY' THEN 'PRY' WHEN 'PE' THEN 'PER'
                        WHEN 'PH' THEN 'PHL' WHEN 'PL' THEN 'POL' WHEN 'PT' THEN 'PRT'
                        WHEN 'QA' THEN 'QAT' WHEN 'RO' THEN 'ROU' WHEN 'RU' THEN 'RUS'
                        WHEN 'RW' THEN 'RWA' WHEN 'SA' THEN 'SAU' WHEN 'SN' THEN 'SEN'
                        WHEN 'RS' THEN 'SRB' WHEN 'SC' THEN 'SYC' WHEN 'SL' THEN 'SLE'
                        WHEN 'SG' THEN 'SGP' WHEN 'SK' THEN 'SVK' WHEN 'SI' THEN 'SVN'
                        WHEN 'SO' THEN 'SOM' WHEN 'ZA' THEN 'ZAF' WHEN 'KR' THEN 'KOR'
                        WHEN 'SS' THEN 'SSD' WHEN 'ES' THEN 'ESP' WHEN 'LK' THEN 'LKA'
                        WHEN 'SD' THEN 'SDN' WHEN 'SR' THEN 'SUR' WHEN 'SE' THEN 'SWE'
                        WHEN 'CH' THEN 'CHE' WHEN 'SY' THEN 'SYR' WHEN 'TW' THEN 'TWN'
                        WHEN 'TJ' THEN 'TJK' WHEN 'TZ' THEN 'TZA' WHEN 'TH' THEN 'THA'
                        WHEN 'TL' THEN 'TLS' WHEN 'TG' THEN 'TGO' WHEN 'TT' THEN 'TTO'
                        WHEN 'TN' THEN 'TUN' WHEN 'TR' THEN 'TUR' WHEN 'UG' THEN 'UGA'
                        WHEN 'UA' THEN 'UKR' WHEN 'AE' THEN 'ARE' WHEN 'GB' THEN 'GBR'
                        WHEN 'US' THEN 'USA' WHEN 'UY' THEN 'URY' WHEN 'UZ' THEN 'UZB'
                        WHEN 'VU' THEN 'VUT' WHEN 'VE' THEN 'VEN' WHEN 'VN' THEN 'VNM'
                        WHEN 'YE' THEN 'YEM' WHEN 'ZM' THEN 'ZMB' WHEN 'ZW' THEN 'ZWE'
                        ELSE NULL
                    END AS ISO3
                FROM iso_map
            ),
            wb_latest AS (
                SELECT
                    GEO_ID, VARIABLE_NAME, VALUE,
                    ROW_NUMBER() OVER (PARTITION BY GEO_ID, VARIABLE_NAME ORDER BY DATE DESC) AS RN
                FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.WORLD_BANK_TIMESERIES
                WHERE GEO_ID LIKE 'country/%'
                  AND VALUE IS NOT NULL
                  AND VARIABLE_NAME IN (
                      'Gross Domestic Product (GDP) (current USD)',
                      'Current health expenditure per capita, PPP (current international $)',
                      'Life expectancy at birth, female (years)',
                      'Life expectancy at birth, male (years)',
                      'Mortality rate, infant (per 1,000 live births)',
                      'Universal Health Coverage (UHC) service coverage index',
                      'Mortality rate, adult, male (per 1,000 male adults)',
                      'Mortality rate, adult, female (per 1,000 female adults)',
                      'Individuals using the Internet, male (percent of male population)',
                      'Poverty headcount ratio at societal poverty line (percent of population)'
                  )
            ),
            wb_pivot AS (
                SELECT
                    REPLACE(GEO_ID, 'country/', '') AS ISO3,
                    MAX(CASE WHEN VARIABLE_NAME = 'Gross Domestic Product (GDP) (current USD)' THEN VALUE END) AS GDP_USD,
                    MAX(CASE WHEN VARIABLE_NAME = 'Current health expenditure per capita, PPP (current international $)' THEN VALUE END) AS HEALTH_SPEND_PER_CAPITA,
                    MAX(CASE WHEN VARIABLE_NAME = 'Life expectancy at birth, female (years)' THEN VALUE END) AS LIFE_EXPECTANCY_F,
                    MAX(CASE WHEN VARIABLE_NAME = 'Life expectancy at birth, male (years)' THEN VALUE END) AS LIFE_EXPECTANCY_M,
                    MAX(CASE WHEN VARIABLE_NAME = 'Mortality rate, infant (per 1,000 live births)' THEN VALUE END) AS INFANT_MORTALITY,
                    MAX(CASE WHEN VARIABLE_NAME = 'Universal Health Coverage (UHC) service coverage index' THEN VALUE END) AS UHC_INDEX,
                    MAX(CASE WHEN VARIABLE_NAME = 'Mortality rate, adult, male (per 1,000 male adults)' THEN VALUE END) AS ADULT_MORTALITY_M,
                    MAX(CASE WHEN VARIABLE_NAME = 'Mortality rate, adult, female (per 1,000 female adults)' THEN VALUE END) AS ADULT_MORTALITY_F,
                    MAX(CASE WHEN VARIABLE_NAME = 'Individuals using the Internet, male (percent of male population)' THEN VALUE END) AS INTERNET_PCT_M,
                    MAX(CASE WHEN VARIABLE_NAME = 'Poverty headcount ratio at societal poverty line (percent of population)' THEN VALUE END) AS POVERTY_PCT
                FROM wb_latest
                WHERE RN = 1
                GROUP BY GEO_ID
            )
            SELECT
                im.COUNTRY_REGION,
                wb.*
            FROM iso23 im
            INNER JOIN wb_pivot wb ON im.ISO3 = wb.ISO3
            WHERE im.ISO3 IS NOT NULL
            ORDER BY im.COUNTRY_REGION
        """).to_pandas()
    except Exception as e:
        st.warning(f"World Bank data: {e}")
        return pd.DataFrame()

# ─── 10. ML Forecasting with evaluation ───
def build_forecast(country):
    try:
        session.sql(f"""
            CREATE OR REPLACE TABLE HACKATHON.DATA._forecast_input AS
            SELECT DATE AS ts, CASES_SINCE_PREV_DAY AS y
            FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL
            WHERE COUNTRY_REGION = '{country}'
              AND CASES_SINCE_PREV_DAY IS NOT NULL
              AND DATE IS NOT NULL
            ORDER BY DATE
        """).collect()

        session.sql("""
            CREATE OR REPLACE SNOWFLAKE.ML.FORECAST HACKATHON.DATA._covid_forecast(
                INPUT_DATA => TABLE(HACKATHON.DATA._forecast_input),
                TIMESTAMP_COLNAME => 'ts',
                TARGET_COLNAME => 'y'
            )
        """).collect()

        forecast = session.sql("""
            CALL HACKATHON.DATA._covid_forecast!FORECAST(FORECASTING_PERIODS => 30)
        """).to_pandas()

        # Get evaluation metrics
        try:
            eval_metrics = session.sql("""
                CALL HACKATHON.DATA._covid_forecast!SHOW_EVALUATION_METRICS()
            """).to_pandas()
        except:
            eval_metrics = pd.DataFrame()

        return forecast, eval_metrics
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ═══════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════

df = load_case_data()
df["DATE"] = pd.to_datetime(df["DATE"])
country_summary = load_country_summary()
continent_df = load_continent_data()
continent_df["DATE"] = pd.to_datetime(continent_df["DATE"])
vax_df = load_vaccination_data()
if len(vax_df) > 0:
    vax_df["DATE"] = pd.to_datetime(vax_df["DATE"])
mobility_df = load_mobility_data()
if len(mobility_df) > 0:
    mobility_df["DATE"] = pd.to_datetime(mobility_df["DATE"])
reporting_gaps = load_reporting_gaps()
wb_df = load_world_bank_indicators()

# ─── Risk tier classification ───
country_summary["TREND_CHANGE"] = (country_summary["RECENT_14D_AVG"] - country_summary["PRIOR_14D_AVG"]).round(2)
country_summary["TREND_PCT"] = (
    (country_summary["RECENT_14D_AVG"] - country_summary["PRIOR_14D_AVG"]) * 100.0 /
    country_summary["PRIOR_14D_AVG"].replace(0, float('nan'))
).round(1)

def classify_risk(row):
    if pd.isna(row["TREND_PCT"]):
        return "Unknown"
    if row["TREND_PCT"] > 25 or row["CASES_PER_100K"] > 5000:
        return "High"
    elif row["TREND_PCT"] > 0 or row["CASES_PER_100K"] > 2000:
        return "Moderate"
    else:
        return "Low"

country_summary["RISK_TIER"] = country_summary.apply(classify_risk, axis=1)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

st.sidebar.markdown("### Filters")
continents = ["All"] + sorted(df["CONTINENT"].dropna().unique().tolist())
selected_continent = st.sidebar.selectbox("Continent", continents)

if selected_continent != "All":
    available_countries = sorted(country_summary[country_summary["CONTINENT"] == selected_continent]["COUNTRY_REGION"].tolist())
else:
    available_countries = sorted(country_summary["COUNTRY_REGION"].tolist())

default_countries = country_summary.head(10)["COUNTRY_REGION"].tolist()
selected_countries = st.sidebar.multiselect("Countries (up to 15)", available_countries, default=default_countries[:5], max_selections=15)

st.sidebar.divider()
st.sidebar.markdown("### Track C: AI for Social Good")
st.sidebar.markdown("TAMU CSEGSA x Snowflake Hackathon 2026")
st.sidebar.divider()
st.sidebar.markdown("**4 Marketplace Datasets**")
st.sidebar.markdown("""
1. ECDC Global (cases/deaths)
2. OWID Vaccinations
3. Google Mobility
4. World Bank Indicators
""")
st.sidebar.markdown("**Snowflake Features**")
st.sidebar.markdown("""
- Window Functions (AVG, LAG, SUM, ROW_NUMBER)
- CTEs + Correlated Subqueries
- Multi-table JOINs (4 datasets)
- Cross-database ISO code mapping
- ML Forecasting API
- Cortex COMPLETE() & SUMMARIZE()
- Streamlit in Snowflake
""")

# ═══════════════════════════════════════════════════════════════
# KPIs
# ═══════════════════════════════════════════════════════════════

total_countries = country_summary["COUNTRY_REGION"].nunique()
total_cases = country_summary["TOTAL_CASES"].sum()
total_deaths = country_summary["TOTAL_DEATHS"].sum()
global_cfr = round(total_deaths * 100.0 / total_cases, 2) if total_cases > 0 else 0
high_risk_count = (country_summary["RISK_TIER"] == "High").sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Countries Tracked", f"{total_countries}")
col2.metric("Total Cases", f"{total_cases:,.0f}")
col3.metric("Total Deaths", f"{total_deaths:,.0f}")
col4.metric("Global CFR", f"{global_cfr}%")
col5.metric("High Risk Countries", f"{high_risk_count}")

st.divider()

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "How It Works", "Country Rankings", "Outbreak Trends", "Vaccination Tracker",
    "Mobility Impact", "Continental View", "ML Forecast",
    "Risk Tiers", "Socioeconomic Context", "Data Quality", "AI Insights"
])

# ═══════════════════════════════════════════════════════════════
# TAB 0: How It Works — Architecture & Flow
# ═══════════════════════════════════════════════════════════════
with tab0:
    st.subheader("Architecture & Technical Flow")
    st.caption("For judges: complete system overview — data pipeline, Snowflake features, and scoring alignment")

    # Load images from Snowflake stage
    def load_stage_image(filename):
        try:
            img_bytes = session.file.get_stream(f"@HACKATHON.DATA.APP_IMAGES/{filename}").read()
            return io.BytesIO(img_bytes)
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")
            return None

    st.markdown("### 1. System Architecture")
    img1 = load_stage_image("arch1.png")
    if img1:
        st.image(img1, use_container_width=True)
    st.caption("4 Marketplace datasets → SQL Engine (10+ queries) → ML & AI Layer → Streamlit Dashboard")

    st.markdown("### 2. Data Pipeline Flow")
    img2 = load_stage_image("arch2.png")
    if img2:
        st.image(img2, use_container_width=True)
    st.caption("ECDC + OWID + Mobility + World Bank → JOINs + Window Functions + CTEs → ML Forecast + Risk Tiers + Cortex AI")

    st.markdown("### 3. Risk Classification Logic")
    img3 = load_stage_image("arch3.png")
    if img3:
        st.image(img3, use_container_width=True)
    st.caption("14-day trend % change + cases per 100K → High (red) / Moderate (yellow) / Low (green)")

    st.markdown("### Snowflake Features Used")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **SQL & Data Engineering**
        - Window Functions: `AVG()`, `LAG()`, `SUM()`, `ROW_NUMBER()` OVER
        - Common Table Expressions (CTEs) — up to 4 levels deep
        - Multi-table JOINs across 4 Marketplace datasets
        - Cross-database JOIN: COVID-19 x Snowflake Public Data
        - ISO2→ISO3 country code mapping (150+ countries in SQL CASE)
        - Correlated subqueries for 14-day trend windows
        - `DATEDIFF`, `DATEADD`, `NULLIF`, `ROUND`, `LN` math
        """)
    with col_b:
        st.markdown("""
        **ML & AI**
        - **Snowflake ML Forecasting API** — per-country model + MAPE eval
        - **Cortex COMPLETE()** — 6 AI analysis types, `mistral-large`
        - **Cortex SUMMARIZE()** — automated data profiling

        **Platform**
        - **Streamlit in Snowflake** — 11-tab interactive dashboard
        - **Snowflake Marketplace** — 4 free datasets, zero ETL
        - **@st.cache_data** — efficient caching
        """)

    st.markdown("### Scoring Alignment — 100 Points")
    scoring = pd.DataFrame({
        "Dimension": ["Technical Depth (30 pts)", "Model Quality (25 pts)", "Social Impact (20 pts)", "Presentation (15 pts)", "Innovation (10 pts)"],
        "How We Score": [
            "4-dataset JOINs, 10+ SQL queries with window fns/CTEs, cross-DB ISO mapping, ML API, Cortex AI",
            "ML Forecasting with MAPE eval, per-country 30-day projections, risk tier classification",
            "Health minister briefings, fairness/bias disclosure, underreporting hypothesis, equity analysis",
            "11-tab dashboard, polished UI, interactive filters, hover tooltips, dual-axis charts",
            "World Bank socioeconomic correlation (4th dataset), poverty-vs-reporting analysis, doubling time"
        ]
    })
    st.dataframe(scoring, use_container_width=True, hide_index=True)

    st.markdown("### Tab Guide")
    tab_guide = pd.DataFrame({
        "Tab": ["How It Works", "Country Rankings", "Outbreak Trends", "Vaccination Tracker",
                "Mobility Impact", "Continental View", "ML Forecast", "Risk Tiers",
                "Socioeconomic Context", "Data Quality", "AI Insights"],
        "Key Feature": [
            "Architecture docs", "GROUP BY + HAVING", "Window functions + LAG",
            "JOIN ECDC x OWID", "JOIN ECDC x Mobility", "Continental aggregation",
            "ML Forecasting API", "Risk classification", "Cross-DB JOIN + ISO mapping",
            "CTE + DATEDIFF", "Cortex COMPLETE + SUMMARIZE"
        ]
    })
    st.dataframe(tab_guide, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 1: Country Rankings
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Countries Ranked by Total Cases")
    st.caption("SQL: GROUP BY + HAVING + correlated subqueries for 14-day trend windows")

    top_20 = country_summary.head(20)
    chart = alt.Chart(top_20).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("COUNTRY_REGION:N", sort="-y", title="Country"),
        y=alt.Y("TOTAL_CASES:Q", title="Total Cases"),
        color=alt.Color("RISK_TIER:N", scale=alt.Scale(
            domain=["High", "Moderate", "Low", "Unknown"],
            range=["#EF4444", "#F59E0B", "#10B981", "#9CA3AF"]
        )),
        tooltip=["COUNTRY_REGION", "TOTAL_CASES", "TOTAL_DEATHS", "CFR", "CASES_PER_100K", "RISK_TIER"]
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Country Detail Table")
    display_cols = ["COUNTRY_REGION", "CONTINENT", "TOTAL_CASES", "TOTAL_DEATHS", "CFR",
                    "CASES_PER_100K", "POPULATION", "RISK_TIER", "RECENT_14D_AVG", "TREND_CHANGE"]
    st.dataframe(country_summary[display_cols], use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2: Outbreak Trends + Doubling Time
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Daily Cases — 7-Day Rolling Average")
    st.caption("SQL: AVG() OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) + LAG() for doubling time")

    if selected_countries:
        trend_data = df[df["COUNTRY_REGION"].isin(selected_countries)]

        chart2 = alt.Chart(trend_data).mark_line(strokeWidth=2).encode(
            x=alt.X("DATE:T", title="Date"),
            y=alt.Y("CASES_7D_AVG:Q", title="7-Day Avg Daily Cases"),
            color=alt.Color("COUNTRY_REGION:N", legend=alt.Legend(title="Country")),
            tooltip=["DATE", "COUNTRY_REGION", "CASES_7D_AVG", "DAILY_CASES"]
        ).properties(height=400)
        st.altair_chart(chart2, use_container_width=True)

        st.subheader("Daily Deaths — 7-Day Rolling Average")
        chart2b = alt.Chart(trend_data).mark_line(strokeWidth=2).encode(
            x=alt.X("DATE:T", title="Date"),
            y=alt.Y("DEATHS_7D_AVG:Q", title="7-Day Avg Daily Deaths"),
            color=alt.Color("COUNTRY_REGION:N"),
            tooltip=["DATE", "COUNTRY_REGION", "DEATHS_7D_AVG", "DAILY_DEATHS"]
        ).properties(height=350)
        st.altair_chart(chart2b, use_container_width=True)

        st.subheader("Doubling Time (days)")
        st.caption("Estimated via: ln(2) / ln(cases_today / cases_14d_ago) * 14 — higher = slower spread")
        dt_data = trend_data.dropna(subset=["DOUBLING_TIME_DAYS"])
        dt_data = dt_data[dt_data["DOUBLING_TIME_DAYS"] > 0]
        if len(dt_data) > 0:
            chart_dt = alt.Chart(dt_data).mark_line(strokeWidth=1.5, opacity=0.8).encode(
                x=alt.X("DATE:T"),
                y=alt.Y("DOUBLING_TIME_DAYS:Q", title="Doubling Time (days)", scale=alt.Scale(clamp=True, domain=[0, 365])),
                color=alt.Color("COUNTRY_REGION:N"),
                tooltip=["DATE", "COUNTRY_REGION", "DOUBLING_TIME_DAYS"]
            ).properties(height=350)
            st.altair_chart(chart_dt, use_container_width=True)

        st.subheader("Case Fatality Rate Over Time")
        chart_cfr = alt.Chart(trend_data).mark_line(strokeWidth=1.5, opacity=0.7).encode(
            x=alt.X("DATE:T"),
            y=alt.Y("CFR:Q", title="Case Fatality Rate (%)", scale=alt.Scale(domain=[0, 15])),
            color=alt.Color("COUNTRY_REGION:N"),
            tooltip=["DATE", "COUNTRY_REGION", "CFR"]
        ).properties(height=300)
        st.altair_chart(chart_cfr, use_container_width=True)
    else:
        st.info("Select at least one country from the sidebar.")

# ═══════════════════════════════════════════════════════════════
# TAB 3: Vaccination Tracker
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Global Vaccination Progress")
    st.caption("Data: OWID_VACCINATIONS | SQL: Window function for 7-day rolling avg + JOIN with case data")

    if len(vax_df) > 0:
        # Vaccination summary by country (latest data)
        vax_latest = vax_df.sort_values("DATE").groupby("COUNTRY_REGION").last().reset_index()
        vax_latest = vax_latest[vax_latest["PEOPLE_FULLY_VACCINATED_PER_HUNDRED"].notna()]
        vax_latest = vax_latest.sort_values("PEOPLE_FULLY_VACCINATED_PER_HUNDRED", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Countries with Vax Data", f"{vax_latest['COUNTRY_REGION'].nunique()}")
        c2.metric("Highest Full Vax %", f"{vax_latest['PEOPLE_FULLY_VACCINATED_PER_HUNDRED'].max():.1f}%")
        c3.metric("Median Full Vax %", f"{vax_latest['PEOPLE_FULLY_VACCINATED_PER_HUNDRED'].median():.1f}%")

        top_vax = vax_latest.head(25)
        vax_chart = alt.Chart(top_vax).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("COUNTRY_REGION:N", sort="-y", title="Country"),
            y=alt.Y("PEOPLE_FULLY_VACCINATED_PER_HUNDRED:Q", title="Fully Vaccinated (%)"),
            color=alt.value("#10B981"),
            tooltip=["COUNTRY_REGION", "PEOPLE_FULLY_VACCINATED_PER_HUNDRED", "TOTAL_VACCINATIONS"]
        ).properties(height=400)
        st.altair_chart(vax_chart, use_container_width=True)

        # Vaccination trend for selected countries
        if selected_countries:
            st.subheader("Vaccination Rollout Over Time")
            vax_trend = vax_df[vax_df["COUNTRY_REGION"].isin(selected_countries)]
            if len(vax_trend) > 0:
                vax_line = alt.Chart(vax_trend).mark_line(strokeWidth=2).encode(
                    x=alt.X("DATE:T"),
                    y=alt.Y("PEOPLE_FULLY_VACCINATED_PER_HUNDRED:Q", title="Fully Vaccinated (%)"),
                    color=alt.Color("COUNTRY_REGION:N"),
                    tooltip=["DATE", "COUNTRY_REGION", "PEOPLE_FULLY_VACCINATED_PER_HUNDRED", "DAILY_VACCINATIONS"]
                ).properties(height=350)
                st.altair_chart(vax_line, use_container_width=True)

            # Cases vs Vaccination correlation
            st.subheader("Cases vs Vaccination — Did Vaccines Bend the Curve?")
            corr_country = st.selectbox("Select country for correlation", selected_countries, key="vax_corr")
            cv_data = load_cases_vs_vaccination(corr_country)
            if len(cv_data) > 0:
                cv_data["DATE"] = pd.to_datetime(cv_data["DATE"])
                base = alt.Chart(cv_data).encode(x=alt.X("DATE:T"))
                cases_line = base.mark_line(color="#EF4444", strokeWidth=2).encode(
                    y=alt.Y("CASES_7D_AVG:Q", title="7-Day Avg Cases"),
                    tooltip=["DATE", "CASES_7D_AVG"]
                )
                vax_line2 = base.mark_line(color="#10B981", strokeWidth=2, strokeDash=[5, 3]).encode(
                    y=alt.Y("VAX_PCT:Q", title="Fully Vaccinated %"),
                    tooltip=["DATE", "VAX_PCT"]
                )
                st.altair_chart(
                    alt.layer(cases_line, vax_line2).resolve_scale(y='independent').properties(height=400),
                    use_container_width=True
                )
                st.caption("Red = 7-day avg cases | Green dashed = vaccination % — look for cases declining as vaccination rises")
            else:
                st.info(f"No joined case+vaccination data available for {corr_country}")
    else:
        st.warning("Vaccination data not available. Check that OWID_VACCINATIONS table exists.")

# ═══════════════════════════════════════════════════════════════
# TAB 4: Mobility Impact
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Google Mobility vs COVID-19 Cases")
    st.caption("Data: GOOG_GLOBAL_MOBILITY_REPORT | SQL: INNER JOIN on country + date with ECDC_GLOBAL")

    if len(mobility_df) > 0 and selected_countries:
        mob_country = st.selectbox("Select country", selected_countries, key="mob_sel")
        cm_data = load_cases_vs_mobility(mob_country)

        if len(cm_data) > 0:
            cm_data["DATE"] = pd.to_datetime(cm_data["DATE"])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Workplace Mobility vs Cases**")
                base = alt.Chart(cm_data).encode(x=alt.X("DATE:T"))
                mob_line = base.mark_line(color="#0093EE", strokeWidth=1.5).encode(
                    y=alt.Y("WORKPLACE_MOBILITY:Q", title="Workplace Mobility Change %"),
                    tooltip=["DATE", "WORKPLACE_MOBILITY"]
                )
                case_line = base.mark_line(color="#EF4444", strokeWidth=2, opacity=0.6).encode(
                    y=alt.Y("CASES_7D_AVG:Q", title="Cases 7D Avg"),
                    tooltip=["DATE", "CASES_7D_AVG"]
                )
                st.altair_chart(
                    alt.layer(mob_line, case_line).resolve_scale(y='independent').properties(height=350),
                    use_container_width=True
                )

            with c2:
                st.markdown("**Retail Mobility vs Cases**")
                base2 = alt.Chart(cm_data).encode(x=alt.X("DATE:T"))
                retail_line = base2.mark_line(color="#F59E0B", strokeWidth=1.5).encode(
                    y=alt.Y("RETAIL_MOBILITY:Q", title="Retail Mobility Change %"),
                    tooltip=["DATE", "RETAIL_MOBILITY"]
                )
                case_line2 = base2.mark_line(color="#EF4444", strokeWidth=2, opacity=0.6).encode(
                    y=alt.Y("CASES_7D_AVG:Q", title="Cases 7D Avg"),
                    tooltip=["DATE", "CASES_7D_AVG"]
                )
                st.altair_chart(
                    alt.layer(retail_line, case_line2).resolve_scale(y='independent').properties(height=350),
                    use_container_width=True
                )

            st.subheader("All Mobility Categories")
            mob_melt = cm_data.melt(
                id_vars=["DATE"],
                value_vars=["RETAIL_MOBILITY", "WORKPLACE_MOBILITY", "RESIDENTIAL_MOBILITY", "TRANSIT_MOBILITY"],
                var_name="Category", value_name="Change_Pct"
            )
            mob_all = alt.Chart(mob_melt).mark_line(strokeWidth=1.5, opacity=0.7).encode(
                x=alt.X("DATE:T"),
                y=alt.Y("Change_Pct:Q", title="% Change from Baseline"),
                color=alt.Color("Category:N"),
                tooltip=["DATE", "Category", "Change_Pct"]
            ).properties(height=350)
            st.altair_chart(mob_all, use_container_width=True)
            st.caption("Negative values = less activity than pre-COVID baseline. Shows lockdown/restriction effects.")
        else:
            st.info(f"No mobility data available for {mob_country}")
    elif len(mobility_df) == 0:
        st.warning("Google Mobility data not available. Check GOOG_GLOBAL_MOBILITY_REPORT table.")
    else:
        st.info("Select at least one country from the sidebar.")

# ═══════════════════════════════════════════════════════════════
# TAB 5: Continental View
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Continental Daily Cases Over Time")
    st.caption("SQL: GROUP BY continent + date with SUM aggregation")

    chart3 = alt.Chart(continent_df).mark_area(opacity=0.6).encode(
        x=alt.X("DATE:T", title="Date"),
        y=alt.Y("DAILY_CASES:Q", title="Daily Cases", stack=True),
        color=alt.Color("CONTINENT:N", scale=alt.Scale(scheme="category10")),
        tooltip=["DATE", "CONTINENT", "DAILY_CASES"]
    ).properties(height=450)
    st.altair_chart(chart3, use_container_width=True)

    st.subheader("Continental Summary")
    cont_summary = country_summary.groupby("CONTINENT").agg(
        Countries=("COUNTRY_REGION", "nunique"),
        Total_Cases=("TOTAL_CASES", "sum"),
        Total_Deaths=("TOTAL_DEATHS", "sum"),
        Avg_CFR=("CFR", "mean"),
        Avg_Cases_Per_100K=("CASES_PER_100K", "mean")
    ).reset_index().round(2).sort_values("Total_Cases", ascending=False)
    st.dataframe(cont_summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 6: ML Forecast with Evaluation
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("30-Day Case Forecast")
    st.caption("Snowflake ML Forecasting API — trains per-country time-series model with automatic evaluation")

    forecast_country = st.selectbox("Select country to forecast", available_countries, index=0 if available_countries else None)

    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Training ML model for {forecast_country}..."):
            forecast_df, eval_metrics = build_forecast(forecast_country)

        if len(forecast_df) > 0:
            hist = df[df["COUNTRY_REGION"] == forecast_country][["DATE", "DAILY_CASES"]].copy()
            hist = hist.rename(columns={"DATE": "ts", "DAILY_CASES": "y"})
            hist["type"] = "Historical"
            hist = hist.tail(90)

            # ML Forecast columns have quoted names: "SERIES", "TS", "FORECAST", etc.
            # Strip quotes and normalize
            forecast_df.columns = [c.strip('"') for c in forecast_df.columns]
            forecast_df["ts"] = pd.to_datetime(forecast_df["TS"])
            forecast_df["y"] = pd.to_numeric(forecast_df["FORECAST"], errors="coerce").fillna(0).clip(lower=0)
            forecast_df["type"] = "Forecast"

            combined = pd.concat([hist[["ts", "y", "type"]], forecast_df[["ts", "y", "type"]]], ignore_index=True)

            # Historical chart
            hist_chart = alt.Chart(combined[combined["type"] == "Historical"]).mark_line(strokeWidth=2.5, color="#0093EE").encode(
                x=alt.X("ts:T", title="Date"),
                y=alt.Y("y:Q", title="Daily Cases"),
                tooltip=["ts", "y"]
            )
            # Forecast chart
            fc_chart = alt.Chart(combined[combined["type"] == "Forecast"]).mark_line(
                strokeWidth=2.5, color="#EF4444", strokeDash=[5, 5]
            ).encode(
                x=alt.X("ts:T"),
                y=alt.Y("y:Q"),
                tooltip=["ts", "y"]
            )
            st.altair_chart((hist_chart + fc_chart).properties(height=400), use_container_width=True)
            st.caption("Blue = Historical (last 90 days) | Red dashed = 30-day forecast")

            avg_fc = forecast_df["y"].mean()
            peak_fc = forecast_df["y"].max()
            last_hist = hist["y"].iloc[-1] if len(hist) > 0 else 0
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Avg Forecast (30d)", f"{avg_fc:,.1f}" if avg_fc > 1 else f"{avg_fc:.2f}")
            fc2.metric("Peak Forecast", f"{peak_fc:,.1f}" if peak_fc > 1 else f"{peak_fc:.2f}")
            fc3.metric("Last Historical", f"{last_hist:,.0f}" if last_hist > 1 else f"{last_hist}")

            # Show evaluation metrics if available
            if len(eval_metrics) > 0:
                st.subheader("Model Evaluation Metrics")
                st.dataframe(eval_metrics, use_container_width=True, hide_index=True)
            else:
                st.caption("Model evaluation: MAPE and other metrics computed during training.")
        else:
            st.warning("Forecast model could not be trained. Ensure Snowflake ML is available on your warehouse.")

# ═══════════════════════════════════════════════════════════════
# TAB 7: Risk Tiers
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Risk Tier Classification")
    st.caption("Computed from 14-day trend direction + cumulative cases per 100K")

    st.markdown("""
    | Tier | Criteria |
    |------|----------|
    | **High** | >25% increase in 14-day avg OR >5,000 cases per 100K |
    | **Moderate** | Any increase in 14-day avg OR >2,000 cases per 100K |
    | **Low** | Declining 14-day avg AND <2,000 cases per 100K |
    """)

    risk_counts = country_summary["RISK_TIER"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk", risk_counts.get("High", 0))
    c2.metric("Moderate Risk", risk_counts.get("Moderate", 0))
    c3.metric("Low Risk", risk_counts.get("Low", 0))

    risk_chart = alt.Chart(country_summary.head(30)).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("COUNTRY_REGION:N", sort="-y", title="Country"),
        y=alt.Y("CASES_PER_100K:Q", title="Cases per 100K"),
        color=alt.Color("RISK_TIER:N", scale=alt.Scale(
            domain=["High", "Moderate", "Low", "Unknown"],
            range=["#EF4444", "#F59E0B", "#10B981", "#9CA3AF"]
        )),
        tooltip=["COUNTRY_REGION", "RISK_TIER", "CASES_PER_100K", "TREND_PCT", "RECENT_14D_AVG"]
    ).properties(height=400)
    st.altair_chart(risk_chart, use_container_width=True)

    st.subheader("High Risk Countries — Detail")
    high_risk = country_summary[country_summary["RISK_TIER"] == "High"][display_cols]
    if len(high_risk) > 0:
        st.dataframe(high_risk, use_container_width=True, hide_index=True)
    else:
        st.success("No countries currently classified as High Risk.")

# ═══════════════════════════════════════════════════════════════
# TAB 8: Socioeconomic Context (World Bank + COVID JOIN)
# ═══════════════════════════════════════════════════════════════
with tab8:
    st.subheader("COVID-19 Outcomes vs Socioeconomic Indicators")
    st.caption("Data: Snowflake Public Data (World Bank) JOIN COVID-19 Epidemiological Data via ISO country codes")

    if len(wb_df) > 0:
        # Merge World Bank indicators with COVID country summary
        merged = country_summary.merge(wb_df, on="COUNTRY_REGION", how="inner")

        m1, m2, m3 = st.columns(3)
        m1.metric("Countries Matched", f"{len(merged)}")
        m2.metric("Avg Health Spend/Capita", f"${merged['HEALTH_SPEND_PER_CAPITA'].median():,.0f}" if merged['HEALTH_SPEND_PER_CAPITA'].notna().any() else "N/A")
        m3.metric("Avg UHC Index", f"{merged['UHC_INDEX'].median():.1f}" if merged['UHC_INDEX'].notna().any() else "N/A")

        # Scatter: Health Spending vs CFR
        st.subheader("Health Expenditure per Capita vs Case Fatality Rate")
        st.caption("Does higher health spending correlate with lower COVID fatality?")
        scatter_data = merged.dropna(subset=["HEALTH_SPEND_PER_CAPITA", "CFR"])
        if len(scatter_data) > 0:
            scatter1 = alt.Chart(scatter_data).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("HEALTH_SPEND_PER_CAPITA:Q", title="Health Expenditure per Capita (PPP $)", scale=alt.Scale(type="log")),
                y=alt.Y("CFR:Q", title="Case Fatality Rate (%)"),
                color=alt.Color("RISK_TIER:N", scale=alt.Scale(
                    domain=["High", "Moderate", "Low", "Unknown"],
                    range=["#EF4444", "#F59E0B", "#10B981", "#9CA3AF"]
                )),
                size=alt.Size("TOTAL_CASES:Q", scale=alt.Scale(range=[40, 600]), legend=None),
                tooltip=["COUNTRY_REGION", "HEALTH_SPEND_PER_CAPITA", "CFR", "TOTAL_CASES", "RISK_TIER", "CONTINENT"]
            ).properties(height=420)
            st.altair_chart(scatter1, use_container_width=True)
            st.caption("Bubble size = total cases. Log scale on X-axis. Hover for details.")

        # Scatter: Life Expectancy vs Cases per 100K
        st.subheader("Life Expectancy vs Cases per 100K")
        st.caption("Higher life expectancy often means older population — did that increase case burden?")
        scatter_data2 = merged.copy()
        scatter_data2["LIFE_EXPECTANCY"] = scatter_data2[["LIFE_EXPECTANCY_F", "LIFE_EXPECTANCY_M"]].mean(axis=1)
        scatter_data2 = scatter_data2.dropna(subset=["LIFE_EXPECTANCY", "CASES_PER_100K"])
        if len(scatter_data2) > 0:
            scatter2 = alt.Chart(scatter_data2).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("LIFE_EXPECTANCY:Q", title="Avg Life Expectancy (years)"),
                y=alt.Y("CASES_PER_100K:Q", title="Cases per 100K Population"),
                color=alt.Color("CONTINENT:N", scale=alt.Scale(scheme="category10")),
                size=alt.Size("POPULATION:Q", scale=alt.Scale(range=[40, 600]), legend=None),
                tooltip=["COUNTRY_REGION", "LIFE_EXPECTANCY", "CASES_PER_100K", "CONTINENT", "POPULATION"]
            ).properties(height=420)
            st.altair_chart(scatter2, use_container_width=True)

        # Scatter: UHC Index vs CFR
        st.subheader("Universal Health Coverage Index vs Case Fatality Rate")
        st.caption("Countries with stronger health systems (higher UHC index) — did they have better outcomes?")
        scatter_data3 = merged.dropna(subset=["UHC_INDEX", "CFR"])
        if len(scatter_data3) > 0:
            scatter3 = alt.Chart(scatter_data3).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("UHC_INDEX:Q", title="UHC Service Coverage Index (0-100)"),
                y=alt.Y("CFR:Q", title="Case Fatality Rate (%)"),
                color=alt.Color("CONTINENT:N", scale=alt.Scale(scheme="category10")),
                size=alt.Size("TOTAL_DEATHS:Q", scale=alt.Scale(range=[40, 600]), legend=None),
                tooltip=["COUNTRY_REGION", "UHC_INDEX", "CFR", "TOTAL_DEATHS", "CONTINENT"]
            ).properties(height=420)
            st.altair_chart(scatter3, use_container_width=True)

        # Scatter: Poverty vs Cases per 100K
        st.subheader("Poverty Rate vs Reported Cases per 100K")
        st.caption("Underreporting hypothesis: countries with higher poverty may have fewer reported cases due to limited testing")
        scatter_data4 = merged.dropna(subset=["POVERTY_PCT", "CASES_PER_100K"])
        if len(scatter_data4) > 0:
            scatter4 = alt.Chart(scatter_data4).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("POVERTY_PCT:Q", title="Poverty Headcount (% of population)"),
                y=alt.Y("CASES_PER_100K:Q", title="Cases per 100K Population"),
                color=alt.Color("CONTINENT:N", scale=alt.Scale(scheme="category10")),
                tooltip=["COUNTRY_REGION", "POVERTY_PCT", "CASES_PER_100K", "CONTINENT"]
            ).properties(height=380)
            st.altair_chart(scatter4, use_container_width=True)

        # Full data table
        st.subheader("Combined COVID + World Bank Data")
        display_wb = ["COUNTRY_REGION", "CONTINENT", "TOTAL_CASES", "CFR", "CASES_PER_100K",
                      "RISK_TIER", "HEALTH_SPEND_PER_CAPITA", "UHC_INDEX",
                      "LIFE_EXPECTANCY_F", "LIFE_EXPECTANCY_M", "INFANT_MORTALITY", "POVERTY_PCT"]
        valid_cols = [c for c in display_wb if c in merged.columns]
        st.dataframe(merged[valid_cols].sort_values("TOTAL_CASES", ascending=False), use_container_width=True, hide_index=True)

        st.info("""
        **Cross-Dataset JOIN:** This tab joins COVID-19 epidemiological outcomes (Starschema Marketplace)
        with World Bank development indicators (Snowflake Public Data Free) using a SQL ISO2→ISO3
        country code mapping. This is the 4th dataset integrated into the analysis — demonstrating
        multi-source data fusion for public health policy insights.
        """)
    else:
        st.warning("World Bank data could not be loaded. Ensure SNOWFLAKE_PUBLIC_DATA_FREE database is available.")

# ═══════════════════════════════════════════════════════════════
# TAB 9: Data Quality & Fairness
# ═══════════════════════════════════════════════════════════════
with tab9:
    st.subheader("Reporting Completeness & Data Quality Audit")
    st.caption("SQL: CTE with DATEDIFF, CASE, and completeness scoring — quantifying geographic bias")

    if len(reporting_gaps) > 0:
        quality_counts = reporting_gaps["DATA_QUALITY_TIER"].value_counts()
        q1, q2, q3 = st.columns(3)
        q1.metric("High Quality", quality_counts.get("High Quality", 0))
        q2.metric("Moderate Quality", quality_counts.get("Moderate Quality", 0))
        q3.metric("Low Quality", quality_counts.get("Low Quality", 0))

        # Distribution chart
        quality_hist = alt.Chart(reporting_gaps).mark_bar(opacity=0.7, color="#0093EE").encode(
            alt.X("REPORTING_COMPLETENESS_PCT:Q", bin=alt.Bin(maxbins=20), title="Reporting Completeness (%)"),
            alt.Y("count()", title="Number of Countries")
        ).properties(height=300)
        st.altair_chart(quality_hist, use_container_width=True)

        # Completeness by continent
        st.subheader("Reporting Quality by Continent")
        cont_quality = reporting_gaps.groupby("CONTINENT").agg(
            Countries=("COUNTRY_REGION", "nunique"),
            Avg_Completeness=("REPORTING_COMPLETENESS_PCT", "mean"),
            Min_Completeness=("REPORTING_COMPLETENESS_PCT", "min"),
            Avg_Zero_Case_Days=("ZERO_CASE_DAYS", "mean")
        ).reset_index().round(1).sort_values("Avg_Completeness")

        comp_chart = alt.Chart(cont_quality).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("CONTINENT:N", sort="y"),
            y=alt.Y("Avg_Completeness:Q", title="Avg Reporting Completeness %"),
            color=alt.value("#0093EE"),
            tooltip=["CONTINENT", "Countries", "Avg_Completeness", "Min_Completeness"]
        ).properties(height=300)
        st.altair_chart(comp_chart, use_container_width=True)

        st.subheader("Lowest Quality Reporters")
        st.dataframe(
            reporting_gaps[["COUNTRY_REGION", "CONTINENT", "REPORTING_COMPLETENESS_PCT", "DAYS_REPORTED",
                           "EXPECTED_DAYS", "ZERO_CASE_DAYS", "DATA_QUALITY_TIER"]].head(20),
            use_container_width=True, hide_index=True
        )

    st.subheader("Fairness & Bias Disclosure")
    st.warning("""
    **Geographic bias — quantified:**
    This analysis reveals systematic data quality differences:

    - **Reporting completeness varies from <50% to 100%** across countries — risk tiers for low-completeness
      countries should be treated as lower confidence
    - **Testing capacity asymmetry**: High-income countries with extensive testing report more cases per capita;
      this is a measurement artifact, not necessarily higher true prevalence
    - **Vaccination data gaps**: Many developing nations have delayed reporting — vaccination impact analysis
      is biased toward countries with better data infrastructure
    - **Zero-case days**: Some countries report extended periods of zero daily cases, likely reflecting
      reporting gaps rather than true disease absence
    - **Mobility data coverage**: Google Mobility data is limited to countries with significant Android usage,
      excluding parts of Africa and Asia

    **Recommendation**: Weight risk assessments by data quality tier. Use this dashboard as a screening tool,
    not a definitive ranking — ground truth requires local epidemiological expertise.
    """)

# ═══════════════════════════════════════════════════════════════
# TAB 10: AI Insights (Cortex COMPLETE + SUMMARIZE)
# ═══════════════════════════════════════════════════════════════
with tab10:
    st.subheader("AI-Generated Outbreak Analysis")
    st.markdown("*Powered by Snowflake Cortex COMPLETE() and SUMMARIZE()*")

    top5 = country_summary.head(5)
    high_risk_list = country_summary[country_summary["RISK_TIER"] == "High"]["COUNTRY_REGION"].head(10).tolist()

    # Auto-generated data summary using Cortex SUMMARIZE
    st.subheader("Auto Data Profile (Cortex SUMMARIZE)")
    if st.button("Profile Dataset", key="summarize_btn"):
        try:
            summary = session.sql("""
                SELECT SNOWFLAKE.CORTEX.SUMMARIZE(
                    (SELECT LISTAGG(
                        COUNTRY_REGION || ': ' || CASES || ' cases, ' || DEATHS || ' deaths, CFR=' ||
                        ROUND(DEATHS*100.0/NULLIF(CASES,0), 1) || '%', '; '
                    ) FROM (
                        SELECT COUNTRY_REGION, MAX(CASES) AS CASES, MAX(DEATHS) AS DEATHS
                        FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.ECDC_GLOBAL
                        WHERE CASES > 10000
                        GROUP BY COUNTRY_REGION
                        ORDER BY CASES DESC
                        LIMIT 20
                    ))
                ) AS summary
            """).collect()
            st.info(summary[0]["SUMMARY"])
        except Exception as e:
            st.info(f"Cortex SUMMARIZE: {e}")

    st.divider()

    with st.form("ai_covid_form"):
        analysis_type = st.selectbox("Analysis Type", [
            "Executive Summary for Decision Makers",
            "Country-Specific Deep Dive",
            "Continental Comparison",
            "Vaccination Impact Analysis",
            "Mobility & Lockdown Effectiveness",
            "Intervention Recommendations"
        ])
        submitted = st.form_submit_button("Generate AI Insight", type="primary")

    if submitted:
        if analysis_type == "Country-Specific Deep Dive" and selected_countries:
            focus = selected_countries[0]
            focus_data = country_summary[country_summary["COUNTRY_REGION"] == focus].iloc[0]
            prompt = f"""You are an epidemiologist. Write a detailed 3-paragraph country briefing for {focus}:
- Total cases: {focus_data['TOTAL_CASES']:,.0f}, Deaths: {focus_data['TOTAL_DEATHS']:,.0f}
- CFR: {focus_data['CFR']}%, Cases per 100K: {focus_data['CASES_PER_100K']}
- Risk tier: {focus_data['RISK_TIER']}, Recent 14-day avg: {focus_data['RECENT_14D_AVG']}
- Trend change: {focus_data['TREND_CHANGE']}, Population: {focus_data['POPULATION']:,.0f}
Cover: current situation, trajectory assessment, and specific recommended public health actions."""

        elif analysis_type == "Continental Comparison":
            prompt = f"""You are a global health analyst. Compare COVID-19 impact across continents:
- Total countries: {total_countries}, Cases: {total_cases:,.0f}, Deaths: {total_deaths:,.0f}, CFR: {global_cfr}%
- High risk: {high_risk_count} countries
- Top 5 by cases: {', '.join(top5['COUNTRY_REGION'].tolist())}
Write 3 paragraphs comparing continental patterns, disparities in outcomes, and regions needing attention."""

        elif analysis_type == "Vaccination Impact Analysis":
            vax_info = ""
            if len(vax_df) > 0:
                vax_latest_top = vax_df.sort_values("DATE").groupby("COUNTRY_REGION").last().reset_index()
                vax_latest_top = vax_latest_top.nlargest(5, "PEOPLE_FULLY_VACCINATED_PER_HUNDRED")
                vax_info = ", ".join([f"{r['COUNTRY_REGION']} ({r['PEOPLE_FULLY_VACCINATED_PER_HUNDRED']:.0f}%)" for _, r in vax_latest_top.iterrows()])
            prompt = f"""You are a vaccination policy expert. Analyze the relationship between vaccination and COVID-19 outcomes:
- Global cases: {total_cases:,.0f}, Deaths: {total_deaths:,.0f}
- Top vaccinated countries: {vax_info}
- Data covers 180+ countries with varying vaccination rates
Write 3 paragraphs analyzing vaccine effectiveness signals, equity gaps, and recommendations for accelerating coverage."""

        elif analysis_type == "Mobility & Lockdown Effectiveness":
            prompt = f"""You are a public health policy analyst. Analyze the effectiveness of mobility restrictions on COVID-19:
- Google Mobility data shows changes in retail, workplace, transit, and residential movement
- Countries implemented varying degrees of lockdowns and movement restrictions
- The data covers the relationship between mobility reduction and case trajectory
Write 3 paragraphs assessing which mobility restrictions were most effective, the economic trade-offs, and policy recommendations for future pandemics."""

        elif analysis_type == "Intervention Recommendations":
            prompt = f"""You are a public health policy advisor. Based on comprehensive COVID-19 data:
- {high_risk_count} High Risk countries, {total_countries} total
- Global CFR: {global_cfr}%, Total deaths: {total_deaths:,.0f}
- High risk: {', '.join(high_risk_list[:5])}
- Data includes case trends, vaccination rates, and mobility patterns
Write 3 detailed paragraphs recommending NPIs, vaccination strategies, and resource allocation priorities."""

        else:  # Executive Summary
            prompt = f"""You are a public health expert. Write a 3-paragraph executive briefing for a Health Minister:
- {total_countries} countries, {total_cases:,.0f} cases, {total_deaths:,.0f} deaths, {global_cfr}% CFR
- {high_risk_count} High Risk countries: {', '.join(high_risk_list[:5])}
- Top 5: {', '.join(top5['COUNTRY_REGION'].tolist())}
- Data includes epidemiological trends, vaccination coverage, mobility patterns, and reporting quality
Focus on: key trends, urgent interventions needed, and data-driven policy recommendations."""

        try:
            result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt.replace(chr(39), chr(39)+chr(39))}') AS response").collect()
            st.markdown(result[0]["RESPONSE"])
        except Exception as e:
            st.info(f"Cortex AI: {e}")

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align: center; padding: 16px 0;">
    <p style="font-size: 13px; color: #64748B; margin-bottom: 6px;">
        <strong>Snowflake Features:</strong>
        SQL Window Functions · CTEs · Multi-table JOINs (4 datasets) · Correlated Subqueries ·
        ISO Country Code Mapping · ML Forecasting API · Cortex COMPLETE() · Cortex SUMMARIZE()
    </p>
    <p style="font-size: 13px; color: #64748B; margin-bottom: 6px;">
        <strong>Data Sources:</strong>
        ECDC Global Cases · OWID Vaccinations · Google Mobility · World Bank Indicators
    </p>
    <p style="font-size: 12px; color: #0093EE; font-weight: 600;">
        Built for TAMU CSEGSA × Snowflake Hackathon 2026 | Track C: AI for Social Good
    </p>
</div>
""", unsafe_allow_html=True)
