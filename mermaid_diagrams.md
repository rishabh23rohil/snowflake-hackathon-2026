# Mermaid Diagrams — Public Health Trend Intelligence

Paste each diagram into https://mermaid.live to generate PNG images.

---

## 1. System Architecture

```mermaid
graph TB
    subgraph MARKETPLACE["Snowflake Marketplace (Free)"]
        DS1["ECDC Global<br/>Cases & Deaths<br/>180+ countries"]
        DS2["OWID Vaccinations<br/>Vaccination rates"]
        DS3["Google Mobility<br/>Retail, workplace,<br/>transit changes"]
        DS4["World Bank<br/>GDP, health spend,<br/>UHC, poverty"]
    end

    subgraph SQL["SQL Engine — 10+ Advanced Queries"]
        WIN["Window Functions<br/>AVG, LAG, SUM,<br/>ROW_NUMBER OVER"]
        CTE["CTEs &<br/>Subqueries"]
        JOIN["Multi-table JOINs<br/>+ ISO2→ISO3<br/>Country Mapping"]
    end

    subgraph AI["ML & AI Layer"]
        ML["Snowflake ML<br/>Forecasting API<br/>30-day projections"]
        CORTEX["Cortex AI<br/>COMPLETE()<br/>SUMMARIZE()<br/>mistral-large"]
        RISK["Risk Engine<br/>14-day trend +<br/>cases/100K →<br/>High / Med / Low"]
    end

    subgraph UI["Streamlit in Snowflake — 11 Tabs"]
        T0["How It Works"]
        T1["Country Rankings"]
        T2["Outbreak Trends"]
        T3["Vaccination"]
        T4["Mobility"]
        T5["Continental"]
        T6["ML Forecast"]
        T7["Risk Tiers"]
        T8["Socioeconomic"]
        T9["Data Quality"]
        T10["AI Insights"]
    end

    DS1 --> JOIN
    DS2 --> JOIN
    DS3 --> JOIN
    DS4 -->|ISO2→ISO3| JOIN
    JOIN --> WIN
    JOIN --> CTE
    WIN --> ML
    WIN --> RISK
    CTE --> CORTEX
    ML --> UI
    CORTEX --> UI
    RISK --> UI

    style MARKETPLACE fill:#0D2137,stroke:#58A6FF,color:#fff
    style SQL fill:#161B22,stroke:#30363D,color:#fff
    style AI fill:#1C2333,stroke:#7EE787,color:#fff
    style UI fill:#0D1117,stroke:#58A6FF,color:#fff
```

---

## 2. Data Pipeline Flow

```mermaid
flowchart LR
    subgraph Sources["4 Marketplace Datasets"]
        A["ECDC_GLOBAL<br/>42K+ rows"]
        B["OWID_VACCINATIONS<br/>169K rows"]
        C["GOOG_MOBILITY<br/>131K rows"]
        D["WORLD_BANK<br/>173 countries"]
    end

    subgraph Transform["SQL Transformations"]
        E["Window Functions<br/>7d/14d rolling avg<br/>LAG, SUM cumulative"]
        F["JOINs<br/>Country + Date<br/>Cross-DB ISO mapping"]
        G["CTEs<br/>Trend windows<br/>Reporting gaps"]
    end

    subgraph Output["ML + AI Output"]
        H["ML Forecast<br/>30-day per country"]
        I["Risk Classification<br/>High / Moderate / Low"]
        J["Cortex AI<br/>6 analysis types"]
    end

    subgraph Dashboard["Streamlit Dashboard"]
        K["11 Interactive Tabs<br/>Charts, Tables, AI"]
    end

    A --> F
    B --> F
    C --> F
    D -->|ISO2→ISO3| F
    F --> E
    E --> G
    G --> H
    G --> I
    G --> J
    H --> K
    I --> K
    J --> K

    style Sources fill:#0D2137,stroke:#58A6FF
    style Transform fill:#161B22,stroke:#30363D
    style Output fill:#1C2333,stroke:#7EE787
    style Dashboard fill:#0D1117,stroke:#58A6FF
```

---

## 3. Risk Classification Logic

```mermaid
flowchart TD
    A["Country Data"] --> B{"14-day trend<br/>% change?"}
    B -->|">25% increase<br/>OR cases/100K > 5000"| C["HIGH RISK"]
    B -->|"Any increase<br/>OR cases/100K > 2000"| D["MODERATE"]
    B -->|"Declining trend<br/>AND < 2000/100K"| E["LOW RISK"]
    B -->|"No data"| F["UNKNOWN"]

    C --> G["Dashboard:<br/>Red indicators"]
    D --> H["Dashboard:<br/>Yellow indicators"]
    E --> I["Dashboard:<br/>Green indicators"]

    style C fill:#EF4444,color:#fff
    style D fill:#F59E0B,color:#000
    style E fill:#10B981,color:#fff
    style F fill:#9CA3AF,color:#fff
```

---

## 4. Cross-Database JOIN Flow

```mermaid
flowchart LR
    subgraph DB1["COVID19_EPIDEMIOLOGICAL_DATA"]
        T1["ECDC_GLOBAL<br/>ISO3166_1 = 'US'<br/>(2-letter code)"]
    end

    subgraph MAPPING["SQL CASE Statement<br/>150+ country codes"]
        M["ISO2 → ISO3<br/>'US' → 'USA'<br/>'IN' → 'IND'<br/>'BR' → 'BRA'<br/>..."]
    end

    subgraph DB2["SNOWFLAKE_PUBLIC_DATA_FREE"]
        T2["WORLD_BANK_TIMESERIES<br/>GEO_ID = 'country/USA'<br/>(3-letter code)"]
    end

    subgraph RESULT["Merged Dataset"]
        R["91 countries matched<br/>COVID outcomes +<br/>GDP, health spend,<br/>UHC, life expectancy,<br/>poverty rate"]
    end

    T1 --> M
    M --> T2
    T1 --> RESULT
    T2 --> RESULT

    style DB1 fill:#0D2137,stroke:#58A6FF
    style DB2 fill:#0D2137,stroke:#58A6FF
    style MAPPING fill:#1C2333,stroke:#7EE787
    style RESULT fill:#161B22,stroke:#F59E0B
```

---

## 5. ML Forecasting Pipeline

```mermaid
flowchart TD
    A["Select Country"] --> B["Extract time-series<br/>CASES_SINCE_PREV_DAY<br/>by DATE"]
    B --> C["Write to temp table<br/>HACKATHON.DATA._forecast_input"]
    C --> D["CREATE SNOWFLAKE.ML.FORECAST<br/>Auto-trains model"]
    D --> E["CALL !FORECAST<br/>30 periods ahead"]
    D --> F["CALL !SHOW_EVALUATION_METRICS<br/>MAPE, RMSE"]
    E --> G["Altair chart:<br/>Blue = historical 90d<br/>Red dashed = forecast 30d"]
    F --> G
    G --> H["Metrics display:<br/>Avg forecast, Peak,<br/>Last historical"]

    style D fill:#7EE787,color:#000
    style E fill:#58A6FF,color:#fff
    style F fill:#F59E0B,color:#000
```
