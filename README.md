# ZGTM Freight Calculator

A comprehensive Python-based freight rate aggregation system for Australian e-commerce operations, specifically designed for bulky BBQ grill shipments.

## 🎯 Project Overview

This project aggregates freight rates from multiple carrier platforms to optimize logistics for our Australian BBQ e-commerce business. The system is designed to handle large, bulky products (grills) that typically ship on skids/pallets, with special consideration for residential vs commercial delivery requirements.

## 🏗️ Business Context

- **Product**: Barbecue grills sold in Australia
- **Shipping**: Large, bulky items typically shipped on skids/pallets
- **Delivery Types**: 
  - Residential delivery (HDS - Home Delivery Service)
  - Commercial delivery (Business)
- **Special Requirements**: Some carriers require tailgate services for home deliveries

## 🚛 Platform Integrations

Currently, we have **one active integration** with plans to expand to multiple freight platforms for comprehensive rate comparison.

### Active Integration

#### 🟢 NFM (National Freight Management) - Cario Platform
- **Status**: ✅ Complete and Production Ready
- **Platform**: Cario API
- **Account Structure**: Parent-child (master-slave) setup
- **Authentication**: Single token supports multiple customer IDs
- **Account Types**:
  - **HDS**: Home Delivery Service (residential + hand unload)
  - **Business**: Commercial delivery (no hand unload)
- **Coverage**: 8,775+ unique routes across Australia
- **Performance**: 50-100+ requests/second with 95-99% success rate

### Planned Integrations

#### 🟡 BigPost - Machship Wrapper
- **Status**: 🚧 Planned
- **Platform**: Custom stack built on Machship (wrapper)
- **Implementation**: TBD
- **Expected Features**: Similar HDS/Business mode support

#### 🟡 CFS - Direct Machship
- **Status**: 🚧 Planned  
- **Platform**: Direct Machship integration
- **Implementation**: TBD
- **Expected Features**: Standard Machship API integration

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas httpx tqdm rapidfuzz orjson uvloop
```

### Environment Setup

```bash
# Set your Cario session token (currently the only active integration)
export CARIO_SESSION_TOKEN="your_jwt_token_here"
```

### First-Time Setup (Cario/NFM Integration)

```bash
# 1. Fetch location data (one-time setup)
python cario_get_location.py

# 2. Test with a small sample
python cario_get_rates.py --test 5

# 3. Run full analysis
python cario_get_rates.py
```

> **Note**: Currently only Cario (NFM) integration is available. BigPost and CFS integrations are planned for future releases.

## 📋 Usage Examples

### Cario/NFM Integration Usage

```bash
# Test mode (3 routes)
python cario_get_rates.py --test

# Custom test count
python cario_get_rates.py --test 12

# Full analysis - HDS mode (default)
python cario_get_rates.py

# Full analysis - Business mode
python cario_get_rates.py --mode bus

# Run both modes for comparison
python cario_get_rates.py --mode both

# Post-process existing data
python cario_get_rates.py --post-process
```

> **Note**: All current usage examples are for the Cario (NFM) integration. Future integrations will have their own command-line interfaces.

### Mode Differences

| Mode | isResidential | optionHandUnload | Use Case |
|------|---------------|------------------|----------|
| **HDS** | `true` | `true` | Home delivery with hand unload |
| **Business** | `false` | `false` | Commercial delivery, no hand unload |

## 📁 Project Structure

```
zgtm-freight-calculator/
├── cario_get_rates.py          # NFM/Cario integration script (ACTIVE)
├── cario_get_location.py       # Cario location data scraper
├── cario_analysis.ipynb        # Cario analysis notebook
├── function.py                 # Utility functions
├── input/                      # Input data files
│   ├── australian_postcodes_2025.csv
│   ├── parameters.xlsx
│   └── ...
├── data/                       # Cached location data
│   └── cario_location_data.csv
├── output/                     # Generated results (excluded from git)
│   ├── cario_hds_raw_data_*.csv
│   ├── cario_bus_raw_data_*.csv
│   ├── cario_hds_cheapest_per_*.csv
│   ├── cario_bus_cheapest_per_*.csv
│   └── logs/
├── unified_tracking/           # Tracking integration
├── bigpost_get_rates.py        # BigPost integration (PLANNED)
├── cfs_get_rates.py            # CFS integration (PLANNED)
└── README.md                   # This file
```

### Current Status
- **Active**: Cario/NFM integration only
- **Planned**: BigPost and CFS integrations will follow similar structure

## 📊 Output Files

### Cario/NFM Integration Output

#### Raw Data (Preserved for Re-processing)
- `cario_hds_raw_data_YYYYMMDD-HHMMSS.csv` - Complete API responses (HDS mode)
- `cario_bus_raw_data_YYYYMMDD-HHMMSS.csv` - Complete API responses (Business mode)

#### Processed Data
- `cario_hds_cheapest_per_lane_YYYYMMDD-HHMMSS.csv` - Cheapest rates per route (HDS)
- `cario_hds_cheapest_per_postcode_YYYYMMDD-HHMMSS.csv` - Cheapest rates per postcode (HDS)
- `cario_bus_cheapest_per_lane_YYYYMMDD-HHMMSS.csv` - Cheapest rates per route (Business)
- `cario_bus_cheapest_per_postcode_YYYYMMDD-HHMMSS.csv` - Cheapest rates per postcode (Business)

#### Logs
- `output/logs/cario_api_responses_YYYYMMDD_HHMMSS.log` - Detailed API logs

### Future Platform Outputs
- **BigPost**: `bigpost_*_data_*.csv` files (planned)
- **CFS**: `cfs_*_data_*.csv` files (planned)

## ⚡ Performance Characteristics

### Cario/NFM Integration Performance

| Dataset Size | Routes | Estimated Time | Success Rate |
|--------------|--------|----------------|--------------|
| Test Mode | 3-12 | 20-45 seconds | 95-99% |
| Small | 1,000 | 2-5 minutes | 95-99% |
| Medium | 10,000 | 15-30 minutes | 95-99% |
| Full Dataset | 8,775+ | 45-90 minutes | 95-99% |

### Throughput
- **Requests/second**: 50-100+ (vs 2-5 in sequential systems)
- **Concurrent requests**: 96+ (32 per account × 3 accounts)
- **Multi-account support**: Uses multiple Cario accounts simultaneously

> **Note**: Performance metrics are for the current Cario integration. Future integrations (BigPost, CFS) will have their own performance characteristics.

## 🔧 Configuration

### Cario/NFM Integration Configuration

#### Multi-Account Setup

```python
CREDENTIALS = {
    "MEL": {"customer_id": 16071, "auth_token": SESSION_TOKEN, "tenant_id": ""},
    "SYD": {"customer_id": 16070, "auth_token": SESSION_TOKEN, "tenant_id": ""},
    "BNE": {"customer_id": 16072, "auth_token": SESSION_TOKEN, "tenant_id": ""},
}

WH_TO_ACCOUNT = {
    "Epping": "SYD",
    "Chipping Norton": "SYD", 
    "Berrinba": "BNE",
}
```

#### Performance Tuning

```python
CONCURRENT_REQUESTS_PER_ACCOUNT = 32  # Concurrent requests per account
MAX_RETRIES = 8                       # Maximum retry attempts
SLEEP_BASE = 0.5                      # Base delay for exponential backoff
REQUEST_TIMEOUT = 60.0                # Request timeout in seconds
```

### Future Platform Configurations
- **BigPost**: Will have its own Machship-based configuration
- **CFS**: Will have its own direct Machship configuration

## 🛠️ Development

### Key Features

- **High-Performance Async Architecture**: Multi-account support with HTTP/2
- **Comprehensive Logging**: Detailed API response logging
- **Error Recovery**: Advanced retry logic with exponential backoff
- **Raw Data Preservation**: Complete API responses saved for re-processing
- **Notebook-Style Processing**: Exact replication of Jupyter notebook processing

### Adding New Carriers

The project is designed to be extensible. To add a new carrier:

1. Create a new module following the Cario pattern
2. Implement the core functions:
   - `prepare_lanes_df()` - Data preparation
   - `get_rates_for_routes()` - Rate fetching
   - `process_results()` - Data processing
3. Add configuration for the new carrier
4. Update this README

## 🐛 Troubleshooting

### Common Issues

**"SESSION_TOKEN not configured"**
```bash
export CARIO_SESSION_TOKEN="your_token_here"
```

**"Location cache not found"**
```bash
python cario_get_location.py
```

**Low success rate**
- Verify SESSION_TOKEN is valid and not expired
- Check customer IDs in CREDENTIALS
- Reduce CONCURRENT_REQUESTS_PER_ACCOUNT if getting rate limited

### Performance Issues

**Too slow?**
- Increase CONCURRENT_REQUESTS_PER_ACCOUNT (watch for rate limits)
- Add more accounts to CREDENTIALS
- Ensure uvloop and orjson are installed

**Too many errors?**
- Reduce CONCURRENT_REQUESTS_PER_ACCOUNT
- Increase SLEEP_BASE for longer delays
- Check server-side rate limits

## 📈 Data Analysis

### Loading Results

```python
import pandas as pd

# Load processed data
hds_lanes = pd.read_csv('output/cario_hds_cheapest_per_lane_*.csv')
bus_lanes = pd.read_csv('output/cario_bus_cheapest_per_lane_*.csv')

# Compare modes
print(f"HDS average price: ${hds_lanes['freight_price'].mean():.2f}")
print(f"Business average price: ${bus_lanes['freight_price'].mean():.2f}")
```

### Key Metrics

- **Average freight price** by mode and route
- **Transit time analysis** across carriers
- **Carrier performance** comparison
- **Geographic pricing** patterns

## 🎯 Project Goals

1. **Aggregate freight rates** across multiple platforms
2. **Provide accurate comparisons** for bulky freight (skids/pallets)
3. **Feed results** into freight analysis systems
4. **Optimize logistics** for BBQ e-commerce website
5. **Support both residential and commercial** delivery scenarios

## 📝 Version Control

### Git Setup

```bash
# Initialize repository
git init
git remote add origin https://github.com/your-org/zgtm-freight-calculator.git

# Exclude output files (too large for git)
echo "output/" >> .gitignore
echo "*.log" >> .gitignore
echo "__pycache__/" >> .gitignore
```

### Excluded Files

- `/output/` - Generated results and logs (too large for version control)
- `__pycache__/` - Python cache files
- `*.log` - Log files

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive logging for new features
3. Test with small datasets before full runs
4. Update documentation for new carriers or features
5. Ensure backward compatibility with existing data formats

## 📞 Support

For issues or questions:
1. Check `output/logs/` for detailed error information
2. Verify all dependencies are installed correctly
3. Ensure location cache is built and up to date
4. Test with `--test` flag first before running full analysis
5. Check SESSION_TOKEN is valid and properly set

---

**Built for Z Grills Australia** - Optimizing freight logistics for BBQ e-commerce 🍖
