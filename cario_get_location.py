#!/usr/bin/env python3
"""
Cario Location Data Scraper

This script fetches location data from the Cario API for all Australian postcodes
and caches the results in a CSV file for fast lookup. Uses async/httpx for 
high-performance concurrent requests.

Usage:
    python fetch_location_data.py

The script will:
1. Load Australian postcodes from input/australian_postcodes (1).csv
2. Fetch location data from Cario API for each unique postcode
3. Cache results in data/cario_location_data.csv
4. Skip postcodes that already exist in the cache (resume capability)
"""

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx
import pandas as pd
from tqdm.asyncio import tqdm

# Cario API Configuration
BASE_URL = "https://integrate.cario.com.au/api/"
CUSTOMER_ID = 7115
COUNTRY_ID = 36
TENANT_ID = ""

# Authentication token for Cario API
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJGTVNUb2tlbiI6ImV5SmhiR2NpT2lKSVV6STFOaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpvZEhSd09pOHZjMk5vWlcxaGN5NTRiV3h6YjJGd0xtOXlaeTkzY3k4eU1EQTFMekExTDJsa1pXNTBhWFI1TDJOc1lXbHRjeTl1WVcxbGFXUmxiblJwWm1sbGNpSTZJakV6TWpVd0lpd2lhSFIwY0RvdkwzTmphR1Z0WVhNdWVHMXNjMjloY0M1dmNtY3ZkM012TWpBd05TOHdOUzlwWkdWdWRHbDBlUzlqYkdGcGJYTXZibUZ0WlNJNklrTjFjM1J2YlVCcGJuUmxaM0poZEdVM01URTFJaXdpYUhSMGNEb3ZMM05qYUdWdFlYTXVlRzFzYzI5aGNDNXZjbWN2ZDNNdk1qQXdOUzh3TlM5cFpHVnVkR2wwZVM5amJHRnBiWE12WlcxaGFXeGhaR1J5WlhOeklqb2lRM1Z6ZEc5dFFHbHVkR1ZuY21GMFpUY3hNVFV1YVdRaUxDSkJjM0JPWlhRdVNXUmxiblJwZEhrdVUyVmpkWEpwZEhsVGRHRnRjQ0k2SWtoSlNUSk5TakpYUTFCTlJrNUNObFZIUVROSVNWSTBRazlhTmpOSU5rZElJaXdpYUhSMGNEb3ZMM2QzZHk1aGMzQnVaWFJpYjJsc1pYSndiR0YwWlM1amIyMHZhV1JsYm5ScGRIa3ZZMnhoYVcxekwzUmxibUZ1ZEVsa0lqb2lPU0lzSWtOMWMzUnZiV1Z5U1VRaU9pSTNNVEUxSWl3aVFYQndiR2xqWVhScGIyNU9ZVzFsSWpvaVEzVnpkRzl0SWl3aWMzVmlJam9pTVRNeU5UQWlMQ0pxZEdraU9pSTBPRE15T1dReU1pMDFORGhpTFRReE5UTXRPVGt3WlMwd01EY3dOVE15TnpnNVpUa2lMQ0pwWVhRaU9qRTNNamszTlRNd05UZ3NJbWx6Y3lJNklrWk5VeUlzSW1GMVpDSTZJa1pOVXlKOS5LSEFoaUpRMDU5Qk5UNWhMYnNNNTVoZF94RjdWbmw4UWJfSDM1bEFuQ09nIiwibmJmIjoxNzI5NzUzMDU4LCJleHAiOjE3NjEyODkwNTgsImlzcyI6ImludGVncmF0ZS5jYXJpby5jb20uYXUiLCJhdWQiOiJpbnRlZ3JhdGVkY2FyaW9jdXN0b21lcnMifQ.E-KuuetkhAWtF-dm-1w133n82kq93svCXnhXKezQYio"

# HTTP headers for API requests
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "CustomerId": str(CUSTOMER_ID),
    "TenantId": TENANT_ID,
    "Content-Type": "application/json"
}

# Configuration
MAX_CONCURRENT_REQUESTS = 10  # Limit concurrent requests to avoid overwhelming the API
REQUEST_TIMEOUT = 30  # Timeout for individual requests in seconds
MAX_RETRIES = 3  # Maximum retry attempts for failed requests
RETRY_DELAY = 1.0  # Base delay between retries (exponential backoff)

# File paths
INPUT_FILE = Path("input/australian_postcodes (1).csv")
OUTPUT_FILE = Path("data/cario_location_data.csv")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('location_scraper.log')
    ]
)
logger = logging.getLogger(__name__)


class LocationScraper:
    """High-performance location data scraper for Cario API."""
    
    def __init__(self):
        self.session: Optional[httpx.AsyncClient] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            headers=HEADERS,
            limits=httpx.Limits(max_connections=MAX_CONCURRENT_REQUESTS)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    async def fetch_location_data(self, postcode: str) -> Dict:
        """
        Fetch location data for a single postcode from Cario API.
        
        Args:
            postcode: 4-digit postcode string
            
        Returns:
            Dictionary containing postcode and API response data
        """
        # Ensure postcode is 4 digits with leading zeros
        postcode_padded = str(postcode).zfill(4)
        url = f"{BASE_URL}Location/FindByCountryId/{COUNTRY_ID}/{postcode_padded}"
        
        async with self.semaphore:  # Limit concurrent requests
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await self.session.get(url)
                    
                    if response.status_code == 200:
                        locations = response.json()
                        logger.debug(f"Successfully fetched {len(locations)} locations for postcode {postcode_padded}")
                        return {
                            'Postcode': postcode_padded,
                            'Response': locations
                        }
                    elif response.status_code == 404:
                        # No locations found for this postcode - this is normal
                        logger.debug(f"No locations found for postcode {postcode_padded}")
                        return {
                            'Postcode': postcode_padded,
                            'Response': []
                        }
                    else:
                        logger.warning(f"HTTP {response.status_code} for postcode {postcode_padded}, attempt {attempt + 1}")
                        
                except httpx.TimeoutException:
                    logger.warning(f"Timeout for postcode {postcode_padded}, attempt {attempt + 1}")
                except httpx.RequestError as e:
                    logger.warning(f"Request error for postcode {postcode_padded}, attempt {attempt + 1}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error for postcode {postcode_padded}, attempt {attempt + 1}: {e}")
                
                # Exponential backoff for retries
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
            
            # All retries failed
            logger.error(f"Failed to fetch data for postcode {postcode_padded} after {MAX_RETRIES + 1} attempts")
            return {
                'Postcode': postcode_padded,
                'Response': None,
                'Error': f'Failed after {MAX_RETRIES + 1} attempts'
            }


def load_existing_postcodes(output_file: Path) -> Set[str]:
    """
    Load postcodes that already exist in the output file.
    
    Args:
        output_file: Path to the CSV output file
        
    Returns:
        Set of postcodes that already have data
    """
    existing_postcodes = set()
    
    if output_file.exists():
        try:
            df = pd.read_csv(output_file)
            existing_postcodes = set(df['Postcode'].astype(str).str.zfill(4))
            logger.info(f"Found {len(existing_postcodes)} existing postcodes in cache")
        except Exception as e:
            logger.warning(f"Error reading existing cache file: {e}")
    
    return existing_postcodes


def load_postcodes_to_fetch(input_file: Path, existing_postcodes: Set[str]) -> List[str]:
    """
    Load unique postcodes from input file, excluding those already cached.
    
    Args:
        input_file: Path to the input CSV file with Australian postcodes
        existing_postcodes: Set of postcodes already in cache
        
    Returns:
        List of unique postcodes to fetch
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load postcodes from input file
    df = pd.read_csv(input_file)
    
    # Get unique postcodes, ensuring they have chargezone data (valid postcodes)
    valid_postcodes = df[df['chargezone'].notna()]['postcode'].unique()
    
    # Convert to 4-digit strings and filter out existing ones
    postcodes_to_fetch = []
    for pc in valid_postcodes:
        pc_str = str(pc).zfill(4)
        if pc_str not in existing_postcodes:
            postcodes_to_fetch.append(pc_str)
    
    logger.info(f"Found {len(valid_postcodes)} unique postcodes in input file")
    logger.info(f"Need to fetch data for {len(postcodes_to_fetch)} postcodes")
    
    return sorted(postcodes_to_fetch)


def save_results_to_csv(results: List[Dict], output_file: Path, append: bool = True):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
        append: Whether to append to existing file or overwrite
    """
    if not results:
        logger.warning("No results to save")
        return
    
    # Prepare data for CSV
    csv_data = []
    for result in results:
        csv_data.append({
            'Postcode': result['Postcode'],
            'Response': json.dumps(result['Response']) if result['Response'] is not None else '',
            'Error': result.get('Error', '')
        })
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Save to CSV
    if append and output_file.exists():
        # Append to existing file
        df.to_csv(output_file, mode='a', header=False, index=False)
        logger.info(f"Appended {len(df)} records to {output_file}")
    else:
        # Create new file or overwrite
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")


async def main():
    """Main function to orchestrate the location data scraping."""
    logger.info("Starting Cario location data scraper")
    
    try:
        # Ensure output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing postcodes to avoid re-fetching
        existing_postcodes = load_existing_postcodes(OUTPUT_FILE)
        
        # Load postcodes that need to be fetched
        postcodes_to_fetch = load_postcodes_to_fetch(INPUT_FILE, existing_postcodes)
        
        if not postcodes_to_fetch:
            logger.info("All postcodes already cached. Nothing to fetch.")
            return
        
        logger.info(f"Starting to fetch location data for {len(postcodes_to_fetch)} postcodes")
        
        # Create scraper and fetch data
        async with LocationScraper() as scraper:
            # Create tasks for all postcodes
            tasks = [scraper.fetch_location_data(pc) for pc in postcodes_to_fetch]
            
            # Execute tasks with progress bar
            results = []
            batch_size = 100  # Save results in batches to avoid memory issues
            
            for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter out exceptions and add valid results
                valid_results = []
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                    else:
                        valid_results.append(result)
                
                # Save batch results
                if valid_results:
                    save_results_to_csv(valid_results, OUTPUT_FILE, append=True)
                    results.extend(valid_results)
        
        # Summary
        successful = len([r for r in results if r.get('Response') is not None])
        failed = len(results) - successful
        
        logger.info(f"Scraping completed!")
        logger.info(f"Successfully fetched: {successful} postcodes")
        logger.info(f"Failed: {failed} postcodes")
        logger.info(f"Results saved to: {OUTPUT_FILE}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check dependencies
    try:
        import httpx
        import pandas as pd
        from tqdm.asyncio import tqdm
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install with: pip install httpx pandas tqdm")
        sys.exit(1)
    
    # Run the scraper
    asyncio.run(main())
