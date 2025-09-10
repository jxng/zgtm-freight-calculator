import pandas as pd
import requests
import json
import time
import sys
#import openpyxl
import folium
import numpy as np
# import geopandas as gpd
from iteration_utilities import duplicates
import math 
import time
import random
import copy
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
from functools import partial
import logging

#from math import radians, cos, sin, asin, sqrt


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def fast_distance(df):
# highly optimized function to calculate distance between two points
    return haversine_np(df['lon1'], df['lat1'], df['lon2'], df['lat2'])



def merge_2_dataframe(data_1,data_2):
# This function is use to merge two dataframes together such that if the two original dataframes had X and Y rows respectively, the resulting dataframe will have X*Y rows
# This mainly used to associate each postcode/depot with a warehouse so we can find the cheapest route.

    # Dummy keys
    data_1['key'] = 1
    data_2['key'] = 1

    merged = pd.merge(data_1,data_2, on = 'key')

    del merged['key']
    del data_1['key']
    del data_2['key']
    
    return merged

def update_machship_prices(index, data, quote_json, bp_zone_location_df, dimensions_array, is_residential=True, hand_unload=True):
    """
    Update prices using MachShip API for a single record.
    
    Args:
        index: Index of the record in the dataframe
        data: DataFrame containing shipment data
        quote_json: Template JSON for MachShip API
        bp_zone_location_df: DataFrame with location data
        dimensions_array: Array with dimensions [length, width, height, weight, quantity]
        is_residential: Boolean indicating if delivery is to a residential address
        hand_unload: Boolean indicating if hand unloading is required
    
    Returns:
        None - updates data in place
    """
    locality_df = bp_zone_location_df.copy()
    warehouse_suburb = data['Warehouse Suburb'][index]
    
    quote_json_copy = copy.deepcopy(quote_json)
    
    # Set current date and time for dispatch
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    quote_json_copy["despatchDateTimeUtc"] = current_time
    quote_json_copy["despatchDateTimeLocal"] = current_time
    
    # Configure special instructions based on delivery options
    special_instructions = ""
    if is_residential:
        special_instructions += "Residential delivery. "
    if hand_unload:
        special_instructions += "Hand unload required. "
    quote_json_copy["specialInstructions"] = special_instructions
    
    # Setup from location (warehouse)
    quote_json_copy["fromName"] = "Z Grills Australia"
    quote_json_copy["fromAddressLine1"] = data['Warehouse Address'][index]
    quote_json_copy["fromLocation"]["suburb"] = warehouse_suburb
    quote_json_copy["fromLocation"]["postcode"] = str(data['Warehouse Postcode'][index])
    
    # Setup to location (customer)
    buyer_postcode = str(data['Customer Postcode'][index])
    buyer_suburb = data['Customer Locality'][index]
    quote_json_copy["toAddressLine1"] = data['Customer Address'][index] if 'Customer Address' in data.columns else ""
    quote_json_copy["toLocation"]["suburb"] = buyer_suburb
    quote_json_copy["toLocation"]["postcode"] = buyer_postcode
    
    # Configure item details
    quote_json_copy["items"][0] = {
        "companyItemId": 0,
        "itemType": 1,  # 1 appears to be for Pallet
        "name": "Pallet",
        "sku": "",
        "quantity": dimensions_array[4],
        "height": dimensions_array[2],
        "weight": dimensions_array[3],
        "length": dimensions_array[0],
        "width": dimensions_array[1],
        "palletSpaces": 1  # Assuming each pallet takes 1 space
    }
    
    # Use MachShip API endpoint
    url = "https://api.machship.com/"
    endpoint = "apiv2/quotes/createQuote"
    auth_key = "YOUR_MACHSHIP_API_KEY"  # You'll need to replace this with the actual API key
    headers = {"content-type": "application/json", "Authorization": "Bearer " + auth_key}
    
    # Get suburbs from matching postcode similar to the existing approach
    locality_arr = locality_df[locality_df['Customer Postcode'] == int(buyer_postcode)].reset_index(drop=True)
    locality_arr = locality_arr['Customer Locality'].tolist()
    
    # Place the current locality at the front of the list
    try:
        if buyer_suburb in locality_arr:
            locality_arr.insert(0, locality_arr.pop(locality_arr.index(buyer_suburb)))
    except Exception as e:
        pass
    
    time.sleep(0.1)
    # Try with the first suburb, if that fails try with other suburbs for the same postcode
    for target_locality in locality_arr:
        quote_json_copy["toLocation"]["suburb"] = target_locality
        payload = json.dumps(quote_json_copy)
        
        retry_attempts = 5  # Reduced from 200 in the original function
        for i in range(retry_attempts):
            time.sleep(0.01 * i * i)  # Introduce delay after each attempt
            try:
                r = requests.post(
                    url + endpoint, headers=headers,
                    data=payload, timeout=25)
                
                if r.status_code != 200 and i < 4:
                    sys.stdout.write('\r' + 'Request Error Occurred. Status: ' + str(r.status_code) + ' Attempts: ' + str(i) + '\n')
                elif r.status_code != 200 and i >= 4:
                    sys.stdout.write('\r' + 'Completed 5 Attempts. Affected Postcode: ' + buyer_postcode + ' to ' + buyer_suburb)
                    break
                else:
                    # Successful response
                    try:
                        response = r.json()
                        data.loc[index, 'Response'] = json.dumps(response)
                        # We can extract specific pricing information if needed
                        # data.loc[index, 'Price'] = response.get('price', np.nan)
                        return  # Exit the function once we have a successful response
                    except Exception as e:
                        sys.stdout.write('\r' + 'Error parsing JSON response: ' + str(e) + '\n')
                    break
            except requests.exceptions.Timeout:
                if i == retry_attempts - 1:
                    error_message = 'Timeout error occurred...: ' + str(i) + ' | Skip Postcode ' + str(buyer_postcode) + '\n'
                    sys.stdout.write('\r' + error_message)
            except Exception as e:
                sys.stdout.write('\r' + 'Error: ' + str(e) + '\n')
                break


def price_quote_machship(input_data, bp_zone_location_df, dimensions_array, is_residential=True, hand_unload=True):
    """
    Get price quotes from MachShip API for multiple records.
    
    Args:
        input_data: DataFrame containing shipment data
        bp_zone_location_df: DataFrame with location data
        dimensions_array: Array with dimensions [length, width, height, weight, quantity]
        is_residential: Boolean indicating if delivery is to a residential address
        hand_unload: Boolean indicating if hand unloading is required
    
    Returns:
        DataFrame with pricing information added
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('machship_quotes')
    
    # Define the JSON template directly in the function (no need for external file)
    quote_json = {
        "despatchDateTimeUtc": "2023-05-15T03:30:00.000Z",
        "despatchDateTimeLocal": "2023-05-15T13:30:00.000Z",
        "fromName": "Z Grills Australia",
        "fromAddressLine1": "123 Warehouse St",
        "fromLocation": {
            "suburb": "Epping",
            "state": "VIC",
            "postcode": "3076",
            "country": "AU"
        },
        "toName": "Customer",
        "toAddressLine1": "456 Customer St",
        "toLocation": {
            "suburb": "Richmond",
            "state": "VIC",
            "postcode": "3121",
            "country": "AU"
        },
        "items": [
            {
                "companyItemId": 0,
                "itemType": 1,
                "name": "Pallet",
                "sku": "",
                "quantity": 1,
                "height": 105,
                "weight": 110,
                "length": 84,
                "width": 57,
                "palletSpaces": 1
            }
        ],
        "specialInstructions": "",
        "customerReference": "ZG Quote",
        "dgsDeclaration": False,
        "questions": []
    }

    logger.info(f"Starting MachShip quote process for {len(input_data)} records")
    
    data = input_data.copy()
    # Copy the whole MachShip response into the CSV column
    data['Response'] = ''
    
    # Multiprocess the API calls
    df_indexes = data.index.to_list()
    threads = 1  # Start with 1 thread for more predictable behavior
    logger.info(f"Using {threads} thread(s) for API calls")
    
    # Define the update_machship_prices function inside to have access to logger
    def update_machship_prices(index, data, quote_json, bp_zone_location_df, dimensions_array, is_residential=True, hand_unload=True):
        """
        Update prices using MachShip API for a single record.
        
        Args:
            index: Index of the record in the dataframe
            data: DataFrame containing shipment data
            quote_json: Template JSON for MachShip API
            bp_zone_location_df: DataFrame with location data
            dimensions_array: Array with dimensions [length, width, height, weight, quantity]
            is_residential: Boolean indicating if delivery is to a residential address
            hand_unload: Boolean indicating if hand unloading is required
        
        Returns:
            None - updates data in place
        """
        try:
            locality_df = bp_zone_location_df.copy()
            warehouse_suburb = data['Warehouse Suburb'][index]
            
            quote_json_copy = copy.deepcopy(quote_json)
            
            # Set current date and time for dispatch
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            quote_json_copy["despatchDateTimeUtc"] = current_time
            quote_json_copy["despatchDateTimeLocal"] = current_time
            
            # Configure special instructions based on delivery options
            special_instructions = ""
            if is_residential:
                special_instructions += "Residential delivery. "
            if hand_unload:
                special_instructions += "Hand unload required. "
            quote_json_copy["specialInstructions"] = special_instructions
            
            # Setup question IDs as per requirements
            question_ids = []
            # Question ID 13 for Residential Delivery
            if is_residential:
                question_ids.append({"id": 13, "answer": "Yes"})
            
            # Question ID 36 for Hand Unload
            if hand_unload:
                question_ids.append({"id": 36, "answer": "Yes"})
            
            # Question ID 80 for Delivery Tailgate (commented out but available for future use)
            # question_ids.append({"id": 80, "answer": "Yes"})
            
            quote_json_copy["questions"] = question_ids
            
            # Setup from location (warehouse)
            quote_json_copy["fromName"] = "Z Grills Australia"
            quote_json_copy["fromAddressLine1"] = data['Warehouse Address'][index] if 'Warehouse Address' in data.columns else ""
            quote_json_copy["fromLocation"]["suburb"] = warehouse_suburb
            quote_json_copy["fromLocation"]["postcode"] = str(data['Warehouse Postcode'][index])
            if 'Warehouse State' in data.columns:
                quote_json_copy["fromLocation"]["state"] = data['Warehouse State'][index]
            
            # Setup to location (customer)
            buyer_postcode = str(data['Customer Postcode'][index])
            buyer_suburb = data['Customer Locality'][index]
            quote_json_copy["toAddressLine1"] = data['Customer Address'][index] if 'Customer Address' in data.columns else ""
            quote_json_copy["toLocation"]["suburb"] = buyer_suburb
            quote_json_copy["toLocation"]["postcode"] = buyer_postcode
            if 'Customer State' in data.columns:
                quote_json_copy["toLocation"]["state"] = data['Customer State'][index]
            
            # Configure item details based on product type
            product_type = data['Product Type'][index] if 'Product Type' in data.columns else 'grill'
            
            if product_type.lower() == 'pellet':
                item_name = "Pellets"
                # Pellets might have different dimensions
            else:  # Default to grill
                item_name = "Grill"
            
            quote_json_copy["items"][0] = {
                "companyItemId": 0,
                "itemType": 1,  # 1 appears to be for Pallet
                "name": item_name,
                "sku": "",
                "quantity": dimensions_array[4],
                "height": dimensions_array[2],
                "weight": dimensions_array[3],
                "length": dimensions_array[0],
                "width": dimensions_array[1],
                "palletSpaces": 1  # Assuming each pallet takes 1 space
            }
            
            # Use MachShip API endpoint
            url = "https://api.machship.com/"
            endpoint = "apiv2/quotes/createQuote"
            auth_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJGTVNUb2tlbiI6ImV5SmhiR2NpT2lKSVV6STFOaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpvZEhSd09pOHZjMk5vWlcxaGN5NTRiV3h6YjJGd0xtOXlaeTkzY3k4eU1EQTFMekExTDJsa1pXNTBhWFI1TDJOc1lXbHRjeTl1WVcxbGFXUmxiblJwWm1sbGNpSTZJakV6TWpVd0lpd2lhSFIwY0RvdkwzTmphR1Z0WVhNdWVHMXNjMjloY0M1dmNtY3ZkM012TWpBd05TOHdOUzlwWkdWdWRHbDBlUzlqYkdGcGJYTXZibUZ0WlNJNklrTjFjM1J2YlVCcGJuUmxaM0poZEdVM01URTFJaXdpYUhSMGNEb3ZMM05qYUdWdFlYTXVlRzFzYzI5aGNDNXZjbWN2ZDNNdk1qQXdOUzh3TlM5cFpHVnVkR2wwZVM5amJHRnBiWE12WlcxaGFXeGhaR1J5WlhOeklqb2lRM1Z6ZEc5dFFHbHVkR1ZuY21GMFpUY3hNVFV1YVdRaUxDSkJjM0JPWlhRdVNXUmxiblJwZEhrdVUyVmpkWEpwZEhsVGRHRnRjQ0k2SWtoSlNUSk5TakpYUTFCTlJrNUNObFZIUVROSVNWSTBRazlhTmpOSU5rZElJaXdpYUhSMGNEb3ZMM2QzZHk1aGMzQnVaWFJpYjJsc1pYSndiR0YwWlM1amIyMHZhV1JsYm5ScGRIa3ZZMnhoYVcxekwzUmxibUZ1ZEVsa0lqb2lPU0lzSWtOMWMzUnZiV1Z5U1VRaU9pSTNNVEUxSWl3aVFYQndiR2xqWVhScGIyNU9ZVzFsSWpvaVEzVnpkRzl0SWl3aWMzVmlJam9pTVRNeU5UQWlMQ0pxZEdraU9pSTBPRE15T1dReU1pMDFORGhpTFRReE5UTXRPVGt3WlMwd01EY3dOVE15TnpnNVpUa2lMQ0pwWVhRaU9qRTNNamszTlRNd05UZ3NJbWx6Y3lJNklrWk5VeUlzSW1GMVpDSTZJa1pOVXlKOS5LSEFoaUpRMDU5Qk5UNWhMYnNNNTVoZF94RjdWbmw4UWJfSDM1bEFuQ09nIiwibmJmIjoxNzI5NzUzMDU4LCJleHAiOjE3NjEyODkwNTgsImlzcyI6ImludGVncmF0ZS5jYXJpby5jb20uYXUiLCJhdWQiOiJpbnRlZ3JhdGVkY2FyaW9jdXN0b21lcnMifQ.E-KuuetkhAWtF-dm-1w133n82kq93svCXnhXKezQYio"
            headers = {"content-type": "application/json", "Authorization": "Bearer " + auth_key}
            
            # Get suburbs from matching postcode similar to the existing approach
            locality_arr = locality_df[locality_df['Customer Postcode'] == int(buyer_postcode)].reset_index(drop=True)
            if not locality_arr.empty:
                locality_arr = locality_arr['Customer Locality'].tolist()
                
                # Place the current locality at the front of the list
                try:
                    if buyer_suburb in locality_arr:
                        locality_arr.insert(0, locality_arr.pop(locality_arr.index(buyer_suburb)))
                except Exception as e:
                    logger.warning(f"Error reordering localities: {str(e)}")
            else:
                locality_arr = [buyer_suburb]  # Just use the provided suburb if no match in locality_df
                
            logger.debug(f"Processing index {index}, postcode: {buyer_postcode}, suburb: {buyer_suburb}")
            
            time.sleep(0.1)
            # Try with the first suburb, if that fails try with other suburbs for the same postcode
            for target_locality in locality_arr:
                quote_json_copy["toLocation"]["suburb"] = target_locality
                payload = json.dumps(quote_json_copy)
                
                retry_attempts = 5  # Reduced from 200 in the original function
                for i in range(retry_attempts):
                    time.sleep(0.01 * i * i)  # Introduce delay after each attempt
                    try:
                        logger.debug(f"Attempt {i+1} for index {index}, suburb: {target_locality}")
                        r = requests.post(
                            url + endpoint, headers=headers,
                            data=payload, timeout=25)
                        
                        if r.status_code != 200 and i < 4:
                            logger.warning(f"Request error: Status {r.status_code}, Attempt {i+1} for index {index}")
                        elif r.status_code != 200 and i >= 4:
                            logger.warning(f"All attempts failed for index {index}, postcode: {buyer_postcode}, suburb: {buyer_suburb}")
                            break
                        else:
                            # Successful response
                            try:
                                response = r.json()
                                data.loc[index, 'Response'] = json.dumps(response)
                                logger.info(f"Success for index {index}, postcode: {buyer_postcode}")
                                return  # Exit the function once we have a successful response
                            except Exception as e:
                                logger.error(f"Error parsing JSON response for index {index}: {str(e)}")
                            break
                    except requests.exceptions.Timeout:
                        logger.warning(f"Timeout on attempt {i+1} for index {index}")
                        if i == retry_attempts - 1:
                            logger.error(f"All retry attempts failed (timeout) for index {index}, postcode: {buyer_postcode}")
                    except Exception as e:
                        logger.error(f"Exception for index {index}: {str(e)}")
                        break
            
            # If we get here, all attempts for all localities failed
            logger.error(f"Failed to get quote for index {index}, postcode: {buyer_postcode} after trying all localities")
            data.loc[index, 'Response'] = json.dumps({"error": "Failed to get quote after all attempts"})
            
        except Exception as e:
            # Catch any exceptions in the overall process
            error_msg = f"Critical error processing index {index}: {str(e)}"
            logger.error(error_msg)
            data.loc[index, 'Response'] = json.dumps({"error": error_msg})
    
    try:
        with ThreadPool(processes=threads) as pool:
            list(tqdm(pool.imap_unordered(partial(
                update_machship_prices, data=data, quote_json=quote_json, 
                bp_zone_location_df=bp_zone_location_df, dimensions_array=dimensions_array,
                is_residential=is_residential, hand_unload=hand_unload), 
                df_indexes), total=len(df_indexes)))
    except Exception as e:
        logger.critical(f"Critical error in thread pool: {str(e)}")
    
    logger.info(f"Completed MachShip quote process for {len(input_data)} records")
    return data

# Function to validate locations with MachShip before quoting (can be used if needed)
def validate_machship_locations(suburb, postcode):
    """
    Validate location with MachShip API.
    
    Args:
        suburb: Suburb name
        postcode: Postcode
    
    Returns:
        List of valid locations or None if error
    """
    url = "https://api.machship.com/"
    endpoint = "apiv2/locations/returnLocations"
    auth_key = "YOUR_MACHSHIP_API_KEY"  # Replace with actual API key
    headers = {"content-type": "application/json", "Authorization": "Bearer " + auth_key}
    
    payload = json.dumps({
        "suburb": suburb,
        "postcode": postcode
    })
    
    try:
        r = requests.post(url + endpoint, headers=headers, data=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"Error validating location: {r.status_code}")
            return None
    except Exception as e:
        print(f"Exception during location validation: {str(e)}")
        return None
