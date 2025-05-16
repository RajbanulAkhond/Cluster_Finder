"""
Asynchronous utility functions for Materials Project API queries.

This module implements batched and asynchronous utilities for efficiently
retrieving data from the Materials Project API.
"""

import asyncio
import aiohttp
import logging
import os
from typing import List, Dict, Any, Optional, Union, Set
from mp_api.client import MPRester

logger = logging.getLogger(__name__)

# Global event loop for reuse
_event_loop = None

def get_event_loop():
    """
    Get or create an event loop safely.
    
    Returns:
        asyncio.AbstractEventLoop: A running event loop
    """
    global _event_loop
    
    try:
        # Try to get the existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            _event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_event_loop)
            return _event_loop
        return loop
    except RuntimeError:
        # If no event loop exists, create one
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
        return _event_loop

def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """
    Get API key from provided value or environment variable.
    
    Parameters:
        api_key (str, optional): API key provided by the caller

    Returns:
        str: API key from argument or environment variable, or None if not found
        
    Raises:
        ValueError: If the API key format is invalid (not 32 characters for new API)
    """
    if api_key:
        key = api_key
    else:
        # Try to get API key from environment variable
        key = os.environ.get("MAPI_KEY")
    
    # Check if we have a key and if it's in the right format
    if key:
        if len(key) == 16:
            logger.warning(
                "You're using a legacy 16-character Materials Project API key. "
                "Please get a new 32-character API key from https://materialsproject.org/api"
            )
        elif len(key) != 32:
            logger.warning(
                f"API key has unexpected length ({len(key)} characters). "
                "New Materials Project API keys should be 32 characters."
            )
    
    return key

async def async_batch_get_properties(
    material_ids: List[str],
    properties: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 50
) -> Dict[str, Dict[str, Any]]:
    """
    Asynchronously retrieve properties for multiple materials using batched API requests.
    
    Parameters:
        material_ids (List[str]): List of Materials Project IDs
        properties (List[str]): List of properties to retrieve
        api_key (str, optional): Materials Project API key or None to use environment variable
        batch_size (int, optional): Number of materials to query in each batch
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping material IDs to their properties
    """
    # Get API key from argument or environment variable
    api_key = get_api_key(api_key)
    
    results = {}
    # Create batches of material IDs
    batches = [material_ids[i:i+batch_size] for i in range(0, len(material_ids), batch_size)]
    
    # List to store all the tasks
    tasks = []
    
    async def process_batch(batch, semaphore):
        async with semaphore:
            try:
                with MPRester(api_key) as mpr:
                    # Create query fields based on the properties requested
                    fields = ["material_id"] + properties
                    
                    # Execute batch query
                    docs = mpr.summary.search(material_ids=batch, fields=fields)
                    
                    # Process results
                    batch_results = {}
                    for doc in docs:
                        prop_dict = {}
                        for prop in properties:
                            if hasattr(doc, prop):
                                prop_dict[prop] = getattr(doc, prop)
                        if prop_dict:  # Only add if we found at least one property
                            batch_results[doc.material_id] = prop_dict
                    
                    return batch_results
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                return {}
    
    # Create a semaphore to limit concurrency (to avoid overwhelming the API)
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
    
    # Create tasks for each batch
    for batch in batches:
        task = asyncio.create_task(process_batch(batch, semaphore))
        tasks.append(task)
    
    # Await all tasks
    batch_results = await asyncio.gather(*tasks)
    
    # Merge results from all batches
    for batch_result in batch_results:
        results.update(batch_result)
    
    return results

async def async_http_batch_query(
    material_ids: List[str],
    properties: List[str],
    api_key: str,
    batch_size: int = 50
) -> Dict[str, Dict[str, Any]]:
    """
    Perform batched HTTP requests directly to the Materials Project API.
    
    This is a fallback method when the MPRester approach doesn't work.
    
    Parameters:
        material_ids (List[str]): List of Materials Project IDs
        properties (List[str]): List of properties to retrieve
        api_key (str): Materials Project API key
        batch_size (int, optional): Number of materials to query in each batch
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping material IDs to their properties
    """
    # Get API key from argument or environment variable
    api_key = get_api_key(api_key)
    
    results = {}
    # Create batches of material IDs
    batches = [material_ids[i:i+batch_size] for i in range(0, len(material_ids), batch_size)]
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(5)
    
    async def process_batch(batch, session):
        async with semaphore:
            headers = {"X-API-KEY": api_key}
            query_string = ",".join(batch)
            url = f"https://api.materialsproject.org/materials/summary?material_ids={query_string}"
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        batch_results = {}
                        
                        if "data" in data:
                            for item in data["data"]:
                                material_id = item.get("material_id")
                                if material_id:
                                    prop_dict = {}
                                    for prop in properties:
                                        if prop in item:
                                            prop_dict[prop] = item[prop]
                                    
                                    if prop_dict:  # Only add if we found at least one property
                                        batch_results[material_id] = prop_dict
                        
                        return batch_results
                    else:
                        logger.error(f"HTTP request failed with status {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Error with HTTP request: {e}")
                return {}
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_batch(batch, session) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
    
    # Merge results from all batches
    for batch_result in batch_results:
        results.update(batch_result)
    
    return results

def get_properties_batch(
    material_ids: List[str],
    properties: List[str],
    api_key: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Synchronous wrapper for async_batch_get_properties.
    
    Parameters:
        material_ids (List[str]): List of Materials Project IDs
        properties (List[str]): List of properties to retrieve
        api_key (str, optional): Materials Project API key
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping material IDs to their properties
    """
    # Get or create an event loop (our improved function)
    loop = get_event_loop()
    
    try:
        # Try with MPRester first
        results = loop.run_until_complete(
            async_batch_get_properties(material_ids, properties, api_key)
        )
        
        # If we got results for all materials, return them
        if len(results) == len(material_ids):
            return results
        
        # Otherwise, try with direct HTTP requests for missing materials
        missing_ids = [mid for mid in material_ids if mid not in results]
        if missing_ids:
            logger.info(f"Trying direct HTTP requests for {len(missing_ids)} missing materials")
            http_results = loop.run_until_complete(
                async_http_batch_query(missing_ids, properties, api_key)
            )
            results.update(http_results)
        
        return results
    except Exception as e:
        logger.error(f"Error in batch property retrieval: {e}")
        # In case of any issues, fall back to a sequential approach
        results = {}
        with MPRester(api_key) as mpr:
            for material_id in material_ids:
                try:
                    docs = mpr.summary.search(material_ids=[material_id], fields=["material_id"] + properties)
                    if docs and len(docs) > 0:
                        doc = docs[0]
                        prop_dict = {}
                        for prop in properties:
                            if hasattr(doc, prop):
                                prop_dict[prop] = getattr(doc, prop)
                        if prop_dict:
                            results[material_id] = prop_dict
                except Exception as inner_e:
                    logger.error(f"Error retrieving {material_id}: {inner_e}")
        return results

def search_compounds_batch(
    elements: List[str],
    api_key: str,
    min_elements: int = 2,
    max_elements: int = 4,
    min_magnetization: float = 0.001,
    max_magnetization: float = 3,
    include_fields: Optional[List[str]] = None
) -> List[Any]:
    """
    Search for compounds with batched requests to Materials Project.
    
    Parameters:
        elements (List[str]): List of elements to include in the search
        api_key (str): Materials Project API key
        min_elements (int): Minimum number of elements in compound
        max_elements (int): Maximum number of elements in compound
        min_magnetization (float): Minimum magnetization value
        max_magnetization (float): Maximum magnetization value
        include_fields (List[str], optional): Additional fields to include in results
        
    Returns:
        List[Any]: List of matching entries
    """
    # Get API key from argument or environment variable
    api_key = get_api_key(api_key)
    
    # Define default fields to retrieve
    default_fields = ["material_id", "formula_pretty", "structure", "total_magnetization"]
    
    # Combine with additional fields if provided
    if include_fields:
        fields = list(set(default_fields + include_fields))
    else:
        fields = default_fields
    
    # This search is already efficiently implemented as a batch request in MPRester
    with MPRester(api_key) as mpr:
        entries = mpr.materials.summary.search(
            elements=elements,
            num_elements=(min_elements, max_elements),
            total_magnetization=(min_magnetization, max_magnetization),
            fields=fields
        )
    
    return entries