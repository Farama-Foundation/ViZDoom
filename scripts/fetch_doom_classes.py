#!/usr/bin/env python3
"""
Script to fetch Doom class names from ZDoom wiki and generate C++ header file with categories.

This script scrapes the ZDoom wiki to extract Doom class names and their associated
metadata (DoomEd numbers, Spawn IDs, and Identifiers). It also fetches ZDoom categories
from the Spawnable page and organizes classes by category. It then generates a C++ header
file containing the required data structures with definitions.

The generated C++ header file includes:
- All Doom classes found on the ZDoom wiki
- ZDoom categories organized by type (Monster, Weapon, Ammo, etc.)
- Deduplicated class-to-category mapping by setting primary category
- Proper C++ includes and pragma once directive
- Definitions of all data structures

Features:
- Caching system to reduce server load

Usage:
    python fetch_doom_classes_WIP.py [-o output_base] [-cp copy_to] [-s sleep_delay] [--cache-dir cache_dir] [--force-refresh] [--cache-only] [--clear-cache]

Dependencies:
    - bs4 (BeautifulSoup4) for HTML parsing
    - curl command-line tool for web requests
    - tqdm for progress bars (optional)
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import bs4


try:
    from tqdm import tqdm
except ImportError:
    tqdm = list


class CacheManager:
    """Manages caching for web requests to reduce server load."""

    def __init__(self, cache_dir: str = ".cache", default_ttl_hours: int = 24):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files
            default_ttl_hours: Default time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.stats = {"hits": 0, "misses": 0, "expired": 0, "errors": 0}

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key for a URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a URL."""
        cache_key = self._get_cache_key(url)
        return self.cache_dir / f"{cache_key}.json"

    def get(self, url: str, ttl: Optional[timedelta] = None) -> Optional[str]:
        """
        Get cached content for a URL.

        Args:
            url: The URL to get cached content for
            ttl: Time-to-live override (uses default if None)

        Returns:
            Cached content if valid, None otherwise
        """
        cache_path = self._get_cache_path(url)
        ttl = ttl or self.default_ttl

        if not cache_path.exists():
            self.stats["misses"] += 1
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                cache_data = json.load(f)

            # Check if cache is expired
            cached_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cached_time > ttl:
                self.stats["expired"] += 1
                return None

            self.stats["hits"] += 1
            return cache_data["content"]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Cache error for {url}: {e}", file=sys.stderr)
            self.stats["errors"] += 1
            return None

    def set(self, url: str, content: str) -> None:
        """
        Cache content for a URL.

        Args:
            url: The URL to cache content for
            content: The content to cache
        """
        cache_path = self._get_cache_path(url)

        try:
            cache_data = {
                "url": url,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            print(f"Failed to cache {url}: {e}", file=sys.stderr)
            self.stats["errors"] += 1

    def clear(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Failed to delete {cache_file}: {e}", file=sys.stderr)

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self.stats.copy()

    def print_stats(self) -> None:
        """Print cache statistics."""
        stats = self.get_stats()
        total = stats["hits"] + stats["misses"] + stats["expired"]
        if total > 0:
            hit_rate = (stats["hits"] / total) * 100
            print(
                f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, "
                f"{stats['expired']} expired, {stats['errors']} errors "
                f"({hit_rate:.1f}% hit rate)"
            )
        else:
            print("No cache activity recorded")


SCRIPT_NAME = (
    f"{os.path.split(os.path.dirname(__file__))[1]}/{os.path.basename(__file__)}"
)


def fetch_url_with_cache(
    cache_manager: CacheManager, url: str, force_refresh: bool = False
) -> str:
    """
    Fetch URL content with caching support.

    Args:
        cache_manager: The cache manager instance
        url: The URL to fetch
        force_refresh: If True, ignore cache and fetch fresh content

    Returns:
        The HTML content from the URL or cache

    Raises:
        subprocess.CalledProcessError: If the curl command fails
    """
    if not force_refresh:
        cached_content = cache_manager.get(url)
        if cached_content:
            print(f"Using cached content for {url}")
            return cached_content

    print(f"Fetching fresh content from {url}")
    try:
        result = subprocess.run(
            ["curl", "-s", url], capture_output=True, text=True, check=True
        )
        html_content = result.stdout

        # Cache the fresh content
        cache_manager.set(url, html_content)
        return html_content

    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch {url}: {e}", file=sys.stderr)
        raise


def fetch_doom_classes(
    cache_manager: CacheManager,
    url: str = "https://zdoom.org/wiki/Classes:Doom",
    force_refresh: bool = False,
) -> list[str]:
    """
    Fetch Doom class names from the ZDoom wiki with caching.

    This function scrapes the ZDoom wiki page to extract class names from links
    that follow the pattern "/wiki/Classes:ClassName".

    Args:
        cache_manager: The cache manager instance
        url (str): The URL to fetch class names from (default: "https://zdoom.org/wiki/Classes:Doom")
        force_refresh (bool): If True, ignore cache and fetch fresh content

    Returns:
        List[str]: Sorted list of unique Doom class names

    Raises:
        subprocess.CalledProcessError: If the curl command fails to fetch the page
        ValueError: If the main content area cannot be found in the HTML
    """
    html_content = fetch_url_with_cache(cache_manager, url, force_refresh)

    soup = bs4.BeautifulSoup(html_content, "html.parser")

    # Find the main content area
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        raise ValueError("Could not find main content area")

    class_names: list[str] = []

    # Look for class links in <pre> tags (where the class lists are)
    for pre_tag in content.find_all("pre"):  # type: ignore
        for link in pre_tag.find_all("a", href=True):
            href = link.get("href", "")
            # Check if it's a link to a class page (Classes:ClassName format)
            if href and href.startswith("/wiki/Classes:"):
                # Extract the class name from the href
                class_name = href.replace("/wiki/Classes:", "")
                # Skip if it's not a valid class name (contains special characters)
                if (
                    re.match(r"^[A-Za-z0-9_]+$", class_name)
                    and class_name not in class_names
                ):
                    class_names.append(class_name)

    return sorted(list(set(class_names)))


def fetch_class_data(
    cache_manager: CacheManager,
    class_names: list[str],
    fetch_delay: float = 5.0,
    force_refresh: bool = False,
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """
    Fetch DoomEd numbers, Spawn IDs, and Identifiers from individual class pages with caching.

    This function iterates through each class name and fetches its individual
    wiki page to extract metadata like DoomEd numbers, Spawn IDs, and Identifiers
    from HTML tables. It includes a configurable delay between requests to be
    respectful to the server and uses caching to reduce server load.

    Args:
        cache_manager: The cache manager instance
        class_names: List of class names to fetch data for
        fetch_delay: Delay in seconds between requests (default: 5.0)
        force_refresh: If True, ignore cache and fetch fresh content

    Returns:
        Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]: A tuple containing:
            - doomed_numbers: Dictionary mapping class names to DoomEd numbers
            - spawn_ids: Dictionary mapping class names to Spawn IDs
            - identifiers: Dictionary mapping class names to Identifiers
    """
    doomed_numbers: dict[str, int] = {}
    spawn_ids: dict[str, int] = {}
    identifiers: dict[str, str] = {}

    for class_name in tqdm(class_names):
        try:
            # Fetch the individual class page with caching
            url = f"https://zdoom.org/wiki/Classes:{class_name}"
            # Add a small delay to be respectful to the server (only for fresh requests)
            if not cache_manager.get(url):
                time.sleep(fetch_delay)
            html_content = fetch_url_with_cache(cache_manager, url, force_refresh)

            soup = bs4.BeautifulSoup(html_content, "html.parser")
            content = soup.find("div", {"id": "mw-content-text"})
            if not content:
                continue

            # Look for tables with class data
            for table in content.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        for i in range(0, len(cells), 2):
                            header = cells[i].get_text().strip().lower()
                            value = cells[i + 1].get_text().strip()

                            # Extract DoomEd number
                            if "doomed" in header or "editor number" in header:
                                number_match = re.search(r"\b(\d+)\b", value)
                                if number_match:
                                    doomed_numbers[class_name] = int(
                                        number_match.group(1)
                                    )

                            # Extract Spawn ID
                            elif "spawn" in header and "id" in header:
                                id_match = re.search(r"\b(\d+)\b", value)
                                if id_match:
                                    spawn_ids[class_name] = int(id_match.group(1))

                            # Extract Identifier
                            elif "identifier" in header:
                                identifiers[class_name] = value

        except subprocess.CalledProcessError as e:
            print(f"Failed to fetch data for {class_name}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error processing {class_name}: {e}", file=sys.stderr)
            continue

    return doomed_numbers, spawn_ids, identifiers


def fetch_zdoom_categories_by_type(
    cache_manager: CacheManager,
    url: str = "https://zdoom.org/wiki/Category:Spawnable",
    force_refresh: bool = False,
) -> dict[str, list[str]]:
    """
    Fetch ZDoom categories and their associated classes from the Spawnable page.

    This function scrapes the ZDoom Spawnable page to extract category links and then
    visits each category page to get the list of classes in that category.

    Args:
        cache_manager: The cache manager instance
        url (str): The URL to fetch categories from (default: "https://zdoom.org/wiki/Category:Spawnable")
        force_refresh (bool): If True, ignore cache and fetch fresh content

    Returns:
        Dict[str, List[str]]: Dictionary with categories as keys and lists of class names as values

    Raises:
        subprocess.CalledProcessError: If curl command fails
        Exception: If HTML parsing fails
    """
    print(f"Fetching ZDoom categories from: {url}")

    # Fetch the main Spawnable page
    html_content = fetch_url_with_cache(cache_manager, url, force_refresh)
    soup = bs4.BeautifulSoup(html_content, "html.parser")

    # Find the mw-parser-output section
    parser_output = soup.find("div", {"class": "mw-parser-output"})
    if not parser_output:
        print("Could not find mw-parser-output section")
        return {}

    # Find the "Categories by type" section
    categories_section = None
    for h3 in parser_output.find_all("h3"):
        if "Categories by type" in h3.get_text():
            categories_section = h3.find_next("table")
            break

    if not categories_section:
        print("Could not find 'Categories by type' section")
        return {}

    # Extract category links
    category_links = []
    for link in categories_section.find_all("a", href=True):
        href = link.get("href", "")
        if href and href.startswith("/wiki/Category:"):
            category_name = href.replace("/wiki/Category:", "")
            category_links.append((category_name, href))

    print(f"Found {len(category_links)} category links")

    # Fetch classes for each category
    categories: dict[str, list[str]] = {}

    for category_name, category_href in tqdm(category_links):
        try:
            category_url = f"https://zdoom.org{category_href}"
            category_html = fetch_url_with_cache(
                cache_manager, category_url, force_refresh
            )
            category_soup = bs4.BeautifulSoup(category_html, "html.parser")

            # Find the mw-category-generated section
            category_generated = category_soup.find(
                "div", {"class": "mw-category-generated"}
            )  # type: ignore
            if not category_generated:
                continue

            # Extract class links from all pages (handle pagination)
            class_names = []
            current_url = category_url
            current_soup = category_soup

            while current_url and current_soup:
                # Find the mw-category-generated section
                category_generated = current_soup.find(
                    "div", {"class": "mw-category-generated"}
                )  # type: ignore
                if not category_generated:
                    break

                # Find the "Pages in category" section
                pages_section = category_generated.find("div", {"id": "mw-pages"})  # type: ignore
                if not pages_section:
                    break

                # Extract class links from current page
                for link in pages_section.find_all("a", href=True):
                    href = link.get("href", "")
                    if href and href.startswith("/wiki/Classes:"):
                        class_name = href.replace("/wiki/Classes:", "")
                        # Skip if it's not a valid class name (contains special characters)
                        if re.match(r"^[A-Za-z0-9_]+$", class_name):
                            class_names.append(class_name)

                # Check for next page link
                next_page_link = None
                for link in category_generated.find_all("a", href=True):
                    href = link.get("href", "")
                    link_text = link.get_text().lower()
                    if href and "pagefrom=" in href and "next page" in link_text:
                        next_page_link = href
                        break

                if next_page_link:
                    # Fetch next page
                    if next_page_link.startswith("/w/"):
                        next_page_url = f"https://zdoom.org{next_page_link}"
                    else:
                        next_page_url = str(next_page_link)

                    print(f"    Fetching next page for {category_name}...")
                    category_html = fetch_url_with_cache(
                        cache_manager, next_page_url, force_refresh
                    )
                    current_soup = bs4.BeautifulSoup(category_html, "html.parser")
                    current_url = next_page_url
                else:
                    current_url = None

            if class_names:
                categories[category_name] = sorted(class_names)
                print(f"  {category_name}: {len(class_names)} classes")

        except Exception as e:
            print(f"Error processing category {category_name}: {e}", file=sys.stderr)
            continue

    print(f"Extracted classes from {len(categories)} categories")
    return categories


def generate_header_file(
    class_names: list[str],
    categories: dict[str, list[str]],
    output_base: str = "doom_classes",
) -> None:
    """
    Generate C++ header file from the list of class names and categories.

    This function creates a single header file (.h) with definitions for all
    the required data structures.

    Args:
        class_names: List of Doom class names to include
        categories: Dictionary with categories as keys and lists of class names as values
        output_base: Base name for output file (without extension)
    """
    # Casefold all class names
    casefolded_class_names = [class_name.casefold() for class_name in class_names]

    # Add the special MarineChainsawVzd class to Monster category
    marine_chainsaw_vzd = "marinechainsawvzd"
    if marine_chainsaw_vzd not in casefolded_class_names:
        casefolded_class_names.append(marine_chainsaw_vzd)

    # Casefold categories and their class lists
    casefolded_categories: dict[str, list[str]] = dict()
    for category, class_list in categories.items():
        locate_underscore = category.find("_")
        if locate_underscore > 0:
            next_char = category[locate_underscore + 1]
            category = category.replace("_" + next_char, next_char.upper())
        casefolded_class_list = [class_name.casefold() for class_name in class_list]
        casefolded_categories[category] = casefolded_class_list

    # Add MarineChainsawVzd to Monster category if it exists
    if "Monster" in casefolded_categories:
        if marine_chainsaw_vzd not in casefolded_categories["Monster"]:
            casefolded_categories["Monster"].append(marine_chainsaw_vzd)
            casefolded_categories["Monster"].sort()  # Keep sorted

    # Remove classes that have multiple categories, use primary category
    print(
        "Handling classes that have multiple categories by setting primary category..."
    )
    # https://www.zdoom.org/wiki/Classes:HeadCandles
    casefolded_categories["Gore"].remove("headcandles")
    # https://www.zdoom.org/wiki/Classes:PlayerPawn
    casefolded_categories["Player"].remove("playerpawn")
    # https://www.zdoom.org/wiki/Classes:ZCorpseSitting
    casefolded_categories["Breakable"].remove("zcorpsesitting")

    # Check if we are successful
    for class_name in casefolded_class_names:
        appearance = sum(
            class_name in class_list for class_list in casefolded_categories.values()
        )
        assert (
            appearance == 1
        ), f"Class {class_name} has appeared in {appearance} categories"

    # Generate header file with definitions
    header_file = f"{output_base}.h"

    # Create the header content with definitions
    header_content = f"""#pragma once

#include <string>
#include <vector>
#include <unordered_map>

// Doom class information auto-generated from ZDoom wiki.
// Generated by {SCRIPT_NAME}

"""  # noqa

    # Add category to classes mapping
    header_content += "// Listing default object categories in Doom\n"
    header_content += "const std::vector<std::string> categories = {\n"

    for category, class_list in sorted(casefolded_categories.items()):
        if category == "SFX":
            header_content += '    "Self",\n'
        header_content += f'    "{category}",\n'

    header_content += "};\n\n"

    # Add class to category mapping
    header_content += "// Mapping from class names to their category\n"
    header_content += (
        "const std::unordered_map<std::string, std::string> classToCategory = {\n"
    )

    for category, class_list in sorted(casefolded_categories.items()):
        for class_name in class_list:
            header_content += f'    {{"{class_name}", "{category}"}},\n'

    header_content += "};\n"

    with open(header_file, "w", encoding="utf-8") as f:
        f.write(header_content)

    print(f"Generated header file: {header_file}")
    print(
        f"Generated C++ header file with {len(casefolded_class_names)} doom classes from {len(casefolded_categories)} categories"
    )
    if marine_chainsaw_vzd in casefolded_class_names:
        print(f"Added special class: {marine_chainsaw_vzd} to Monster category")


def main() -> None:
    """Main function to fetch classes and generate C++ header file with ZDoom categories."""
    parser = argparse.ArgumentParser(
        description="Fetch Doom class names from ZDoom wiki and generate C++ header file with ZDoom categories (with caching)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="src/vizdoom/src/viz_doom_classes",
        help="Base name for output file (without extension) (default: src/vizdoom/src/viz_doom_classes)",
    )
    parser.add_argument(
        "-cp",
        "--copy-to",
        default="",
        help="Copy the generated file to the specified directory (default: empty)",
    )
    parser.add_argument(
        "-s",
        "--sleep",
        type=float,
        default=5.0,
        help="Delay (in seconds) between each request (default: 5)",
    )
    parser.add_argument(
        "--cache-dir",
        default="test_cache",
        help="Directory to store cache files (default: test_cache)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=8766,
        help="Cache time-to-live in hours (default: 8766, one year)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh all cached data",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use only cached data, don't fetch fresh content",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached data before running",
    )

    args = parser.parse_args()
    if args.sleep < 0.75:
        raise ValueError(
            f"Sleep time {args.sleep} < 0.75s!\nPlease be gentle with zdoom.org :-("
        )

    # Initialize cache manager
    cache_manager = CacheManager(args.cache_dir, args.cache_ttl)

    if args.clear_cache:
        print("Clearing cache...")
        cache_manager.clear()
        print("Cache cleared.")

    try:
        # Fetch ZDoom categories by type first
        print("Fetching ZDoom categories by type...")
        categories = fetch_zdoom_categories_by_type(
            cache_manager, force_refresh=args.force_refresh
        )

        if not categories:
            print(
                "No categories found. Check if the wiki structure has changed.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Flatten categories into a single list of class names
        print("Flattening categories into class list...")
        all_class_names = set()
        for class_list in categories.values():
            all_class_names.update(class_list)

        class_names = sorted(list(all_class_names))
        print(
            f"Found {len(class_names)} unique class names from {len(categories)} categories"
        )

        # Fetch lookup tables from individual class pages (optional, for future use)
        # print("Fetching class data from individual pages...")
        # doomed_numbers, spawn_ids, identifiers = fetch_class_data(
        #     cache_manager, class_names, args.sleep, args.force_refresh
        # )

        # Generate header file
        generate_header_file(class_names, categories, args.output)

        # Print cache statistics
        cache_manager.print_stats()

        print(f"Successfully generated C++ header file!\n- {args.output}.h")

        if args.copy_to:
            print(f"Copying generated file to {args.copy_to}...")
            subprocess.run(
                ["cp", f"{args.output}.h", args.copy_to],
                check=True,
            )
            print(f"Copied file to {args.copy_to}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
