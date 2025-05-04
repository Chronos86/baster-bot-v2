# bot/sentiment.py

import requests
from bs4 import BeautifulSoup
import re
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(__name__)

def get_sentiment(config: dict) -> float:
    """Scrapes CoinDesk headlines for Bitcoin sentiment.

    Args:
        config: Dictionary containing configuration, expected to have
                config["sentiment"]["coindesk_url"].

    Returns:
        A sentiment score between -1.0 (very negative) and 1.0 (very positive),
        or 0.0 if scraping fails or no relevant headlines are found.
    """
    try:
        url = config.get("sentiment", {}).get("coindesk_url")
        if not url:
            logger.warning("CoinDesk URL not found in configuration. Skipping sentiment analysis.")
            return 0.0

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        logger.info(f"Fetching headlines from {url} for sentiment analysis...")
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, "html.parser")

        # --- Headline Selection Logic --- 
        # This needs to be robust and potentially updated if CoinDesk changes layout.
        # Searching for common headline patterns within link tags.
        headlines_elements = soup.select("a h2, a h3, a h4, .card-title a") # Combine selectors
        
        if not headlines_elements:
            logger.warning(f"No headline elements found on {url} using current selectors. Check CoinDesk page structure.")
            return 0.0

        headlines_text = []
        for el in headlines_elements:
            # Extract text, preferring text within the link if structure is <a><h3>...</h3></a>
            text = el.get_text(strip=True)
            if text and len(text) > 10: # Basic filter for meaningful headlines
                headlines_text.append(text.lower())
        
        # Remove duplicates
        headlines_text = list(set(headlines_text))
        
        if not headlines_text:
             logger.warning(f"No meaningful headlines extracted from {url}. Check selectors and page content.")
             return 0.0
             
        logger.info(f"Found {len(headlines_text)} unique headlines. Analyzing first 10 for sentiment...")

        # --- Sentiment Scoring Logic --- 
        positive_words = ["bullish", "surge", "rise", "growth", "up", "high", "positive", "gain", "rally", "boom", "adopt", "support", "approve", "launch"]
        negative_words = ["bearish", "crash", "fall", "decline", "down", "low", "negative", "loss", "slump", "drop", "ban", "reject", "fear", "hack", "scam"]
        
        score = 0
        analyzed_count = 0

        for text in headlines_text[:10]: # Analyze up to 10 unique headlines
            logger.debug(f"Analyzing headline: {text}")
            headline_score = 0
            for word in positive_words:
                # Use word boundaries to avoid partial matches (e.g., \'up\' in \'support\')
                if re.search(r"\b" + re.escape(word) + r"\b", text):
                    headline_score += 1
            for word in negative_words:
                if re.search(r"\b" + re.escape(word) + r"\b", text):
                    headline_score -= 1
            
            # Add headline score to total, capping influence per headline
            score += max(min(headline_score, 1), -1) # Cap score per headline to [-1, 1]
            analyzed_count += 1

        if analyzed_count == 0:
            logger.warning("No headlines were analyzed.")
            return 0.0

        # Normalize final score based on the number of headlines analyzed
        # Max possible score is analyzed_count, min is -analyzed_count
        # Scale to [-1, 1]
        final_score = score / analyzed_count 
        final_score = max(min(final_score, 1.0), -1.0) # Ensure score is strictly within [-1, 1]

        logger.info(f"Calculated sentiment score: {final_score:.3f}")
        return final_score

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching CoinDesk page at {url}: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"An unexpected error occurred during sentiment analysis: {e}", exc_info=True)
        return 0.0

# --- Example Usage (for testing) --- 
if __name__ == \'__main__\':
    print("Testing sentiment analysis...")
    # Mock config for testing
    mock_config = {
        "sentiment": {
            "coindesk_url": "https://www.coindesk.com/tag/bitcoin"
            # "coindesk_url": "https://httpbin.org/html" # Use for basic HTML structure test
        }
    }
    sentiment = get_sentiment(mock_config)
    print(f"Sentiment Score: {sentiment}")

    mock_config_no_url = {"sentiment": {}}
    sentiment_no_url = get_sentiment(mock_config_no_url)
    print(f"Sentiment Score (no URL): {sentiment_no_url}")

    mock_config_bad_url = {"sentiment": {"coindesk_url": "https://invalid-url-that-does-not-exist.xyz"}}
    sentiment_bad_url = get_sentiment(mock_config_bad_url)
    print(f"Sentiment Score (bad URL): {sentiment_bad_url}")

