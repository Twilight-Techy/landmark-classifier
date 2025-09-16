import signal
import time
import re
from contextlib import contextmanager
import requests

DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
MAX_TOKEN_RETRIES = 3
TOKEN_RETRY_DELAY = 5  # seconds between token fetch retries

KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor": "Google"}


def is_valid_token(token):
    """
    Validate that the token is suitable for use in HTTP headers.
    """
    if not token:
        return False
    
    # Check for invalid characters that would cause header issues
    invalid_chars = ['\n', '\r', '\t', '\0']
    for char in invalid_chars:
        if char in token:
            return False
    
    # Check if token starts or ends with whitespace
    if token != token.strip():
        return False
    
    # Basic format check - tokens should be alphanumeric with some special chars
    if not re.match(r'^[a-zA-Z0-9+/=._-]+$', token):
        return False
    
    return True


def fetch_token_with_retry(max_retries=MAX_TOKEN_RETRIES, retry_delay=TOKEN_RETRY_DELAY):
    """
    Fetch the keep-alive token with retry logic and validation.
    """
    for attempt in range(max_retries):
        try:
            print(f"Fetching keep-alive token (attempt {attempt + 1}/{max_retries})...")
            
            response = requests.get(
                TOKEN_URL, 
                headers=TOKEN_HEADERS, 
                timeout=10
            )
            response.raise_for_status()
            
            token = response.text.strip()
            
            if is_valid_token(token):
                print("Valid token obtained successfully")
                return token
            else:
                print(f"Invalid token received on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching token (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Unexpected error fetching token (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to obtain valid token after {max_retries} attempts")


def request_handler(headers, max_retries=2):
    """
    Create a signal handler for keep-alive requests with error handling.
    """
    def _handler(signum, frame):
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    KEEPALIVE_URL, 
                    headers=headers, 
                    timeout=10
                )
                response.raise_for_status()
                print("Keep-alive request successful")
                return
                
            except requests.exceptions.InvalidHeader as e:
                print(f"Invalid header error in keep-alive request: {e}")
                # This suggests token might be corrupted, but we can't easily refresh
                # it from within a signal handler
                break
                
            except requests.exceptions.RequestException as e:
                print(f"Keep-alive request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Brief delay before retry
                    
            except Exception as e:
                print(f"Unexpected error in keep-alive request: {e}")
                break
        
        print("All keep-alive request attempts failed")
    
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Context manager for maintaining active session with robust error handling.
    
    Example:
        from workspace_utils import active_session
        with active_session():
            # do long-running work here
    """
    print("Starting active session...")
    
    # Fetch token with retry logic
    try:
        token = fetch_token_with_retry()
        headers = {'Authorization': f"STAR {token}"}
    except Exception as e:
        print(f"Failed to initialize active session: {e}")
        print("Continuing without keep-alive functionality")
        yield
        return
    
    # Validate delay and interval
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    
    print(f"Setting up keep-alive with delay={delay}s, interval={interval}s")
    
    # Store original signal handler
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set up the keep-alive signal handler
        signal.signal(signal.SIGALRM, request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        
        print("Active session initialized successfully")
        yield
        
    except Exception as e:
        print(f"Error during active session: {e}")
        raise
        
    finally:
        # Always restore original signal handler and clear timer
        try:
            signal.signal(signal.SIGALRM, original_handler)
            signal.setitimer(signal.ITIMER_REAL, 0)
            print("Active session cleanup completed")
        except Exception as e:
            print(f"Error during active session cleanup: {e}")


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Keep session awake while iterating over an iterable.
    
    Example:
        from workspace_utils import keep_awake
        for i in keep_awake(range(5)):
            # do iteration with lots of work here
    """
    with active_session(delay, interval):
        yield from iterable


# Optional: Add a function to test the keep-alive functionality
def test_keep_alive():
    """
    Test function to verify keep-alive functionality is working.
    """
    try:
        token = fetch_token_with_retry()
        headers = {'Authorization': f"STAR {token}"}
        
        response = requests.post(KEEPALIVE_URL, headers=headers, timeout=10)
        response.raise_for_status()
        
        print("Keep-alive test successful!")
        return True
        
    except Exception as e:
        print(f"Keep-alive test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the functionality
    test_keep_alive()