from typing import Callable, Any
from tqdm import tqdm
import time

class ProgressBar:
    def __init__(self, func: Callable[..., Any]):
        self.func = func

    def __call__(self, *args, **kwargs):
        # Set up the progress bar
        with tqdm(total=100) as pbar:
            # Wrap the original function with a progress-updating function
            def update_progress(*args, **kwargs):
                result = self.func(*args, **kwargs)
                pbar.update(1)
                return result

            # Call the original function with the progress-updating wrapper
            result = update_progress(*args, **kwargs)

            # Wait until the progress bar is full
            while pbar.n < pbar.total:
                pbar.update(0)
                time.sleep(0.1)

            return result