
import asyncio
import sys

import asyncio
import inspect # Helps check if the callback is async

def run_loop(cb, input_name="input: "):
    """
    Runs an asynchronous interactive loop, dynamically handling sync/async callbacks.
    """

    async def inner_loop():
        query = '' 
        while query != "quit":
            try:
                query = input(input_name).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Exiting loop.")
                break

            if query == "quit":
                break

            if query:
                try:
                    # Dynamically check if the callback needs 'await'
                    if inspect.iscoroutinefunction(cb):
                        # Await the asynchronous callback
                        await cb(query) 
                    else:
                        # Call the synchronous callback
                        cb(query)

                except Exception as e:
                    print(f"\n--- ERROR IN CALLBACK EXECUTION ---")
                    print(f"Error: {e}")
                    print(f"-----------------------------------\n")

    # Run the inner_loop asynchronous loop
    asyncio.run(inner_loop())
