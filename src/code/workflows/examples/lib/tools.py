import os
from langchain.tools import tool
from langchain_google_community.search import GoogleSearchAPIWrapper
from google import genai
from google.genai import types
from langchain_core.tools import Tool 
import json
import os
import shutil
from typing import Any, AsyncGenerator, Dict, List, Optional
from langchain.tools import tool # Required for the @tool decorator
from langchain.tools import tool
from google import genai
from google.genai import types
import math # Keep math import if you later need functions like sin/cos/sqrt
from langchain.tools import tool # Required for the @tool decorator


# -------------------------------
# 1. Calculator Tool (Consolidated)
# -------------------------------

@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression (e.g., '10 * (5 + 2) / 7') and returns the result.
    It supports basic arithmetic operators (+, -, *, /), parentheses, and floats.
    
    Args:
        expression: The mathematical expression to evaluate as a string.
    """
    try:
        # Step 1: Sanitize the input to prevent malicious code execution.
        # Allow digits, parentheses, basic operators, and decimal points.
        allowed_chars = "0123456789.()+-*/ "
        sanitized_expression = "".join(c for c in expression if c in allowed_chars)
        
        # Check if the sanitized expression is empty or just whitespace
        if not sanitized_expression.strip():
             return f"ERROR: Invalid or empty mathematical expression."

        # Step 2: Evaluate the expression.
        # We rely on the sanitization to ensure safety before using eval().
        result = eval(sanitized_expression)
        
        return f"SUCCESS: The result of '{expression}' is {result}."
        
    except ZeroDivisionError:
        return f"ERROR: Division by zero encountered in expression: '{expression}'."
    except SyntaxError:
        return f"ERROR: Invalid mathematical syntax in expression: '{expression}'."
    except Exception as e:
        # Catch all other possible errors
        return f"ERROR evaluating expression '{expression}': {type(e).__name__}: {e}"



# -------------------------------
# 2. Code  Executing Tools
# -------------------------------

@tool
def code_execute(contents:str):
    """
    Executes python code
    
    Args:
        contents: content of the code.
    """
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution)]
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        if part.executable_code is not None:
            print(part.executable_code.code)
        if part.code_execution_result is not None:
            print(part.code_execution_result.output)

# -------------------------------
# 2. Code Executing Tool (Robust)
# -------------------------------
import subprocess
import shlex
from langchain.tools import tool


@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command (e.g., 'ls -l', 'python my_script.py') in a /bin/sh
    shell on the local machine and returns the complete stdout and stderr.
    Command     Description                                         Example Usage
    -------     -----------                                         -------------
    dir         Displays a list of files and subdirectories.        dir "C:/Windows"
    cd          Changes the current working directory.              cd "C:/Users/Username/Documents"
    cd ..       Moves up one directory level.                       cd ..
    cd "/"      Moves to the root directory of the current drive.   cd "/"
    mkdir       Creates a new directory (folder).                   mkdir NewFolder
    rmdir       Deletes an empty directory.                         rmdir OldFolder
    copy        Copies one or more files to another location.       copy "file.txt" "D:/backup"
    move        Moves files or directories, or renames a directory. move "old_name.txt" "new_location\"
    del         Deletes files.                                      del old_file.txt
    ren         Renames files or directories.                       ren oldname.txt newname.txt
    type        Displays the contents of a text file.               type readme.txt
    tree        Graphically displays the folder structure           tree "C:/Users/Username" /F
            



    
    Args:
        command: The shell command to execute.
    """
    try:
        # Use shlex.split() to handle quoted arguments safely
        # Note: This is safer but not a perfect sandbox.
        # For simple commands, this is fine. For complex user input,
        # 'shell=True' is dangerous, so we avoid it.
        # We set shell=True here to allow for commands like 'ls -l' or 'grep "a" file.txt'
        # which are hard to parse otherwise.
        print(command)
        # Let's use a 5-second timeout for safety
        process = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10, # 10-second timeout
            check=True   # Raise an error if the command fails
        )
        
        stdout = process.stdout
        stderr = process.stderr
        print(stdout, stderr)
        
        if stdout and stderr:
            return f"SUCCESS:\n[STDOUT]:\n{stdout}\n[STDERR]:\n{stderr}"
        elif stdout:
            return f"SUCCESS:\n[STDOUT]:\n{stdout}"
        
        elif stderr:
            return f"SUCCESS (No STDOUT):\n[STDERR]:\n{stderr}"
        else:
            return "SUCCESS: Command ran with no output to STDOUT or STDERR."

    except subprocess.CalledProcessError as e:
        # Command returned a non-zero exit code
        return (
            f"ERROR: Command failed with exit code {e.returncode}.\n"
            f"[STDOUT]:\n{e.stdout}\n"
            f"[STDERR]:\n{e.stderr}"
        )
    except subprocess.TimeoutExpired as e:
        return f"ERROR: Command timed out after 10 seconds.\n[STDOUT]:\n{e.stdout}\n[STDERR]:\n{e.stderr}"
    except Exception as e:
        return f"ERROR executing command '{command}': {type(e).__name__}: {e}"
               

# -------------------------------
# 2. File System Tools (Enhanced)
# -------------------------------
# Define limits to protect the LLM's context window
MAX_READ_SIZE = 10000  # Max characters to read from a file
MAX_LIST_COUNT = 100   # Max items to list from a directory

# -------------------------------
# 2. File System Tool (Consolidated)
# -------------------------------

def _read_file_logic(file_path: str) -> str:
    """Internal logic for reading a file, truncating large ones."""
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found at '{file_path}'."
        if os.path.isdir(file_path):
            return f"ERROR: Path '{file_path}' is a directory. Use 'list' operation."
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return f"SUCCESS: Read file '{file_path}'. The file is [EMPTY]."

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(MAX_READ_SIZE + 1)
        
        if len(content) > MAX_READ_SIZE:
            return f"SUCCESS: Read file '{file_path}'. Content [TRUNCATED] to {MAX_READ_SIZE} chars:\n{content[:MAX_READ_SIZE]}..."
        else:
            return f"SUCCESS: Read file '{file_path}'. Content:\n{content}"
            
    except UnicodeDecodeError:
        return f"ERROR: File '{file_path}' is not UTF-8 text-readable. It may be a binary file."
    except PermissionError:
        return f"ERROR: Permission denied when trying to read '{file_path}'."
    except Exception as e:
        return f"ERROR reading file '{file_path}': {type(e).__name__}: {e}"

def _write_file_logic(file_path: str, content: str) -> str:
    """Internal logic for writing/overwriting a file with verification."""
    content_size = len(content)
    try:
        if os.path.isdir(file_path):
            return f"ERROR writing to file: '{file_path}' is an existing directory, not a file."

        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        if not os.path.exists(file_path):
            return f"ERROR writing to file '{file_path}': Write operation succeeded internally, but the file was NOT found on the filesystem during verification."
        
        verified_size = os.path.getsize(file_path)
        if verified_size == 0 and content_size > 0:
            return f"WARNING writing to file '{file_path}': File was created, but size is 0 bytes. Expected {content_size} characters. Review content or permissions."

        return (
            f"SUCCESS: File '{file_path}' successfully CREATED/OVERWRITTEN. "
            f"Verified size: {verified_size} bytes (Expected: {content_size} chars)."
        )
    except Exception as e:
        return f"ERROR writing to file '{file_path}': {type(e).__name__}: {e}"

def _delete_file_logic(path: str) -> str:
    """Internal logic for deleting a file or directory with verification."""
    is_dir = os.path.isdir(path)
    is_file = os.path.isfile(path)

    if not (is_dir or is_file):
        return f"WARNING: No file or directory found at path: '{path}'. Nothing was deleted."

    object_type = "directory" if is_dir else "file"

    try:
        if is_dir:
            shutil.rmtree(path)
        elif is_file:
            os.remove(path)
        
        if os.path.exists(path):
            return f"ERROR deleting {object_type} '{path}': Deletion reported success, but the {object_type} still EXISTS on the filesystem during verification."
            
        return f"SUCCESS: Successfully deleted {object_type}: '{path}'. Verified ABSENT on filesystem."
    
    except Exception as e:
        return f"ERROR deleting '{path}': {type(e).__name__}: {e}"

def _append_file_logic(file_path: str, content: str) -> str:
    """Internal logic for appending content to an existing file."""
    try:
        if not os.path.exists(file_path):
             return f"ERROR: File not found at '{file_path}'. Use 'write' operation to create it first."
        if os.path.isdir(file_path):
            return f"ERROR: Path '{file_path}' is a directory. Cannot append."
            
        before_size = os.path.getsize(file_path)
        
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
            
        after_size = os.path.getsize(file_path)
        appended_size = after_size - before_size
        
        return f"SUCCESS: Appended {len(content)} characters (verified {appended_size} bytes) to '{file_path}'. New size: {after_size} bytes."
        
    except PermissionError:
        return f"ERROR: Permission denied when appending to '{file_path}'."
    except Exception as e:
        return f"ERROR appending to file '{file_path}': {type(e).__name__}: {e}"

def _move_file_logic(src_path: str, dest_path: str) -> str:
    """Internal logic for moving or renaming a file/directory with verification."""
    try:
        if not os.path.exists(src_path):
            return f"ERROR: Source path '{src_path}' does not exist. Nothing to move."

        dest_exists = os.path.exists(dest_path)
        dest_is_dir = os.path.isdir(dest_path)
        src_basename = os.path.basename(src_path)
        
        if dest_is_dir:
            final_dest_path = os.path.join(dest_path, src_basename)
        else:
            final_dest_path = dest_path
            
        final_dest_existed = os.path.exists(final_dest_path)

        shutil.move(src_path, dest_path)
        
        if os.path.exists(src_path):
            return f"ERROR: Move failed. Source path '{src_path}' still exists after operation."
        if not os.path.exists(final_dest_path):
            return f"ERROR: Move failed. Destination path '{final_dest_path}' was not found after operation."

        if dest_is_dir:
            msg = f"SUCCESS: Moved '{src_path}' INTO directory '{dest_path}'."
            if final_dest_existed:
                msg += " (Overwrote existing item inside directory)"
        elif dest_exists:
            msg = f"SUCCESS: Moved '{src_path}' to '{dest_path}', OVERWRITING the existing file."
        else:
            msg = f"SUCCESS: Moved/Renamed '{src_path}' to '{dest_path}'."
        
        return msg
        
    except Exception as e:
        return f"ERROR moving file from '{src_path}' to '{dest_path}': {type(e).__name__}: {e}"


def _copy_file_logic(src_path: str, dest_path: str) -> str:
    """Internal logic for copying a file with verification."""
    try:
        if not os.path.exists(src_path):
            return f"ERROR: Source path '{src_path}' does not exist. Nothing to copy."
        if os.path.isdir(src_path):
            return f"ERROR: Source path '{src_path}' is a directory. This operation only copies single files."
        if not os.path.isfile(src_path):
            return f"ERROR: Source path '{src_path}' is not a file."

        dest_exists = os.path.exists(dest_path)
        dest_is_dir = os.path.isdir(dest_path)
        src_basename = os.path.basename(src_path)
        
        if dest_is_dir:
            final_dest_path = os.path.join(dest_path, src_basename)
        else:
            final_dest_path = dest_path
            
        final_dest_existed = os.path.exists(final_dest_path)

        shutil.copy2(src_path, dest_path)
        
        if not os.path.exists(src_path):
            return f"ERROR: Source file '{src_path}' is missing after copy. This should not happen."
        if not os.path.exists(final_dest_path):
            return f"ERROR: Copy failed. Destination file '{final_dest_path}' was not found after operation."

        if dest_is_dir:
            msg = f"SUCCESS: Copied '{src_path}' INTO directory '{dest_path}'."
            if final_dest_existed:
                msg += " (Overwrote existing file inside directory)"
        elif dest_exists:
            msg = f"SUCCESS: Copied '{src_path}' to '{dest_path}', OVERWRITING the existing file."
        else:
            msg = f"SUCCESS: Copied '{src_path}' to '{dest_path}'."
        
        return msg
        
    except Exception as e:
        return f"ERROR copying file '{src_path}' to '{dest_path}': {type(e).__name__}: {e}"

def _list_dir_logic(directory_path: str, extensions: List[str] = None) -> str:
    """Internal logic for listing a directory's contents with filtering and truncation."""
    try:
        if not os.path.exists(directory_path):
            return f"ERROR: Directory not found at '{directory_path}'."
        if not os.path.isdir(directory_path):
            return f"ERROR: Path '{directory_path}' is a file, not a directory. Use 'read' operation."

        directories = []
        files = []
        
        with os.scandir(directory_path) as it:
            for entry in it:
                if len(directories) + len(files) >= MAX_LIST_COUNT:
                    if entry.is_dir():
                        directories.append(f"... [TRUNCATED at {MAX_LIST_COUNT} items]")
                    else:
                        files.append(f"... [TRUNCATED at {MAX_LIST_COUNT} items]")
                    break

                if entry.is_dir():
                    directories.append(entry.name)
                elif entry.is_file():
                    if extensions:
                        if any(entry.name.endswith(ext) for ext in extensions):
                            files.append(entry.name)
                    else:
                        files.append(entry.name)
        
        result = {"directories": directories, "files": files}
        return f"SUCCESS: Contents of '{directory_path}': {json.dumps(result)}"

    except PermissionError:
        return f"ERROR: Permission denied when listing directory '{directory_path}'."
    except Exception as e:
        return f"ERROR listing directory '{directory_path}': {type(e).__name__}: {e}"

@tool
def file_manager(operation: str, args: Dict[str, Any]) -> str:
    """
    Manages all file system operations (read, write, delete, append, move, copy, list).
    
    Args:
        operation: The file operation to perform. Must be one of: 
                   'read', 'write', 'delete', 'append', 'move', 'copy', 'list'.
        args: A dictionary of arguments for the operation.
              - 'read': {'file_path': str}
              - 'write': {'file_path': str, 'content': str}
              - 'delete': {'path': str}
              - 'append': {'file_path': str, 'content': str}
              - 'move': {'src_path': str, 'dest_path': str}
              - 'copy': {'src_path': str, 'dest_path': str}
              - 'list': {'directory_path': str, 'extensions': List[str] (optional)}
    """
    op = operation.lower().strip()
    
    try:
        if op == 'read':
            return _read_file_logic(**args)
        elif op == 'write':
            return _write_file_logic(**args)
        elif op == 'delete':
            return _delete_file_logic(**args)
        elif op == 'append':
            return _append_file_logic(**args)
        elif op == 'move':
            return _move_file_logic(**args)
        elif op == 'copy':
            return _copy_file_logic(**args)
        elif op == 'list':
            # Handle optional 'extensions' argument for list operation
            if 'extensions' not in args:
                args['extensions'] = None
            return _list_dir_logic(**args)
        else:
            return f"ERROR: Invalid operation '{operation}'. Must be one of: read, write, delete, append, move, copy, list."
    
    except TypeError as e:
        return f"ERROR: Missing or incorrect arguments for operation '{operation}'. Details: {e}. Check the required 'args' dictionary in the tool description."
    except Exception as e:
        return f"ERROR: An unexpected error occurred in the file_manager dispatch for '{operation}': {e}"


async def finish_logic(input: str) -> str:
    """This tool is the final action to return the answer."""
    return input

finish = Tool(
    name="finish",
    description="The FINAL action to take. Use this when you have the complete, final answer. The input should be the final answer.",
    func=finish_logic,
    coroutine=finish_logic
)

# -------------------------------
# 1. Search Tool (Robust)
# -------------------------------
@tool
def google_search(query: str) -> str:
    """
    Searches the web using the Google Custom Search API and returns a string
    of search results (snippets, source titles, and URLs).
    
    Args:
        query: The query to search for.
    """
    # Check for required environment variables
    if "GOOGLE_CSE_ID" not in os.environ:
        return "ERROR: The 'GOOGLE_CSE_ID' environment variable is not set. This tool cannot function."
    if "GOOGLE_API_KEY" not in os.environ:
        return "ERROR: The 'GOOGLE_API_KEY' environment variable is not set. This tool cannot function."

    try:
        # Note: By default, this wrapper uses the GOOGLE_API_KEY and GOOGLE_CSE_ID
        # environment variables, so we don't need to pass them in.
        print("Google searching...")
        search_wrapper = GoogleSearchAPIWrapper(k=5) # Get top 5 results
        
        # .results() returns a list of dictionaries
        # .run() returns a formatted string, which is better for an LLM Observation
        results = search_wrapper.run(query) 
        
        if not results:
            print("Google Search Results Not Found!")
            return f"No results found for query: '{query}'"
        print("Google Search Successfull!")
        return f"Search results for '{query}':\n{results}"
        
    except Exception as e:
        print("Google Search Error!")

        return f"ERROR running search for '{query}': {type(e).__name__}: {e}"
    
@tool
def search(query:str):
    """Grounding with Google Search connects the Gemini model to real-time web content and works with all available languages. This allows Gemini to provide more accurate answers and cite verifiable sources beyond its knowledge cutoff.
    Args:
        query: The query to search for.
    """
    client = genai.Client()

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config,
    )
    results = response.text
    print(response)
    return(results)
