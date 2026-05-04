"""
Tool executor — Parses and executes tool calls from LLM responses.

The LLM generates tool calls using XML tags, and this module
extracts and executes them. Supports: read_file, write_file,
edit_file, bash_command, search, find_files.
"""

import os
import re
import glob
import subprocess
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse XML tool calls from LLM response.

    Supported tags:
    - <read_file><path>...</path></read_file>
    - <write_file><path>...</path><content>...</content></write_file>
    - <edit_file><path>...</path><target>...</target><replacement>...</replacement></edit_file>
    - <bash_command><command>...</command></bash_command>
    - <search><pattern>...</pattern><path>...</path></search>
    - <find_files><pattern>...</pattern><path>...</path></find_files>

    Also supports legacy create_file and list_files tags.
    """
    tool_calls = []

    # --- write_file / create_file ---
    for tag in ('write_file', 'create_file'):
        for m in re.finditer(rf'<{tag}>\s*(.*?)\s*</{tag}>', text, re.DOTALL):
            body = m.group(1)
            path_m = re.search(r'<path>\s*(.*?)\s*</path>', body, re.DOTALL)
            content_m = re.search(r'<content>\s*(.*?)\s*</content>', body, re.DOTALL)
            if path_m and content_m:
                tool_calls.append({
                    'tool': 'write_file',
                    'args': [path_m.group(1).strip(), content_m.group(1)],
                })
            else:
                lines = body.split('\n', 1)
                if len(lines) >= 2:
                    fp = re.sub(r'</?path>', '', lines[0]).strip()
                    fc = re.sub(r'^<content>\s*', '', lines[1])
                    fc = re.sub(r'\s*</content>$', '', fc)
                    tool_calls.append({'tool': 'write_file', 'args': [fp, fc]})

    # --- read_file ---
    for m in re.finditer(r'<read_file>\s*(.*?)\s*</read_file>', text, re.DOTALL):
        body = m.group(1)
        path_m = re.search(r'<path>\s*(.*?)\s*</path>', body, re.DOTALL)
        fp = path_m.group(1).strip() if path_m else body.strip()
        fp = re.sub(r'</?path>', '', fp).strip()
        tool_calls.append({'tool': 'read_file', 'args': [fp]})

    # --- edit_file ---
    for m in re.finditer(r'<edit_file>\s*(.*?)\s*</edit_file>', text, re.DOTALL):
        body = m.group(1)
        path_m = re.search(r'<path>\s*(.*?)\s*</path>', body, re.DOTALL)
        target_m = re.search(r'<target>\s*(.*?)\s*</target>', body, re.DOTALL)
        repl_m = re.search(r'<replacement>\s*(.*?)\s*</replacement>', body, re.DOTALL)
        diff_m = re.search(r'<diff>\s*(.*?)\s*</diff>', body, re.DOTALL)

        if path_m and target_m and repl_m:
            tool_calls.append({
                'tool': 'edit_file',
                'args': [path_m.group(1).strip(), target_m.group(1), repl_m.group(1)],
            })
        elif path_m and diff_m:
            tool_calls.append({
                'tool': 'edit_file_diff',
                'args': [path_m.group(1).strip(), diff_m.group(1)],
            })

    # --- bash_command ---
    for m in re.finditer(r'<bash_command>\s*(.*?)\s*</bash_command>', text, re.DOTALL):
        body = m.group(1)
        cmd_m = re.search(r'<command>\s*(.*?)\s*</command>', body, re.DOTALL)
        cmd = cmd_m.group(1).strip() if cmd_m else body.strip()
        tool_calls.append({'tool': 'bash_command', 'args': [cmd]})

    # --- search ---
    for m in re.finditer(r'<search>\s*(.*?)\s*</search>', text, re.DOTALL):
        body = m.group(1)
        pat_m = re.search(r'<pattern>\s*(.*?)\s*</pattern>', body, re.DOTALL)
        path_m = re.search(r'<path>\s*(.*?)\s*</path>', body, re.DOTALL)
        if pat_m:
            args = [pat_m.group(1).strip()]
            if path_m:
                args.append(path_m.group(1).strip())
            tool_calls.append({'tool': 'search', 'args': args})

    # --- find_files ---
    for m in re.finditer(r'<find_files>\s*(.*?)\s*</find_files>', text, re.DOTALL):
        body = m.group(1)
        pat_m = re.search(r'<pattern>\s*(.*?)\s*</pattern>', body, re.DOTALL)
        path_m = re.search(r'<path>\s*(.*?)\s*</path>', body, re.DOTALL)
        if pat_m:
            args = [pat_m.group(1).strip()]
            if path_m:
                args.append(path_m.group(1).strip())
            tool_calls.append({'tool': 'find_files', 'args': args})

    # --- list_files (legacy) ---
    for m in re.finditer(r'<list_files>\s*(.*?)\s*</list_files>', text, re.DOTALL):
        body = m.group(1)
        lines = body.strip().split('\n', 1)
        args = [lines[0].strip()]
        if len(lines) > 1:
            args.append(lines[1].strip())
        tool_calls.append({'tool': 'find_files', 'args': args})

    return tool_calls


def execute_tool_call(
    tool_name: str,
    args: List[str],
    stream_callback: Optional[Callable] = None,
    working_dir: str = ".",
) -> Optional[Dict[str, Any]]:
    """Execute a single tool call.

    Returns a dict with keys: tool, target, result, exit_code (for bash).
    """
    try:
        if tool_name == "write_file":
            if len(args) < 2:
                return {'tool': 'write_file', 'target': '?', 'result': 'Error: write_file requires path and content'}
            filepath, content = args[0], args[1]
            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)
            parent = os.path.dirname(filepath)
            if parent:
                os.makedirs(parent, exist_ok=True)

            if stream_callback:
                stream_callback('start', filepath, None)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            if stream_callback:
                stream_callback('finish', filepath, None)

            num_lines = content.count('\n') + 1
            return {
                'tool': 'write_file',
                'target': filepath,
                'result': f"Wrote {num_lines} lines to {filepath}",
                'content': content,
                'num_lines': num_lines,
            }

        elif tool_name == "read_file":
            if len(args) < 1:
                return {'tool': 'read_file', 'target': '?', 'result': 'Error: read_file requires path'}
            filepath = args[0]
            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)

            if not os.path.exists(filepath):
                return {'tool': 'read_file', 'target': filepath, 'result': f'Error: File not found: {filepath}'}

            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()

            num_lines = file_content.count('\n') + 1
            return {
                'tool': 'read_file',
                'target': filepath,
                'result': file_content,
                'num_lines': num_lines,
            }

        elif tool_name == "edit_file":
            if len(args) < 3:
                return {'tool': 'edit_file', 'target': '?', 'result': 'Error: edit_file requires path, target, replacement'}
            filepath, target, replacement = args[0], args[1], args[2]
            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)

            if not os.path.exists(filepath):
                return {'tool': 'edit_file', 'target': filepath, 'result': f'Error: File not found: {filepath}'}

            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                original = f.read()

            if target not in original:
                return {'tool': 'edit_file', 'target': filepath, 'result': f'Error: Target block not found in {filepath}'}

            updated = original.replace(target, replacement, 1)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated)

            additions = replacement.count('\n') + 1
            deletions = target.count('\n') + 1
            return {
                'tool': 'edit_file',
                'target': filepath,
                'result': f"Edited {filepath}: +{additions} -{deletions}",
                'additions': additions,
                'deletions': deletions,
                'diff_target': target,
                'diff_replacement': replacement,
            }

        elif tool_name == "edit_file_diff":
            if len(args) < 2:
                return {'tool': 'edit_file', 'target': '?', 'result': 'Error: edit_file_diff requires path and diff'}
            filepath, diff_text = args[0], args[1]
            if not os.path.isabs(filepath):
                filepath = os.path.join(working_dir, filepath)
            return {'tool': 'edit_file', 'target': filepath, 'result': f'Diff-based edit not yet supported for {filepath}'}

        elif tool_name == "bash_command":
            if len(args) < 1:
                return {'tool': 'bash_command', 'target': '?', 'result': 'Error: bash_command requires a command'}
            command = args[0]
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=working_dir,
                )
                output = proc.stdout
                if proc.stderr:
                    output += proc.stderr
                output = output.strip()
                if len(output) > 5000:
                    output = output[:5000] + "\n... (truncated)"
                return {
                    'tool': 'bash_command',
                    'target': command,
                    'result': output or "(no output)",
                    'exit_code': proc.returncode,
                }
            except subprocess.TimeoutExpired:
                return {
                    'tool': 'bash_command',
                    'target': command,
                    'result': 'Error: Command timed out after 30s',
                    'exit_code': -1,
                }

        elif tool_name == "search":
            if len(args) < 1:
                return {'tool': 'search', 'target': '?', 'result': 'Error: search requires a pattern'}
            pattern = args[0]
            search_path = args[1] if len(args) > 1 else "."
            if not os.path.isabs(search_path):
                search_path = os.path.join(working_dir, search_path)

            try:
                proc = subprocess.run(
                    ['grep', '-rn', '--include=*.py', '--include=*.js', '--include=*.ts',
                     '--include=*.json', '--include=*.yaml', '--include=*.yml',
                     '--include=*.md', '--include=*.txt', '--include=*.html',
                     '--include=*.css', '--include=*.rs', '--include=*.go',
                     '--include=*.java', '--include=*.c', '--include=*.cpp',
                     '--include=*.h', '--include=*.sh',
                     pattern, search_path],
                    capture_output=True, text=True, timeout=15,
                )
                output = proc.stdout.strip()
                lines = output.split('\n') if output else []
                num_matches = len(lines)
                files_set = set()
                for line in lines:
                    if ':' in line:
                        files_set.add(line.split(':')[0])

                if num_matches > 30:
                    display = '\n'.join(lines[:30]) + f"\n... (+{num_matches - 30} more matches)"
                else:
                    display = output or "(no matches)"

                return {
                    'tool': 'search',
                    'target': pattern,
                    'result': display,
                    'num_matches': num_matches,
                    'num_files': len(files_set),
                }
            except subprocess.TimeoutExpired:
                return {'tool': 'search', 'target': pattern, 'result': 'Error: Search timed out'}

        elif tool_name == "find_files":
            if len(args) < 1:
                return {'tool': 'find_files', 'target': '?', 'result': 'Error: find_files requires a pattern'}
            pattern = args[0]
            search_path = args[1] if len(args) > 1 else "."
            if not os.path.isabs(search_path):
                search_path = os.path.join(working_dir, search_path)

            matches = []
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__', '.nare_memory', 'venv', '.venv')]
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, search_path)
                    if glob.fnmatch.fnmatch(f, pattern) or glob.fnmatch.fnmatch(rel, pattern):
                        matches.append(rel)

            display = '\n'.join(matches[:50])
            if len(matches) > 50:
                display += f"\n... (+{len(matches) - 50} more files)"
            if not matches:
                display = "(no files found)"

            return {
                'tool': 'find_files',
                'target': pattern,
                'result': display,
                'num_files': len(matches),
            }

        else:
            return {'tool': tool_name, 'target': '?', 'result': f'Error: Unknown tool {tool_name}'}

    except Exception as e:
        logging.error(f"Tool execution failed: {e}")
        return {'tool': tool_name, 'target': str(args[0]) if args else '?', 'result': f'Error: {e}'}


def execute_tools_from_response(
    response: str,
    stream_callback: Optional[Callable] = None,
    working_dir: str = ".",
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Parse and execute all tool calls from LLM response.

    Returns:
        Tuple of (cleaned_response, modified_files, all_results)
    """
    tool_calls = parse_tool_calls(response)
    results = []
    modified_files = []

    for call in tool_calls:
        result = execute_tool_call(call['tool'], call['args'], stream_callback, working_dir)
        if result:
            results.append(result)
            logging.info(f"Executed {call['tool']}: {result.get('result', '')[:100]}")
            if call['tool'] in ('write_file', 'edit_file') and len(call['args']) > 0:
                modified_files.append(call['args'][0])

    # Clean XML tags from response
    cleaned = response
    for tag in ('write_file', 'create_file', 'read_file', 'edit_file',
                'bash_command', 'search', 'find_files', 'list_files'):
        cleaned = re.sub(rf'<{tag}>.*?</{tag}>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    return cleaned, modified_files, results


class ToolExecutor:
    """Parse and execute XML tool calls from LLM responses."""

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir

    def parse_and_execute(self, response: str) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """Parse XML tags from response and execute actions.

        Returns:
            Tuple of (cleaned_response, modified_files, all_results)
        """
        return execute_tools_from_response(response, working_dir=self.working_dir)
