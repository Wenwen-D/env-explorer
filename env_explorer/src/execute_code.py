import subprocess
import tempfile
import os
from typing import Tuple, Optional
from src.data.data_types import ExecutionResult
import re

class CodeExtractor:
    @staticmethod
    def extract_code_from_response(response: str) -> Tuple[Optional[str], str]:
        """Extract code from model response - your existing logic"""
        # Check for explore tags first
        explore_matches = re.findall(r'<explore>(.*?)</explore>', response, re.DOTALL)
        for match in reversed(explore_matches):
            if match.strip():
                return CodeExtractor._clean_code(match), 'explore'
        
        draft_matches = re.findall(r'<draft>(.*?)</draft>', response, re.DOTALL)
        for match in reversed(draft_matches):
            if match.strip():
                return CodeExtractor._clean_code(match), 'draft'
        
        # Check for final tags
        final_matches = re.findall(r'<final>(.*?)</final>', response, re.DOTALL)
        for match in reversed(final_matches):
            if match.strip():
                return CodeExtractor._clean_code(match), 'final'
        
        # Check for code blocks
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip(), 'explore'
        
        return None, 'explore'
    
    @staticmethod
    def _clean_code(raw_code: str) -> str:
        """Clean extracted code by removing markdown fences"""
        code_match = re.search(r'```python\s*(.*?)\s*```', raw_code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return raw_code.strip()
    


class CodeExecutor:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def execute_code(self, code: str, target_dir: str) -> ExecutionResult:
        """Execute Python code in a subprocess"""
        if os.path.isfile(target_dir):
            target_dir = os.path.dirname(target_dir)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run([
                'python', temp_file
            ], 
            cwd=target_dir,
            capture_output=True, 
            text=True, 
            timeout=self.timeout
            )
            
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                success=result.returncode == 0,
                error_type=self._parse_error_type(result.stderr) if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="Execution timed out",
                returncode=-1,
                success=False,
                error_type="TimeoutError"
            )
        finally:
            os.unlink(temp_file)
    
    def _parse_error_type(self, stderr_output: str) -> str:
        """Extract error type from stderr - your existing logic"""
        if not stderr_output:
            return "UnknownError"
        
        import re
        error_patterns = [r'(\w*Error):', r'(\w*Exception):']
        
        lines = stderr_output.strip().split('\n')
        for line in reversed(lines):
            for pattern in error_patterns:
                match = re.search(pattern, line)
                if match:
                    error_name = match.group(1)
                    if error_name not in ['Error', 'Exception']:
                        return error_name
        
        if 'SyntaxError' in stderr_output:
            return 'SyntaxError'
        elif 'IndentationError' in stderr_output:
            return 'IndentationError'
            
        return 'UnknownError'


