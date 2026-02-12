from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Tuple, Literal
from pathlib import Path

class TaskInstance(BaseModel):
    task_id: Union[int, str]
    instruction: str
    gt_path: Union[str, Path]  # Ground truth file path
    files: List[Union[str, Path]]  # List of file names involved in the task
    metadata: Dict[str, Any] = {}

class ConversationTurn(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    returncode: Union[int, None]
    success: Union[bool, None]
    error_type: Optional[str] = None

class TaskResult(BaseModel):
    task_id: Union[int, str]
    instruction: str
    final_code: Optional[str] = None
    has_error: Optional[bool] = None
    conversation: List[ConversationTurn]
    execution_results: List[ExecutionResult]
    num_turns: int
    success: bool
    turn_type: Optional[List[Literal['explore', 'draft', 'final', 'other']]] = None
    turn_accuracy: Optional[List[Optional[float]]] = None
