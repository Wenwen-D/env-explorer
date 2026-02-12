import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.data.data_types import TaskInstance

logger = logging.getLogger(__name__)

import json
from pathlib import Path
from typing import List, Optional, Union

from src.data.data_types import TaskInstance


class TaskLoader:    
    def load_tasks(self, 
                   metadata_file: Union[str, Path], 
                   limit: Optional[int] = None,
                   base_dir: Optional[Union[str, Path]] = None) -> List[TaskInstance]:
        """
        Load tasks from a JSON metadata file.
        
        Args:
            metadata_file: Path to the task metadata JSON file
            limit: Maximum number of tasks to load (None for all)
            
        Returns:
            List of TaskInstance objects
        """
        with open(metadata_file, 'r') as f:
            task_metadata = json.load(f)
        
        if base_dir is None:
            base_dir = Path(metadata_file).parent.parent
        # Apply limit if specified
        if limit is not None:
            task_metadata = task_metadata[:limit]
        
        # Convert to TaskInstance objects
        tasks = []
        for i, task_info in enumerate(task_metadata):
            # Get task_id, use index as fallback
            task_id = task_info.get('task_id', i)
            
            # Get ground truth path (handle different field names)
            # gt_path = task_info.get('gt') or task_info.get('gt_path')
            # gt_path = join(base_dir, gt_path) if gt_path else None
            gt_path = task_info.get('gt') or task_info.get('gt_path')
            if gt_path:
                gt_path = Path(base_dir) / gt_path if isinstance(base_dir, str) else base_dir / gt_path
            
            files = list(task_info.get('files',[]).keys())
            # Create metadata dict with all other fields
            metadata = {k: v for k, v in task_info.items() 
                       if k not in ['task_id', 'instruction', 'gt', 'gt_path']}
            
            # print(f"Loading task {task_id} with instruction: {task_info['instruction']}, gt_path: {gt_path}, files: {files}, metadata: {metadata}")
            task = TaskInstance(
                task_id=task_id,
                instruction=task_info['instruction'],
                gt_path=gt_path,
                files=files,
                metadata=metadata
            )
            tasks.append(task)
        
        return tasks