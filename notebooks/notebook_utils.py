import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

def add_project_path() -> bool:
    """
    Function for adding project cur_path to sys.cur_path
    """
    project_path = Path('.')
    cur_path = Path(project_path.absolute())
    for parent in cur_path.parents:
        if 'Pipfile' in [obj.name for obj in parent.glob('*')]:
            project_path = Path(parent.absolute())
            break

    src_path = project_path.joinpath('src')

    if project_path == '.':
        LOGGER.warning("Can't find project_path")
        return False

    if src_path not in sys.path:
        sys.path.append(str(src_path.absolute()))
    return project_path