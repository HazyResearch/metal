import copy


def create_slice_task(base_task, slice_task_name):
    """Creates a slice task identical to a base task but with different head params"""
    slice_task = copy.copy(base_task)
    slice_task.name = slice_task_name
    slice_task.head_module = copy.deepcopy(base_task.head_module)
    return slice_task
