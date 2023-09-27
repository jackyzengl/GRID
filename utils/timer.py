import time

# class Timer():
#     def __init__(self) -> None:
#         self.start_time = time.time()
        
def get_time():
    return time.time()

def get_duration(start, end=None):
    if end:
        duration = end - start
    else:
        duration = get_time() - start
    return duration

def time_cost(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    time_cost = end_time - start_time
    # print(f"Time cost: {time_cost:.2f} seconds")
    # time_cost_forward = end_time - start_time if result is not None else 0
    # print(f"Time cost: {time_cost_forward:.2f} seconds")
    return *result, time_cost