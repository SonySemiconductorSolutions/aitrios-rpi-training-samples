import gc

class EmptyClass:
    pass

def clear_memory():
    gc.collect()
