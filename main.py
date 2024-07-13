import inspect

def print_hi(name, a: int):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

signature = inspect.signature(print_hi)
print(type(signature.parameters))
print(signature.parameters)