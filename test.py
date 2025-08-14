from beartype import beartype

@beartype
def add(a:int,b:int) -> int:
    return a+b
print("qws")
a = 1
b = 2
print("Result:", add(a, b))  # This will call the beartyped function
