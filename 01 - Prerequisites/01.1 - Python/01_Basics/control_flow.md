# Control Flow & Comprehensions

## Conditionals
```python
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

# ternary
result = "even" if x % 2 == 0 else "odd"
```

## Loops
```python
for i in range(10): ...
for idx, val in enumerate(items): ...
for k, v in dict.items(): ...
for a, b in zip(list1, list2): ...

while condition:
    ...
```

## Comprehensions
```python
# list
[x**2 for x in range(10) if x % 2 == 0]

# dict
{x: x**2 for x in range(5)}

# set
{x % 3 for x in range(10)}

# generator (lazy)
sum(x**2 for x in range(1000000))
```

## Walrus Operator `:=`
```python
if (n := len(items)) > 10:
    print(f"Large batch: {n}")
```
