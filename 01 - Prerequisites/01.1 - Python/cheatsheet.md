# Python Quick Reference for AI

## Comprehensions
```python
[x**2 for x in range(10) if x % 2 == 0]
{x: x**2 for x in range(5)}
```

## Lambda & Map
```python
f = lambda x: x**2
list(map(lambda x: x*2, [1,2,3]))
```

## Decorators
```python
from functools import wraps
def timer(f):
    @wraps(f)
    def wrapper(*a, **kw):
        import time; s=time.time(); r=f(*a,**kw)
        print(f"{f.__name__}: {time.time()-s:.3f}s"); return r
    return wrapper
```

## NumPy
```python
np.zeros((3,4)); np.ones((2,3)); np.arange(10)
a.reshape(2,5); a @ b; np.dot(a,b); np.concatenate([a,b])
```

## Pandas
```python
df = pd.read_csv("file.csv")
df.groupby("col").agg({"val": ["mean","std"]})
pd.merge(df1, df2, on="key", how="left")
```

## PyTorch
```python
x = torch.randn(32, 3, 224, 224)
model = nn.Linear(10, 2).cuda()
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## FastAPI
```python
@app.post("/predict")
def predict(data: InputSchema) -> dict:
    result = model.predict(data.features)
    return {"prediction": result.tolist()}
```

## Pydantic
```python
class Config(BaseSettings):
    api_key: str = Field(..., env="API_KEY")
    model_name: str = "gpt-4"
```
