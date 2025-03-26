class A:
    def __init__(self, a: 'int' | None = None):
        self.a = a

    def __str__(self):
        return f"A({self.a})"
    
if __name__ == "__main__":
    a = A()
    print(a)
