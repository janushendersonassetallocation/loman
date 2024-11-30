# Interactive Debugging

As shown in the quickstart section "Error-handling", loman makes it easy to see a traceback for any exceptions that are shown while calculating nodes, and also makes it easy to update calculation functions in-place to fix errors. However, it is often desirable to use Python's interactive debugger at the exact time that an error occurs. To support this, the `calculate` method takes a parameter `raise_exceptions`. When it is `False` (the default), nodes are set to state ERROR when exceptions occur during their calculation. When it is set to `True` any exceptions are not caught, allowing the user to invoke the interactive debugger

```pycon
comp = Computation()
comp.add_node('numerator', value=1)
comp.add_node('divisor', value=0)
comp.add_node('result', lambda numerator, divisor: numerator / divisor)
comp.compute('result', raise_exceptions=True)
```

```pycon
    ---------------------------------------------------------------------------
    
    ZeroDivisionError                         Traceback (most recent call last)
    
    <ipython-input-38-4243c7243fc5> in <module>()
    ----> 1 comp.compute('result', raise_exceptions=True)
    
    
    [... skipped ...]
    
    
    <ipython-input-36-c0efbf5b74f7> in <lambda>(numerator, divisor)
          3 comp.add_node('numerator', value=1)
          4 comp.add_node('divisor', value=0)
    ----> 5 comp.add_node('result', lambda numerator, divisor: numerator / divisor)
    
    
    ZeroDivisionError: division by zero
```

```pycon
    %debug

    > <ipython-input-36-c0efbf5b74f7>(5)<lambda>()
          1 from loman import *
          2 comp = Computation()
          3 comp.add_node('numerator', value=1)
          4 comp.add_node('divisor', value=0)
    ----> 5 comp.add_node('result', lambda numerator, divisor: numerator / divisor)

    ipdb> p numerator
    1
    ipdb> p divisor
    0
```

