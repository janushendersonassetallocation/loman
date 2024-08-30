# Strategies for using Loman in the Real World

## Fine-grained or Coarse-grained nodes

When using Loman, we have a choice of whether we make each expression in our program a node (very fine-grained), or have one node which executes all the code in our calculation (very coarse-grained) or somewhere in between. Loman is relatively efficient at executing code, but it can never be as efficient as the Python interpreter sequentially running lines of Python. Accordingly, we recommend that you should create a node for each input, result or intermediate value that you might care to inspect, alter, or calculate in a different way.

On the other hand, there is no cost to nodes that are not executed [^f1], and execution is effectively lazy if you specify which nodes you wish to calculate. With this in mind, it can make sense to create large numbers of nodes that import data for example, in the anticipation that it will be useful to have that data to hand at some point, but there is no cost if it is not needed.

[^f1]: This is not quite true. The method for working out computable nodes has not been optimized, so in fact there is linear cost to adding unused nodes, but this limitation is unnecessary and will be removed in due course.

## Converting existing codebases to use Loman

We typically find that production code tends to have a "main" function which loads data from databases and other systems, coordinates running a few calculations, and then loads the results back into other systems. We have had good experiences transferring the data downloading and calculation parts to Loman.

Typically such a "main" function will have small groups of lines responsible for grabbing particular pieces of input data, or for calling out to specific calculation routine. Each of these groups can be converted to a node in a Loman computation easily. Often, Loman's `kwds` input to `add_node` is useful to martial data into existing functions.

It is often helpful to put the creation of a Loman computation object with uninitialized values into a separate function. Then it is easy to experiment with the computation in an interactive environment such as Jupyter.

The final result is that the "main" function will instantiate a computation object, give it objects for database access, and other inputs, such as run date. Exporting calculated results is not within Loman's scope, so the "main" function will coordinate writing results from the computation object to the same systems as before. It is also useful if the "main" function serializes the computation for later inspection if necessary.

Already, having a concrete visualization of the computation's structure, as well as the ability to access intermediates of the computation through the serialized copy will be great steps ahead of the existing system. Experimenting with adding additional parts to the computation will also be easier, as blank or serialized computations can be used as the basis for this work in an interactive environment.

Finally, it is not necessary to "big bang" existing systems over to a Loman-based solution. Instead, small discrete parts of an existing implementation can be converted, and gradually migrate additional parts inside of the Loman computation as desirable.

## Accessing databases and other external systems

To access databases, we recommend [SQLAlchemy](http://www.sqlalchemy.org/) Core. For each database, we recommend creating two nodes in a computation, one for the engine, and another for the metadata object, and these nodes should not be serialized. Then every data access can use the nodes **engine** and **metadata** as necessary. This is not dissimilar to dependency injection:

```pycon
>>> import sqlalchemy as sa
>>> comp = Computation()
>>> comp.add_node('engine', sa.create_engine(...), serialize=False)
>>> comp.add_node('metadata', lambda engine: sa.MetaData(engine), serialize=False)
>>> def get_some_data(engine, ...):
...   [...]
...
>>> comp.add_node('some_data', get_some_data)
```

Accessing other data sources such as scraped websites, or vendor systems can be accessed similarly. For example, here is code to create a logged in browser under the control of Selenium to scrape data from a website:

```pycon
>>> from selenium import webdriver
>>> comp = Computation()
>>> def get_logged_in_browser():
...   browser = webdriver.Chrome()
...   browser.get('http://somewebsite.com')
...   elem = browser.find_element_by_id('userid')
...   elem.send_keys('user@id.com')
...   elem = browser.find_element_by_id('password')
...   elem.send_keys('secret')
...   elem = browser.find_element_by_name('_submit')
...   elem.click()
...   return browser
... comp.add_node('browser', get_logged_in_browser)
```
