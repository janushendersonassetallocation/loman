Introduction
============

Loman is a Python library for keeping track of dependencies between elements of a large computation, allowing you to recalculate only the parts that are necessary as new input data arrives, or as you change how certain elements are calculated.

It stems from experience with real-life systems taking data from many independent source. Often systems are implemented using sets of scheduled tasks. This approach is often pragmatic at first, but suffers several drawbacks as the scale of the system increases:

* When failures occur, such as a required file or data set not being in place on time, then downstream scheduled tasks may execute anyway.
* When re-runs are required, typically each step must be manually invoked. Often it is not clear which steps must be re-run, and so operators re-run everything until things look right. A large proportion of the operational overhead of many real-world systems comes from needing enough capacity to improvised re-runs when systems fail.
* As tasks are added, the schedule may be become tight. It may not be clear which items can be moved earlier or later to make room for new tasks.

Other problems occur at the scale of single programs, which are often programmed as a sequential set of steps. Typically any reasonably complex computation will require multiple iterations before it is correct. A limiting factor is the speed at which the programmer can perform these iterations - there are only so many minutes in each day. Often repeatedly pulling large data sets or re-performing lengthy calculations that will not have changed between iterations ends up substantially slowing progress.

Loman aims to provide a solution to both these problems. Computations are represented explicitly as a directed acyclic graph data structures. A graph is a set of nodes, each representing an input value calculated value, and a set of edges (lines) between them, where one value feeds into the calculation of another. This is similar to a flowchart, the calculation tree in Excel, or the dependency graph used in build tools such as make. Loman keeps track of the current state of each node as the user requests certain elements be calculated, inserts new data into input nodes of the graph, or even changes the functions used to perform calculations. This allows analysts, researchers and developers to iterate quickly, making changes to isolated parts of complicated calculations.

Loman can serialize the entire contents of a graph to disk. When failures occur in batch systems a serialized copy of its computations allows for easy inspection of the inputs and intermediates to determine what failed. Once the error is diagnosed, it can be fixed by inserting updated data if available, and only recalculating what was necessary. Or alternatively, input or intermediate data can be directly updated by the operator. In either case, diagnosing errors is as easy as it can be, and recovering from errors is efficient.

Finally, Loman also provides useful capability to real-time systems, where the cadence of inputs can vary widely between input sources, and the computational requirement for different outputs can also be quite different. In this context, Loman allows updates to fast-calculated outputs for every tick of incoming data, but may limit the rate at which slower calculated outputs are produced.

Hopefully this gives a flavor of the type of problem Loman is trying to solve, and whether it will be useful to you. Our aim is that if you are performing a computational task, Loman should be able to provide value to you, and should be as frictionless as possible to use.