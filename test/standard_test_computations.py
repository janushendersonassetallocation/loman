from loman import ComputationFactory, input_node, calc_node, Computation


@ComputationFactory
class BasicFourNodeComputation:
    a = input_node()

    @calc_node
    def b(a):
        return a + 1

    @calc_node
    def c(a):
        return 2 * a

    @calc_node
    def d(b, c):
        return b + c


def create_example_block_computation():
    comp_inner = BasicFourNodeComputation()
    comp_inner.insert('a', value=7)
    comp_inner.compute_all()
    comp = Computation()
    comp.add_block('foo', comp_inner, keep_values=False, links={'a': 'input_foo'})
    comp.add_block('bar', comp_inner, keep_values=False, links={'a': 'input_bar'})
    comp.add_node('output', lambda x, y: x + y, kwds={'x': 'foo/d', 'y': 'bar/d'})
    comp.add_node('input_foo', value=7)
    comp.add_node('input_bar', value=10)
    return comp