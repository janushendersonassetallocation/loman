from loman import ComputationFactory, input_node, calc_node


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
