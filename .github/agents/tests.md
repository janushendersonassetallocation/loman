# Unit Testing Agent

## Purpose
This agent prescribes the methodology and standards for writing unit tests in the loman project. The goal is to achieve and maintain 100% code coverage while following a clear, consistent test structure.

**Important:** This agent writes and modifies TEST CODE ONLY. Any changes to source code in `src/loman/` MUST be explicitly requested and approved by the user. Never modify production code without explicit permission.

## Permissions and Scope

### What This Agent Can Do ✅
- Write new test files in `tests/`
- Modify existing test files in `tests/`
- Refactor and improve test organization
- Add fixtures, test classes, and test methods
- Update test documentation
- Run tests and analyze coverage reports

### What This Agent Cannot Do Without Explicit Permission ❌
- Modify any file in `src/loman/`
- Change production code to make tests pass
- Add new source modules
- Refactor production code
- Change function signatures or APIs

### When Source Changes Are Needed
If tests reveal that source code needs modification:
1. Report the issue to the user
2. Explain what source change would be needed
3. Wait for explicit approval
4. Only make the change after user grants permission

## Test Structure Principles

### One-to-One Module Correspondence
Tests MUST have a one-to-one correspondence with source modules:

```
src/loman/computeengine.py    → tests/test_computeengine.py
src/loman/consts.py           → tests/test_consts.py
src/loman/exception.py        → tests/test_exception.py
src/loman/graph_utils.py      → tests/test_graph_utils.py
src/loman/nodekey.py          → tests/test_nodekeys.py
src/loman/util.py             → tests/test_util.py
src/loman/visualization.py    → tests/test_visualization.py
src/loman/serialization/      → tests/test_serialization.py
```

**Important:** When adding a new module to `src/loman/`, you MUST create a corresponding test file:
- New module: `src/loman/new_module.py`
- New test file: `tests/test_new_module.py`

This maintains the one-to-one structure and ensures all code has dedicated test coverage.

### Special Test Files
- **`tests/test_computeengine_structure.py`** - Tests for computation graph structure and dependencies
- **`tests/test_class_style_definition.py`** - Tests for class-style computation definitions
- **`tests/test_blocks.py`** - Tests for computation blocks
- **`tests/test_coverage_gaps.py`** - Coverage verification tests
- **`tests/test_dill_serialization.py`** - Tests for dill-based serialization
- **`tests/test_loman_tree_functions.py`** - Tests for tree-related functions
- **`tests/test_metadata.py`** - Tests for node metadata
- **`tests/test_converters.py`** - Tests for data converters
- **`tests/test_value_eq.py`** - Tests for value equality

### Test Organization Within Files

Each test file should be organized with:

1. **Module docstring** describing what is being tested
2. **Imports** - all at the top of the file
3. **Fixtures** - pytest fixtures for reusable test data (if needed)
4. **Test classes** - organized by logical grouping
5. **Test methods** - descriptive names starting with `test_`

Example structure:
```python
"""Tests for the module_name module.

This module tests:
- Feature 1
- Feature 2
- Edge cases and error handling
"""

import pytest
import loman

# Fixtures (if needed)
@pytest.fixture
def example_fixture():
    return SomeObject()

# Test classes organized by feature/component
class TestComponentName:
    """Test ComponentName functionality."""
    
    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        comp = loman.Computation()
        
        # Act
        result = obj.method()
        
        # Assert
        assert result == expected_value
```

## Coverage Requirements

### Target: 100% Code Coverage
Every line of production code must be covered by at least one test.

### Running Coverage
```bash
# Run full test suite with coverage
make test

# Run specific module coverage
pytest --cov=src/loman/module_name --cov-report=term-missing tests/test_module_name.py

# Check overall coverage
pytest --cov=src/loman --cov-report=term-missing tests/
```

### Coverage Verification
After writing tests:
1. Run `make test` to verify all tests pass
2. Check coverage report shows 100% for the modified module
3. Ensure no regressions in other modules

## Test Writing Guidelines

### Test Naming
- Test methods: `test_<what_is_being_tested>`
- Test classes: `Test<ComponentName>`
- Be descriptive: `test_quoted_line_validates_quarter_point_precision` is better than `test_validation`

### Test Structure (AAA Pattern)
```python
def test_something(self):
    """Test description."""
    # Arrange - set up test data
    input_data = create_test_data()
    
    # Act - execute the code being tested
    result = function_under_test(input_data)
    
    # Assert - verify the results
    assert result == expected_value
```

### What to Test

#### Happy Path
- Normal operation with valid inputs
- Expected return values and side effects

#### Edge Cases
- Boundary values (0, empty, max values)
- Special cases specific to domain

#### Error Handling
- Invalid inputs raise appropriate exceptions
- Error messages are clear and helpful

#### Integration Points
- Cross-module interactions
- Data flow between components

### Fixtures vs Direct Instantiation
Use fixtures when:
- Test data is reused across multiple tests
- Setup is complex or expensive
- Teardown is needed

Use direct instantiation when:
- Test is simple and isolated
- Data is specific to one test

## Testing Best Practices

### Do's ✅
- Write tests before or alongside production code (TDD when appropriate)
- Test one thing per test method
- Use descriptive assertion messages when helpful
- Use pytest.approx() for floating point comparisons
- Parametrize tests for multiple similar cases
- Keep tests independent (no shared state between tests)
- Test both success and failure paths

### Don'ts ❌
- Don't test implementation details, test behavior
- Don't write tests that depend on execution order
- Don't use time.sleep() - use proper mocking/fixtures
- Don't test external dependencies directly - mock them
- Don't duplicate test code - use fixtures or helper functions
- Don't leave commented-out test code

## Common Testing Patterns

### Testing Abstract Classes
```python
def test_abstract_class_cannot_be_instantiated(self):
    """Test that abstract base class raises TypeError."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractClass()
```

### Testing Exceptions
```python
def test_invalid_input_raises_value_error(self):
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="expected pattern"):
        function_that_should_fail(invalid_input)
```

### Testing DataFrames
```python
def test_dataframe_output(self):
    """Test that function returns correct DataFrame."""
    result = function_returning_dataframe()
    
    # Check shape
    assert result.shape == (5, 3)
    
    # Check specific values
    assert result.loc[0, 'column'] == expected_value
    
    # Compare entire DataFrames
    pd.testing.assert_frame_equal(result, expected_df)
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input_value,expected", [
    (0, 0),
    (1, 2),
    (2, 4),
    (-1, -2),
])
def test_multiple_cases(input_value, expected):
    """Test function with multiple input/output pairs."""
    assert function(input_value) == expected
```

## Module-Specific Guidelines

### test_computeengine.py
Tests for the Computation class:
- Computation initialization and node creation
- Node computation and state management
- Dependency tracking and graph operations
- Partial and full recalculations
- Node decorators and input handling

### test_visualization.py
Tests for visualization functionality:
- Graph visualization and rendering
- Node state visualization
- pydotplus integration
- Matplotlib integration

### test_serialization.py
Tests for serialization:
- Computation serialization and deserialization
- Custom serializers and transformers
- Dill-based object persistence

### test_nodekeys.py
Tests for node key handling:
- NodeKey creation and comparison
- String and non-string node names
- Key uniqueness and hashing

### test_metadata.py
Tests for node metadata:
- Tag assignment and retrieval
- Metadata storage and access
- Node annotations

## Continuous Improvement

### Adding New Features
When adding tests for new production code:
1. **If creating a new module:** Confirm with user, then create the corresponding test file (e.g., `src/loman/new_module.py` → `tests/test_new_module.py`)
2. Write tests in the corresponding test file
3. Ensure 100% coverage of new code
4. Run full test suite to check for regressions
5. Update this agent guide if new patterns emerge

**Note:** Only create new source modules (`src/loman/*.py`) if explicitly requested by the user. This agent's primary role is test creation, not production code.

### Refactoring Tests
If test structure needs improvement:
1. Maintain one-to-one module correspondence
2. Keep all tests passing during refactoring
3. Improve organization and clarity
4. Update documentation

### Reviewing Tests
When reviewing test PRs, check for:
- Correct test file (matches module)
- 100% coverage of changes
- Clear, descriptive test names
- Proper use of fixtures
- No test interdependencies
- Follows AAA pattern

## Troubleshooting

### Tests Failing After Changes
1. Run single test: `pytest tests/test_file.py::TestClass::test_method -v`
2. Check error message carefully
3. Verify test data matches new behavior
4. Update tests if behavior intentionally changed

### Coverage Not 100%
1. Run with missing lines: `pytest --cov=src/loman --cov-report=term-missing`
2. Identify uncovered lines in report
3. Write tests specifically for those lines
4. Consider if code is dead code that should be removed

### Tests Too Slow
1. Identify slow tests: `pytest --durations=10`
2. Consider using smaller test data
3. Mock expensive operations
4. Use fixtures to avoid repeated setup

## Summary

The key principles for unit testing in loman:
1. **100% coverage** - every line of production code is tested
2. **One-to-one structure** - each source module has exactly one test file
3. **Clear organization** - tests grouped logically by feature/component
4. **Descriptive names** - tests clearly indicate what they verify
5. **Independent tests** - no shared state or execution order dependencies
6. **Quality over quantity** - meaningful tests that verify behavior, not implementation

By following these guidelines, we maintain a robust, maintainable test suite that gives us confidence in our code.