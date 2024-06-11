from dataclasses import fields

def filter_unexpected_fields(cls):
    """
    Decorator function that filters unexpected fields from the keyword arguments passed to the class constructor.

    Args:
        cls: The class to decorate.

    Returns:
        The decorated class.

    Example:
        @filter_unexpected_fields
        class MyClass:
            def __init__(self, name, age):
                self.name = name
                self.age = age

        obj = MyClass(name='John', age=25, gender='Male')
        # The 'gender' keyword argument will be filtered out and not passed to the class constructor.
    """
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        expected_fields = {field.name for field in fields(cls)}
        cleaned_kwargs = {key: value for key, value in kwargs.items() if key in expected_fields}
        original_init(self, *args, **cleaned_kwargs)

    cls.__init__ = new_init
    return cls