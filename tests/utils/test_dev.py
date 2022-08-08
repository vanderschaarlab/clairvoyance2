from clairvoyance2.utils.dev import function_is_notimplemented_stub


class TestFunctionIsNotImplementedStub:
    def test_is(self):
        def my_function():
            raise NotImplementedError("Not implemented!")

        assert function_is_notimplemented_stub(my_function) is True

    def test_is_not(self):
        def my_function(a):
            if a == 10:
                raise NotImplementedError("Didn't implement this.")
            else:
                return 123

        assert function_is_notimplemented_stub(my_function) is False
