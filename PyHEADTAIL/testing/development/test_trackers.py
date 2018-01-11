import inspect
import tracking_model as tm


A = tm.Octupoles(ap_x=1, ap_y=1, ap_xy=1)

A.ap_x

print A.ap_x
print inspect.getmembers(A, inspect.isgetsetdescriptor)
print inspect.getmembers(A, inspect.ismemberdescriptor)
print inspect.getmembers(A, inspect.ismethod)
print [a for a in dir(A)]
[method for method in dir(A) if callable(getattr(A, method))]
