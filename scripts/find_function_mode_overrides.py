"""Audit FUNCTIONS entries for monkey-patch corruption (issue #38).

For each entry, do setattr+restore and check whether any probe in
``probe_battery`` differs. For corrupting entries, check whether
``TorchFunctionMode`` can intercept the operation. Reports the unsolvable
intersection.
"""
import os
import sys
import torch
import torch.overrides

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvista.overrides import FUNCTIONS, NAMESPACE_TO_MODULE


def probe_battery():
    import copy
    import math
    import pickle
    import numpy as np
    F = torch.nn.functional

    probes = [
        # construction
        ('ctor_of_0d_list', lambda: repr(torch.tensor([torch.tensor(1.0)]))),
        ('ctor_of_0d_nested', lambda: repr(torch.tensor([[torch.tensor(1.0)], [torch.tensor(2.0)]]))),
        ('ctor_of_mixed_list', lambda: repr(torch.tensor([torch.tensor(1.0), 2.0, 3]))),
        ('ctor_of_1d_list', lambda: repr(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))),
        ('ctor_from_numpy', lambda: repr(torch.from_numpy(np.array([1.0, 2.0])))),
        ('ctor_empty_list', lambda: repr(torch.tensor([]))),
        ('as_tensor_0d_list', lambda: repr(torch.as_tensor([torch.tensor(1.0)]))),
        ('zeros', lambda: repr(torch.zeros(2, 3))),
        ('ones', lambda: repr(torch.ones(2))),
        ('arange', lambda: repr(torch.arange(0, 5))),
        ('linspace', lambda: repr(torch.linspace(0.0, 1.0, 4))),
        ('full', lambda: repr(torch.full((2, 2), 7.0))),

        # indexing / slicing / masking
        ('idx_int', lambda: repr(torch.tensor([1.0, 2.0])[0])),
        ('idx_neg', lambda: repr(torch.tensor([1.0, 2.0, 3.0])[-1])),
        ('idx_slice', lambda: repr(torch.tensor([1.0, 2.0, 3.0])[1:])),
        ('idx_2d', lambda: repr(torch.tensor([[1.0, 2.0], [3.0, 4.0]])[0, 1])),
        ('idx_ellipsis', lambda: repr(torch.tensor([[1.0, 2.0]])[..., 0])),
        ('idx_bool_mask', lambda: repr(torch.tensor([1.0, 2.0, 3.0])[torch.tensor([True, False, True])])),
        ('idx_tensor', lambda: repr(torch.tensor([10.0, 20.0, 30.0])[torch.tensor([0, 2])])),
        ('idx_set_int', lambda: (lambda t: (t.__setitem__(0, 99.0), repr(t))[1])(torch.tensor([1.0, 2.0]))),
        ('idx_set_slice', lambda: (lambda t: (t.__setitem__(slice(0, 2), 0.0), repr(t))[1])(torch.tensor([1.0, 2.0, 3.0]))),
        ('idx_set_mask', lambda: (lambda t: (t.__setitem__(torch.tensor([True, False]), 0.0), repr(t))[1])(torch.tensor([1.0, 2.0]))),

        # arithmetic
        ('add', lambda: repr(torch.tensor([1.0]) + torch.tensor([2.0]))),
        ('add_scalar', lambda: repr(torch.tensor([1.0]) + 2.0)),
        ('radd', lambda: repr(2.0 + torch.tensor([1.0]))),
        ('sub', lambda: repr(torch.tensor([1.0]) - torch.tensor([2.0]))),
        ('rsub', lambda: repr(2.0 - torch.tensor([1.0]))),
        ('mul', lambda: repr(torch.tensor([1.0]) * torch.tensor([2.0]))),
        ('div', lambda: repr(torch.tensor([1.0]) / torch.tensor([2.0]))),
        ('floordiv', lambda: repr(torch.tensor([5.0]) // torch.tensor([2.0]))),
        ('mod', lambda: repr(torch.tensor([5.0]) % torch.tensor([3.0]))),
        ('pow', lambda: repr(torch.tensor([2.0]) ** 3)),
        ('matmul_op', lambda: repr(torch.tensor([[1.0, 2.0]]) @ torch.tensor([[1.0], [2.0]]))),
        ('neg', lambda: repr(-torch.tensor([1.0]))),
        ('pos', lambda: repr(+torch.tensor([1.0]))),
        ('abs', lambda: repr(abs(torch.tensor([-1.0])))),
        ('inplace_add', lambda: (lambda t: (t.add_(1.0), repr(t))[1])(torch.tensor([1.0]))),
        ('inplace_mul', lambda: (lambda t: (t.mul_(2.0), repr(t))[1])(torch.tensor([1.0]))),
        ('fill_', lambda: (lambda t: (t.fill_(7.0), repr(t))[1])(torch.tensor([1.0, 2.0]))),

        # bitwise / shift
        ('bit_and', lambda: repr(torch.tensor([1], dtype=torch.int32) & torch.tensor([3], dtype=torch.int32))),
        ('bit_or', lambda: repr(torch.tensor([1], dtype=torch.int32) | torch.tensor([2], dtype=torch.int32))),
        ('bit_xor', lambda: repr(torch.tensor([1], dtype=torch.int32) ^ torch.tensor([3], dtype=torch.int32))),
        ('bit_not', lambda: repr(~torch.tensor([True]))),
        ('shift_l', lambda: repr(torch.tensor([1], dtype=torch.int32) << 2)),
        ('shift_r', lambda: repr(torch.tensor([8], dtype=torch.int32) >> 1)),

        # comparison
        ('eq', lambda: repr(torch.tensor([1.0]) == torch.tensor([1.0]))),
        ('ne', lambda: repr(torch.tensor([1.0]) != torch.tensor([2.0]))),
        ('lt', lambda: repr(torch.tensor([1.0]) < torch.tensor([2.0]))),
        ('gt', lambda: repr(torch.tensor([2.0]) > torch.tensor([1.0]))),
        ('le', lambda: repr(torch.tensor([1.0]) <= torch.tensor([1.0]))),
        ('ge', lambda: repr(torch.tensor([1.0]) >= torch.tensor([1.0]))),

        # conversions
        ('to_bool', lambda: bool(torch.tensor(True))),
        ('to_int', lambda: int(torch.tensor(3))),
        ('to_float', lambda: float(torch.tensor(1.5))),
        ('to_complex', lambda: complex(torch.tensor(1.0 + 2.0j))),
        ('to_format', lambda: format(torch.tensor(1.0), '')),
        ('to_str', lambda: str(torch.tensor([1.0, 2.0]))),
        ('to_repr', lambda: repr(torch.tensor([1.0, 2.0]))),
        ('to_numpy', lambda: np.asarray(torch.tensor([1.0])).tolist()),
        ('to_list', lambda: torch.tensor([1.0, 2.0]).tolist()),
        ('to_item_0d', lambda: torch.tensor(3.14).item()),

        # sequence protocol
        ('len_', lambda: len(torch.tensor([1.0, 2.0, 3.0]))),
        ('iter_', lambda: [repr(x) for x in torch.tensor([1.0, 2.0])]),
        ('reversed_', lambda: [repr(x) for x in reversed(torch.tensor([1.0, 2.0]))]),
        ('contains', lambda: 1.0 in torch.tensor([1.0, 2.0])),

        # reshaping / layout
        ('view', lambda: repr(torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2))),
        ('reshape', lambda: repr(torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(4, 1))),
        ('transpose_T', lambda: repr(torch.tensor([[1.0, 2.0], [3.0, 4.0]]).T)),
        ('permute', lambda: repr(torch.tensor([[[1.0, 2.0]]]).permute(2, 0, 1).shape)),
        ('unsqueeze', lambda: repr(torch.tensor([1.0]).unsqueeze(0).shape)),
        ('squeeze', lambda: repr(torch.tensor([[1.0]]).squeeze().shape)),
        ('flatten', lambda: repr(torch.tensor([[1.0, 2.0]]).flatten())),
        ('expand', lambda: repr(torch.tensor([[1.0]]).expand(2, 2))),

        # reductions
        ('sum_', lambda: repr(torch.tensor([1.0, 2.0]).sum())),
        ('mean_', lambda: repr(torch.tensor([1.0, 2.0]).mean())),
        ('max_', lambda: repr(torch.tensor([1.0, 2.0]).max())),
        ('min_', lambda: repr(torch.tensor([1.0, 2.0]).min())),
        ('argmax', lambda: repr(torch.tensor([3.0, 1.0]).argmax())),
        ('norm', lambda: repr(torch.linalg.norm(torch.tensor([3.0, 4.0])))),

        # cloning / device / dtype / serialization
        ('clone', lambda: repr(torch.tensor([1.0]).clone())),
        ('detach', lambda: repr(torch.tensor([1.0]).detach())),
        ('to_dtype', lambda: repr(torch.tensor([1]).to(torch.float32))),
        ('float_method', lambda: repr(torch.tensor([1]).float())),
        ('deepcopy_', lambda: repr(copy.deepcopy(torch.tensor([1.0, 2.0])))),
        ('pickle_roundtrip', lambda: repr(pickle.loads(pickle.dumps(torch.tensor([1.0, 2.0]))))),

        # module-level / linalg / functional
        ('stack', lambda: repr(torch.stack([torch.tensor(1.0), torch.tensor(2.0)]))),
        ('cat', lambda: repr(torch.cat([torch.tensor([1.0]), torch.tensor([2.0])]))),
        ('matmul_fn', lambda: repr(torch.matmul(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0], [2.0]])))),
        ('bmm', lambda: repr(torch.bmm(torch.randn(2, 2, 3), torch.randn(2, 3, 2)).shape)),
        ('einsum', lambda: repr(torch.einsum('i,i->', torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))),
        ('where', lambda: repr(torch.where(torch.tensor([True, False]), torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))),
        ('gather', lambda: repr(torch.gather(torch.tensor([[1.0, 2.0]]), 1, torch.tensor([[0]])))),
        ('relu_fwd', lambda: repr(F.relu(torch.tensor([-1.0, 2.0])))),
        ('softmax_fwd', lambda: repr(F.softmax(torch.tensor([1.0, 2.0]), dim=0))),
        ('linear_fwd', lambda: repr(F.linear(torch.randn(3, 5), torch.randn(7, 5)).shape)),
        ('conv2d_fwd', lambda: repr(F.conv2d(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 2, 2)).shape)),
        ('layernorm_fwd', lambda: repr(F.layer_norm(torch.randn(2, 3), (3,)).shape)),

        # autograd
        ('requires_grad_set', lambda: (lambda t: t.requires_grad_(True).requires_grad)(torch.tensor([1.0]))),
        ('grad_fn_name', lambda: type((torch.tensor([1.0], requires_grad=True) * 2).grad_fn).__name__),

        # exceptions we expect to keep raising
        ('len_0d_raises', lambda: len(torch.tensor(1.0))),
        ('iter_0d_raises', lambda: list(torch.tensor(1.0))),
        ('bool_multi_raises', lambda: bool(torch.tensor([1.0, 2.0]))),
        ('int_of_multi_raises', lambda: int(torch.tensor([1.0, 2.0]))),

        # math module interop
        ('math_floor', lambda: math.floor(torch.tensor(1.7))),
        ('math_ceil', lambda: math.ceil(torch.tensor(1.2))),
        ('math_trunc', lambda: math.trunc(torch.tensor(1.7))),
        ('round_', lambda: round(torch.tensor(1.5).item())),
    ]
    out = {}
    for name, fn in probes:
        try:
            out[name] = ('ok', fn())
        except Exception as e:
            out[name] = ('exc', f'{type(e).__name__}: {e}')
    return out


def make_passthrough(orig):
    def passthrough(*args, **kwargs):
        return orig(*args, **kwargs)
    return passthrough


# Phase A
corrupting = []
for entry in FUNCTIONS:
    namespace = entry['namespace']
    func_name = entry['function']
    module = NAMESPACE_TO_MODULE.get(namespace)
    if module is None:
        continue
    try:
        orig = getattr(module, func_name)
    except AttributeError:
        continue
    if not callable(orig):
        continue

    pt = make_passthrough(orig)
    before = probe_battery()
    try:
        setattr(module, func_name, pt)
        setattr(module, func_name, orig)
    except Exception:
        continue
    after = probe_battery()
    if before != after:
        corrupting.append((namespace, func_name))

# Phase B invocations. Dunders absent here are reported as untestable.
INVOCATIONS = {
    '__getitem__': lambda t: t[0],
    '__setitem__': lambda t: t.__setitem__(0, 0.0),
    '__add__': lambda t: t + t,
    '__radd__': lambda t: 1.0 + t,
    '__iadd__': lambda t: t.clone().__iadd__(t),
    '__sub__': lambda t: t - t,
    '__rsub__': lambda t: 1.0 - t,
    '__isub__': lambda t: t.clone().__isub__(t),
    '__mul__': lambda t: t * t,
    '__rmul__': lambda t: 1.0 * t,
    '__imul__': lambda t: t.clone().__imul__(t),
    '__truediv__': lambda t: t / t,
    '__rtruediv__': lambda t: 1.0 / t,
    '__itruediv__': lambda t: t.clone().__itruediv__(t + 1),
    '__floordiv__': lambda t: t // (t + 1),
    '__rfloordiv__': lambda t: 1.0 // (t + 1),
    '__mod__': lambda t: t % (t + 1),
    '__rmod__': lambda t: 1.0 % (t + 1),
    '__pow__': lambda t: t ** 2,
    '__rpow__': lambda t: 2 ** t,
    '__matmul__': lambda t: torch.tensor([[1.0]]) @ torch.tensor([[2.0]]),
    '__rmatmul__': lambda t: torch.tensor([[1.0]]).__rmatmul__(torch.tensor([[2.0]])),
    '__and__': lambda t: torch.tensor([1], dtype=torch.int32) & torch.tensor([1], dtype=torch.int32),
    '__or__': lambda t: torch.tensor([1], dtype=torch.int32) | torch.tensor([1], dtype=torch.int32),
    '__xor__': lambda t: torch.tensor([1], dtype=torch.int32) ^ torch.tensor([1], dtype=torch.int32),
    '__lshift__': lambda t: torch.tensor([1], dtype=torch.int32) << 1,
    '__rshift__': lambda t: torch.tensor([2], dtype=torch.int32) >> 1,
    '__neg__': lambda t: -t,
    '__pos__': lambda t: +t,
    '__abs__': lambda t: abs(t),
    '__invert__': lambda t: ~torch.tensor([True]),
    '__eq__': lambda t: t == t,
    '__ne__': lambda t: t != t,
    '__lt__': lambda t: t < t,
    '__gt__': lambda t: t > t,
    '__le__': lambda t: t <= t,
    '__ge__': lambda t: t >= t,
    '__bool__': lambda t: bool(t[0:1].any()),
    '__int__': lambda t: int(t[0:1].sum()),
    '__float__': lambda t: float(t[0:1].sum()),
    '__index__': lambda t: torch.tensor(0).__index__(),
    '__len__': lambda t: len(t),
    '__iter__': lambda t: list(t),
    '__contains__': lambda t: 1.0 in t,
    '__repr__': lambda t: repr(t),
    '__hash__': lambda t: hash(t),
    '__array__': lambda t: __import__('numpy').asarray(t),
    '__format__': lambda t: format(t, ''),
    '__complex__': lambda t: complex(t[0:1].sum().to(torch.complex64)),
    '__deepcopy__': lambda t: __import__('copy').deepcopy(t),
    '__round__': lambda t: round(t.sum()),
    '__trunc__': lambda t: __import__('math').trunc(t[0]),
    '__floor__': lambda t: __import__('math').floor(t[0]),
    '__ceil__': lambda t: __import__('math').ceil(t[0]),
    '__reversed__': lambda t: list(reversed(t)),
}


class RecordingMode(torch.overrides.TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.seen.append(func)
        return func(*args, **(kwargs or {}))


# Phase B
unsolvable = []
for namespace, func_name in corrupting:
    invocation = INVOCATIONS.get(func_name)
    if invocation is None:
        unsolvable.append((namespace, func_name, 'no_invocation_defined'))
        continue
    rec = RecordingMode()
    test_tensor = torch.tensor([1.0, 2.0, 3.0])
    try:
        with rec:
            invocation(test_tensor)
    except Exception:
        pass
    if not rec.seen:
        unsolvable.append((namespace, func_name, 'mode_saw_no_func'))


print(f'Phase A: {len(corrupting)} corrupting entries:')
for ns, fn in corrupting:
    print(f'  {ns}.{fn}')
print(f'Phase B: {len(corrupting) - len(unsolvable)} of those are interceptable.')
print()
print('UNSOLVABLE (corrupts AND mode cannot intercept):')
if not unsolvable:
    print('  (none)')
else:
    for ns, fn, reason in unsolvable:
        print(f'  {ns}.{fn}  [{reason}]')
