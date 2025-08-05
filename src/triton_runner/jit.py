import triton
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction, KernelInterface, T
from typing import Callable, Iterable, Optional, Union, overload
from .compiler import native_compile
import os
import json
import re


class RunnerJITFunction(JITFunction[KernelInterface[T]]):

    def get_runner_args_set(self):
        return {"cubin_dir", "ttir_dir", "ttgir_dir", "llir_dir", "ptx_dir", "debug_value", "debug_tensor"}

    def get_source_dir_type(self, need_check_lst):
        runner_args_set = self.get_runner_args_set()
        for k in need_check_lst:
            if not k in runner_args_set:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        for k in need_check_lst:
            if k in runner_args_set:
                return k


class RunnerJITFunctionV3_4_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, kwargs, options, sigkeys):
        return super().get_source_dir_type(
            [k.lower() for k in kwargs if k not in options.__dict__ and k not in sigkeys])

    def need_debug(self, kwargs):
        return "debug_tensor" in kwargs and "debug_value" in kwargs

    def insert_debug_tensor_param(self, full_text):
        pattern = re.compile(r'(tt\.func\s+public\s+@\w+\s*)\((.*?)\)(\s*attributes\s*{[^}]*}\s*{)', re.DOTALL)

        def replacer(match):
            prefix, args_str, suffix = match.groups()
            new_args_str = args_str + ', %debug_tensor: !tt.ptr<f32>'
            return f"{prefix}({new_args_str}){suffix}"

        return pattern.sub(replacer, full_text, count=1)

    def inject_debug_store(self, full_text, ssa_value):
        pattern = re.compile(
            rf'^(?P<indent>\s*){ssa_value}\s*=\s*'
            r'(?:tt\.load)\s+'
            r'(?P<op1>%\d+)\s*,\s*(?P<op2>%\d+)[^:]+:\s*'
            r'tensor<(?P<size>\d+)x[^>]*>'
            r'.*?'
            r'loc\((?P<loc>[^)]+)\)', 
            re.MULTILINE
        )
        def make_replacer(full_text):
            def replacer(match):
                original_line = match.group(0)
                indent = match.group("indent")
                size = match.group("size")
                loc = match.group("loc")
                addptr_value = match.group("op1")
                op2 = match.group("op2")
                splat_pattern = re.compile(
                    rf'^(?P<indent>\s*){addptr_value}\s*=\s*'
                    r'(?:tt\.addptr)\s+'
                    r'(?P<op1>%\d+)\s*,\s*(?P<op2>%\d+)',
                    re.MULTILINE
                )
                addptr_match = splat_pattern.search(full_text)
                if addptr_match:
                    addptr_op2 = addptr_match.group("op2")
                injected_code = f"""{original_line}
{indent}%splat = tt.splat %debug_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>> loc({loc})
{indent}%ptr = tt.addptr %splat, {addptr_op2} : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32> loc({loc})
{indent}tt.store %ptr, {ssa_value}, {op2} : tensor<{size}x!tt.ptr<f32>> loc({loc})"""
                return injected_code
            return replacer
        replacer_with_text = make_replacer(full_text)
        return pattern.sub(replacer_with_text, full_text, count=1)

    def run(self, *args, grid, warmup, **kwargs):
        from triton import knobs
        from triton._utils import find_paths_if, get_iterable_path

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        # compute cache key
        key = str(specialization) + str(options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            # options
            options = backend.parse_options(kwargs)
            # signature
            sigkeys = [x.name for x in self.params]
            sigvals = [x[0] for x in specialization]
            signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
            # check arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"

            # check keyword argument and get source_dir_type
            source_dir_type = self.get_source_dir_type(kwargs, options, sigkeys)
            if self.need_debug(kwargs) and source_dir_type == "ttir_dir":
                signature["debug_tensor"] = "*fp32"
                bound_args["debug_tensor"] = kwargs["debug_tensor"]

            # constexprs
            constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
            constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
            # attributes
            attrvals = [x[1] for x in specialization]
            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}
            if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs],
                               warmup):
                return None
            # compile the kernel
            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            metadata_json = {}
            if source_dir_type:
                source_file_name = f"{self.__name__}.{source_dir_type[:-4]}"
                src = os.path.join(kwargs[source_dir_type], source_file_name)
                if self.need_debug(kwargs) and source_dir_type == "ttir_dir":
                    debug_content = self.insert_debug_tensor_param(open(src, "r").read())
                    debug_content = self.inject_debug_store(debug_content, kwargs["debug_value"])
                    src = os.path.join(kwargs[source_dir_type], "debug.ttir")
                    with open(src, "w") as file:
                        file.write(debug_content)
                if source_dir_type in {"cubin_dir", "llir_dir", "ptx_dir"}:
                    json_file_name = f"{self.__name__}.json"
                    json_path = os.path.join(kwargs[source_dir_type], json_file_name)
                    metadata_json = json.loads(open(json_path, "r").read())
            else:
                src = ast_src
            kernel_signature = tuple((key, arg_type, spec) for key, (arg_type, spec) in zip(bound_args.keys(), specialization))
            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, kernel_signature=kernel_signature)
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs],
                            warmup)

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_3_x(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, kwargs, options, sigkeys):
        return super().get_source_dir_type(
            [k.lower() for k in kwargs if k not in options.__dict__ and k not in sigkeys])

    def run(self, *args, grid, warmup, **kwargs):
        from triton._utils import find_paths_if, get_iterable_path

        kwargs["debug"] = kwargs.get("debug", self.debug) or os.environ.get("TRITON_DEBUG", "0") == "1"

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        # compute cache key
        key = str(specialization) + str(options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            # options
            options = backend.parse_options(kwargs)
            # signature
            sigkeys = [x.name for x in self.params]
            sigvals = [x[0] for x in specialization]
            signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
            # check arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"

            # check keyword argument and get source_dir_type
            source_dir_type = self.get_source_dir_type(kwargs, options, sigkeys)

            # constexprs
            constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
            constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
            # attributes
            attrvals = [x[1] for x in specialization]
            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}
            if self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=True):
                return None
            # compile the kernel
            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            metadata_json = {}
            if source_dir_type:
                source_file_name = f"{self.__name__}.{source_dir_type[:-4]}"
                src = os.path.join(kwargs[source_dir_type], source_file_name)
                if source_dir_type in {"cubin_dir", "llir_dir", "ptx_dir"}:
                    json_file_name = f"{self.__name__}.json"
                    json_path = os.path.join(kwargs[source_dir_type], json_file_name)
                    metadata_json = json.loads(open(json_path, "r").read())
            else:
                src = ast_src

            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__)
            kernel_cache[key] = kernel
            self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=False)

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata,
                       launch_metadata, self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook,
                       *bound_args.values())
        return kernel

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        super().__init__(fn, version, do_not_specialize, do_not_specialize_on_alignment, debug, noinline, repr,
                         launch_metadata)


class RunnerJITFunctionV3_2_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, excess_kwargs, options):
        return super().get_source_dir_type([k.lower() for k in excess_kwargs if k not in options.__dict__])

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", False) or os.environ.get("TRITON_DEBUG", "0") == "1"

        # parse options
        from triton.compiler import make_backend
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        target = driver.active.get_current_target()
        backend = make_backend(target)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        if self.binder is None:
            self.create_binder(backend)

        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)

        # compute cache key
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            # Kernel is not cached; we have to compile.
            options = backend.parse_options(kwargs)

            # deprecated arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"

            # check keyword argument and get source_dir_type
            source_dir_type = self.get_source_dir_type(excess_kwargs, options)

            bound_vals = tuple(bound_args.values())

            # `None` is nullptr. Implicitly convert to *i8. This needs to be
            # done here rather than when we build the signature as otherwise
            # the kernel cache key could not distinguish between byte pointers
            # and None arguments, resulting in a downstream mismatch:
            sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
            sigvals = sig_and_spec[:len(sigkeys)]
            signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

            configs = (backend.get_attrs_descriptor(self.params, bound_vals), )
            constant_params = configs[0].get_constants()
            constants = {
                p.name: v
                for (v, p) in zip(bound_vals, self.params)
                if p.is_constexpr or (p.num in constant_params) or v is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            if self._call_hook(key, signature, device, constants, options, configs, warmup, before=True):
                return None
            # compile the kernel
            ast_src = self.ASTSource(self, signature, constants, configs[0])
            metadata_json = {}
            if source_dir_type:
                source_file_name = f"{self.__name__}.{source_dir_type[:-4]}"
                src = os.path.join(kwargs[source_dir_type], source_file_name)
                if source_dir_type in {"cubin_dir", "llir_dir", "ptx_dir"}:
                    json_file_name = f"{self.__name__}.json"
                    json_path = os.path.join(kwargs[source_dir_type], json_file_name)
                    metadata_json = json.loads(open(json_path, "r").read())
            else:
                src = ast_src

            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__)
            self.cache[device][key] = kernel
            self._call_hook(key, signature, device, constants, options, configs, warmup, before=False)

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                # Arguments are passed as a dict to `grid`, by contract.
                # TODO(jlebar): In the new launch API, pass the compiler flags as a
                # second parameter to `grid`.
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        return kernel


# -----------------------------------------------------------------------------
# jit decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: T) -> RunnerJITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], RunnerJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[RunnerJITFunction[T], Callable[[T], RunnerJITFunction[T]]]:

    def decorator(fn: T) -> RunnerJITFunction[T]:
        assert callable(fn)
        if triton.__version__ in ["3.3.0", "3.3.1"]:
            return RunnerJITFunctionV3_3_x(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        if triton.__version__ == "3.2.0":
            return RunnerJITFunctionV3_2_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        if triton.__version__ == "3.4.0":
            return RunnerJITFunctionV3_4_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
