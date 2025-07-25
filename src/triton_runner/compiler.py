from triton.runtime import driver
from triton.runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend, triton_key, parse, filter_traceback
from triton.compiler.compiler import ASTSource, IRSource, CompiledKernel
from triton._C.libtriton import get_cache_invalidating_env_vars, ir, llvm
import triton
import hashlib
import os
import json
from pathlib import Path
from .check_utils import runner_check_triton


def native_compile(src, ast_src, metadata_json=dict(), target=None, options=None):
    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        context = ir.context()
        if src.endswith("llir"):
            module = Path(src).read_text()
            llvm.init_targets()
        elif src.endswith("cubin"):
            module = Path(src).read_bytes()
        else:
            if triton.__version__ == "3.2.0":
                src = IRSource(src)
            else:
                src = IRSource(src, context, backend)

    ast_extra_options = ast_src.parse_options()

    if isinstance(src, ASTSource) or isinstance(src, IRSource):
        extra_options = src.parse_options()
    else:
        extra_options = {}
    extra_options = extra_options | ast_extra_options
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    env_vars = get_cache_invalidating_env_vars()
    if isinstance(src, ASTSource) or isinstance(src, IRSource):
        src_hash = src.hash()
    elif src.endswith("cubin"):
        src_hash = hashlib.sha256(module).hexdigest()
    else:
        src_hash = hashlib.sha256(module.encode("utf-8")).hexdigest()
    key = f"{triton_key()}-{src_hash}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    store_only_binary = os.environ.get("TRITON_STORE_BINARY_ONLY", "0") == "1"
    fn_override_manager = get_override_manager(src_hash) if enable_override else None
    fn_dump_manager = get_dump_manager(src_hash) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = ast_src.name[:150]
    if metadata_json:
        runner_check_triton(file_name, metadata_json, target)
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0") == "1"
    if not always_compile and metadata_path is not None:
        # cache hit!
        return CompiledKernel(ast_src, metadata_group, hash)
    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = triton.__version__
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options)
    if isinstance(src, ASTSource) or isinstance(src, IRSource):
        src_ext = src.ext
    else:
        src_ext = Path(src).suffix[1:]
    first_stage = list(stages.keys()).index(src_ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1

    # For IRSource, we have already grabbed the context + called both
    # ir.load_dialects and backend.load_dialects.
    if not isinstance(src, IRSource):
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
    if triton.__version__ == "3.2.0":
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation()
    else:
        codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    try:
        if src_ext == "ptx":
            module = src.src
        elif src_ext not in {"llir", "cubin"}:
            module = src.make_ir(options, codegen_fns, module_map, context)
    except Exception as e:
        filter_traceback(e)
        raise
    use_ir_loc = os.environ.get("USE_IR_LOC", None)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if (fn_override_manager is not None and (full_name := fn_override_manager.get_file(ir_filename)) is not None):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
        if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module

    if src_ext == "cubin":
        ir_filename = f"{file_name}.cubin"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    if metadata_json:
        metadata["name"] = metadata_json["name"]
        metadata["shared"] = metadata_json["shared"]
        if not triton.__version__ == "3.2.0":
            metadata["tmem_size"] = metadata_json["tmem_size"]
            metadata["global_scratch_size"] = metadata_json["global_scratch_size"]
            metadata["global_scratch_align"] = metadata_json["global_scratch_align"]

    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    # Compilation completed, disabling multithreading in context.
    # This is needed to safely finalize threads pool inside context: if current process forks before
    # python GC deletes context object, thread pool in child process will be invalid, which could
    # lead to child crash or hang.
    #
    # However disabling multithreading causes the code to hang if the ASAN pass is enabled
    # this is likely due to the llvm-symbolizer forking a process
    # TODO: Reconcile the difference here between the ASAN and non-ASAN path with enabling
    # multithreading in the MLIR context
    if not os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
        context.disable_multithreading()
    # return handle to compiled kernel
    return CompiledKernel(ast_src, metadata_group, hash)
