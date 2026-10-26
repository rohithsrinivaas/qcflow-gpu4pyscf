# cupy links its cuTENSOR bindings at load time, so the cutensor-cu12 wheel's lib directory
# has to be on LD_LIBRARY_PATH before python starts; otherwise gpu4pyscf silently falls back
# to cupy.einsum for tensor contractions (slower). Install as a conda activation hook:
#   cp env/cutensor-activate.sh $CONDA_PREFIX/etc/conda/activate.d/
_lib="$CONDA_PREFIX/lib/python3.12/site-packages/cutensor/lib"
if [ -d "$_lib" ]; then
    export LD_LIBRARY_PATH="$_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _lib
