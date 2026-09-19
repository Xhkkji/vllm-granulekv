from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="solidattention-runtime",
    ext_modules=[
        CUDAExtension(
            name="_solidattention_runtime",
            sources=["solidattention_runtime.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
