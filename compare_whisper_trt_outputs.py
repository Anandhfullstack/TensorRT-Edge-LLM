import numpy as np

python_path = "whisper_python_trt_output_fp16.bin"
cpp_path = "whisper_cpp_trt_output_fp16.bin"

py = np.fromfile(python_path, dtype=np.float16)
cpp = np.fromfile(cpp_path, dtype=np.float16)

print("Python elements:", py.size)
print("C++ elements   :", cpp.size)

assert py.size == cpp.size, "Output sizes do not match"

py32 = py.astype(np.float32)
cpp32 = cpp.astype(np.float32)

diff = np.abs(py32 - cpp32)

mean_diff = np.mean(diff)
median_diff = np.median(diff)
max_diff = np.max(diff)

rmse = np.sqrt(
    np.mean((py32 - cpp32) ** 2)
)

cosine = np.dot(py32, cpp32) / (
    np.linalg.norm(py32) *
    np.linalg.norm(cpp32)
)

exact_mismatch = np.count_nonzero(py != cpp)

print()
print("Mean abs diff :", mean_diff)
print("Median diff   :", median_diff)
print("Max diff      :", max_diff)
print("RMSE          :", rmse)
print("Cosine        :", cosine)
print(
    "FP16 mismatch :",
    exact_mismatch,
    "/",
    py.size,
)