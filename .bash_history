ls
cd root/
ls
cd ..
apt update
apt install git
apt install vim
apt install clang llvm
ls
cd root/
ls
git clone https://github.com/google/atheris.git
ls
cd atheris/
ls
python3 -V
ls
apt install python3-pip
lks
ls
pip install .
pwd
ls
exit
ls 
cd root/
ls
python3 -m venv venv_pytorch_v2
apt install python3.10-venv
dpkg --list
python3 -m venv pytorch_cov/bin/activate
cd pytorch
pip install -r requirements.txt 
apt-get install -y lld
cd ..,
cd ..
source pytorch_cov/bin/activate
ls
python3 -m venv pytorch_cov
rm pytorch_cov/
rm -r pytorch_cov/
rm -r venv_pytorch_v2/
python3 -m venv pytorch_cov
source pytorch_cov/bin/activate
cd pytorch
pip install -r requirements.txt 
apt-get install -y lld
export LLVM_VERSION=17
export LLVM_PREFIX=/usr/lib/llvm-$LLVM_VERSION
# 指定编译器为 clang/clang++
export CC=clang
export CXX=clang++
# llvm-profdata / llvm-cov 所在路径（供 oss_coverage.py 调用）
export LLVM_TOOL_PATH="$LLVM_PREFIX/bin"
export PATH="$LLVM_TOOL_PATH:$PATH"
# 关键：强制使用 LLD 作为链接器（修 PR #136632 同款做法）
# 用 LDFLAGS 注入足够；若你走显式 cmake，可改成 -DCMAKE_*_LINKER_FLAGS
# 这个环境变量当时报错了，设置成下一个就对了
# export LDFLAGS="${LDFLAGS:--} -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld ${LDFLAGS:-}"
# 让构建走 Debug 配置（便于覆盖率 & 符号）
export CMAKE_BUILD_TYPE=Debug
# 让 setup.py / cmake 打开 PyTorch 的 C++ 覆盖率开关 & 单测
# （如果你改走纯 cmake 命令，也可以用 -DUSE_CPP_CODE_COVERAGE=ON -DBUILD_TEST=ON 传参）
export USE_CPP_CODE_COVERAGE=ON
export BUILD_TEST=1
# 运行测试时把 .profraw 落到固定目录（%p=pid, %m=二进制名）
# 若用 oss_coverage.py，它也会自行设置；手动跑测试时建议显式设置
export LLVM_PROFILE_FILE="profile/pytorch-%p-%m.profraw"
python setup.py build --cmake-only
apt-get install libomp-dev
git clean -xfd
# 版本按你机器装的改（常见 16/17/18）
export LLVM_VERSION=17
export LLVM_PREFIX=/usr/lib/llvm-$LLVM_VERSION
# 指定编译器为 clang/clang++
export CC=clang
export CXX=clang++
# llvm-profdata / llvm-cov 所在路径（供 oss_coverage.py 调用）
export LLVM_TOOL_PATH="$LLVM_PREFIX/bin"
export PATH="$LLVM_TOOL_PATH:$PATH"
# 关键：强制使用 LLD 作为链接器（修 PR #136632 同款做法）
# 用 LDFLAGS 注入足够；若你走显式 cmake，可改成 -DCMAKE_*_LINKER_FLAGS
# 这个环境变量当时报错了，设置成下一个就对了
# export LDFLAGS="${LDFLAGS:--} -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld ${LDFLAGS:-}"
# 让构建走 Debug 配置（便于覆盖率 & 符号）
export CMAKE_BUILD_TYPE=Debug
# 让 setup.py / cmake 打开 PyTorch 的 C++ 覆盖率开关 & 单测
# （如果你改走纯 cmake 命令，也可以用 -DUSE_CPP_CODE_COVERAGE=ON -DBUILD_TEST=ON 传参）
export USE_CPP_CODE_COVERAGE=ON
export BUILD_TEST=1
# 运行测试时把 .profraw 落到固定目录（%p=pid, %m=二进制名）
# 若用 oss_coverage.py，它也会自行设置；手动跑测试时建议显式设置
export LLVM_PROFILE_FILE="profile/pytorch-%p-%m.profraw"
python3 -V
python3 setup.py build --cmake-only
cd build
cmake ..   -DUSE_CPP_CODE_COVERAGE=ON   -DBUILD_TEST=ON   -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_C_COMPILER=clang   -DCMAKE_CXX_COMPILER=clang++   -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld"   -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
cmake --build . -j"$(nproc)"
cd ..
python setup.py develop
export LLVM_PROFILE_FILE="pytorch_%p.profraw"
python3 -c "
# import load_profile_runtime  # 如果使用方案4
import torch
x = torch.randn(10, 10)
y = x + x
print(y.shape)
"
llvm-profdata merge -sparse pytorch_*.profraw -o pytorch.profdata
llvm-cov report /root/pytorch/torch/lib/libtorch_cpu.so -instr-profile=pytorch.profdata
unset LLVM_PROFILE_FILE
ls
cd .
cd ..
ls
deactivate
ls
cd pytorch
ls
cd build
ls
cd ..
ls
rm -r build
ls 
cd ..
source pytorch_cov/bin/acvtivate
source pytorch_cov/bin/activate
python3
deactivate
source pytorch_cov/bin/acvtivate
ls
source pytorch_cov/bin/acvtivate
cd pytorch
cd ..
ls
cd pytorch_cov/
ls
cd bin
ls
cd ..
cd .
cd ..
source pytorch_cov/bin/activate
ls
vim torch_fuzz.py
ls
vim py_cov_filter_report.py
ls
cd corpus/
ls
cd ..
export LLVM_PROFILE_FILE="pytorch_%p.profraw"
python3 torch_fuzz.py corpus -runs=0  -print_final_stats=1
vim torch_fuzz.py 
ls
python3 - <<'PY'
import atheris, inspect, sys, pkgutil
print("atheris file:", atheris.__file__)
print("has instrument_imports?:", hasattr(atheris, "instrument_imports"))
print("version attr:", getattr(atheris, "__version__", None))
PY

pip install atheris
python3 torch_fuzz.py corpus -runs=0  -print_final_stats=1
# 这个是链接到so库
export ASAN_OPTIONS=detect_leaks=0
export LD_PRELOAD="$(python3 -c 'import atheris, os; print(os.path.join(atheris.path(), "asan_with_fuzzer.so"))')"
ls -l "$LD_PRELOAD"   # 应该能看到 asan_with_fuzzer.so
ls
export LLVM_PROFILE_FILE="pytorch_%p.profraw"
python3 torch_fuzz.py corpus -runs=0  -print_final_stats=1
llvm-profdata merge -sparse pytorch_*.profraw -o pytorch.profdata
ls
export LLVM_PROFILE_FILE="pytorch_%p.profraw"
python3 -c "
# import load_profile_runtime  # 如果使用方案4
import torch
x = torch.randn(10, 10)
y = x + x
print(y.shape)
"
ls
echo $LLVM_PROFILE_FILE
unset LLVM_PROFILE_FILE
cd pytorch
ls
rm -r build
cd ..
python3 - <<'PY'
import sys, subprocess, json
try:
    # pip>=21 有 JSON 格式
    out=subprocess.check_output([sys.executable,"-m","pip","list","--format","json"], text=True)
    pkgs=[p for p in json.loads(out) if "torch" in p["name"].lower()]
    for p in pkgs:
        print(f"{p['name']:30} {p['version']}")
except Exception:
    # 退化输出
    import pkgutil
    print("pip json failed; fallback via pkgutil:")
    for m in pkgutil.iter_modules():
        if "torch" in m.name.lower():
            print(m.name)
PY

python3 -m pip show torch
ls
export LLVM_PROFILE_FILE="pytorch_%p.profraw"
python3 -c "
# import load_profile_runtime  # 如果使用方案4
import torch
x = torch.randn(10, 10)
y = x + x
print(y.shape)
"
ls
unset LLVM_PROFILE_FILE
cd pytorch
ls
git fetch --all -p
git worktree add /root/pytorch-cov-src HEAD
ls
cd ..
ls
cd pytorch-cov-src/
ls
rm -rf build
git clean -xfd
# 版本按你机器装的改（常见 16/17/18）
export LLVM_VERSION=17
export LLVM_PREFIX=/usr/lib/llvm-$LLVM_VERSION
# 指定编译器为 clang/clang++
export CC=clang
export CXX=clang++
# llvm-profdata / llvm-cov 所在路径（供 oss_coverage.py 调用）
export LLVM_TOOL_PATH="$LLVM_PREFIX/bin"
export PATH="$LLVM_TOOL_PATH:$PATH"
# 关键：强制使用 LLD 作为链接器（修 PR #136632 同款做法）
# 用 LDFLAGS 注入足够；若你走显式 cmake，可改成 -DCMAKE_*_LINKER_FLAGS
# 这个环境变量当时报错了，设置成下一个就对了
# export LDFLAGS="${LDFLAGS:--} -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld ${LDFLAGS:-}"
# 让构建走 Debug 配置（便于覆盖率 & 符号）
export CMAKE_BUILD_TYPE=Debug
# 让 setup.py / cmake 打开 PyTorch 的 C++ 覆盖率开关 & 单测
# （如果你改走纯 cmake 命令，也可以用 -DUSE_CPP_CODE_COVERAGE=ON -DBUILD_TEST=ON 传参）
export USE_CPP_CODE_COVERAGE=ON
export BUILD_TEST=1
# 运行测试时把 .profraw 落到固定目录（%p=pid, %m=二进制名）
# 若用 oss_coverage.py，它也会自行设置；手动跑测试时建议显式设置
export LLVM_PROFILE_FILE="profile/pytorch-%p-%m.profraw"
apt-get install libomp-dev
apt-get install -y lld
python3 setup.py build --cmake-only
git clean -xfd
export LLVM_VERSION=17
export LLVM_PREFIX=/usr/lib/llvm-$LLVM_VERSION
# 指定编译器为 clang/clang++
export CC=clang
export CXX=clang++
# llvm-profdata / llvm-cov 所在路径（供 oss_coverage.py 调用）
export LLVM_TOOL_PATH="$LLVM_PREFIX/bin"
export PATH="$LLVM_TOOL_PATH:$PATH"
# 关键：强制使用 LLD 作为链接器（修 PR #136632 同款做法）
# 用 LDFLAGS 注入足够；若你走显式 cmake，可改成 -DCMAKE_*_LINKER_FLAGS
# 这个环境变量当时报错了，设置成下一个就对了
# export LDFLAGS="${LDFLAGS:--} -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld ${LDFLAGS:-}"
# 让构建走 Debug 配置（便于覆盖率 & 符号）
export CMAKE_BUILD_TYPE=Debug
# 让 setup.py / cmake 打开 PyTorch 的 C++ 覆盖率开关 & 单测
# （如果你改走纯 cmake 命令，也可以用 -DUSE_CPP_CODE_COVERAGE=ON -DBUILD_TEST=ON 传参）
export USE_CPP_CODE_COVERAGE=ON
export BUILD_TEST=1
# 运行测试时把 .profraw 落到固定目录（%p=pid, %m=二进制名）
# 若用 oss_coverage.py，它也会自行设置；手动跑测试时建议显式设置
export LLVM_PROFILE_FILE="profile/pytorch-%p-%m.profraw"
python3 setup.py build --cmake-only
ls
git submodule update --init --recursive
python3 setup.py build --cmake-only
cd build
ls
cmake ..   -DUSE_CPP_CODE_COVERAGE=ON   -DBUILD_TEST=ON   -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_C_COMPILER=clang   -DCMAKE_CXX_COMPILER=clang++   -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld"   -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
cmake --build . -j"$(nproc)"
cd ..
python setup.py develop
cd ..
ls
export LLVM_PROFILE_FILE="pytorch_%p.profraw"
python3 torch_fuzz.py corpus -runs=0  -print_final_stats=1
ls
llvm-profdata merge -sparse pytorch_*.profraw -o pytorch.profdata
ls
cd pytorch-cov-src/
ls
cd torch
ls
cd lib
ls
cd ..
ls
PD=pytorch.profdata
PRIMARY=/root/pytorch-cov-src/torch/lib/libtorch_cpu.so
OBJ_C10=/root/pytorch-cov-src/torch/lib/libc10.so
OBJ_PY=/root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so
MAP_OPT="-path-equivalence=/root/pytorch-cov-src,."
ATEN_FILES=$(bash -c '
  cd /root/pytorch-cov-src || exit 1
  find torch \( \
  	  -path "*/c10/*" -o \
  	  -path "*/cpuinfo/*" -o \
      -path "*/autograd/*" -o \
    \) -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cuh" -o -name "*.h" \) \
  | sed "s#^#/root/pytorch/#"
')
ATEN_FILES=$(bash -c '
  cd /root/pytorch-cov-src || exit 1
  find torch \( -path "*/c10/*" -o -path "*/cpuinfo/*" -o -path "*/autograd/*" \) -type f \
    \( -name "*.cpp" -o -name "*.cc" -o -name "*.cuh" -o -name "*.h" \) \
  | sed "s#^#/root/pytorch-cov-src/#"
')
llvm-cov report "$PRIMARY" -object "$OBJ_C10" -object "$OBJ_PY"   --instr-profile="$PD" $MAP_OPT $ATEN_FILES | tail -n +1
echo $LD_PRELOAD
cp -a ~/.bashrc ~/.bashrc.bak.$(date +%s)
# 追加一段（若已写过就不重复）
grep -q 'asan_with_fuzzer.so' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
# Atheris fuzzing: preload ASan+libFuzzer runtime
export LD_PRELOAD="/root/pytorch_cov/lib/python3.10/site-packages/asan_with_fuzzer.so"
EOF

. ~/.bashrc
cat ~/.bashrc
source pytorch_cov/bin/actviate
ls
cd  pytorch_cov/bin/
ls
cd ..
source pytorch_cov/bin/activate
echo $LD_PRELOAD
exit
cd root/
ls
source pytorch_cov/bin/activate
echo $LD_PRELOAD
exit
ls 
cd rott
cd root
python3 -m venv pytorch_fuzz
ls
source pytorch_fuzz/bin/activate
cd pytorch
pip install -r requirements.txt 
cat >/usr/local/bin/clangxx-noprofile <<'SH'
#!/usr/bin/env bash
REAL="$(command -v clang++)"   # 用系统当前的 clang++，不写死 17
is_shared=0
for a in "$@"; do
  [[ "$a" == "-shared" ]] && is_shared=1
done

if (( is_shared )); then
  args=()
  for a in "$@"; do
    [[ "$a" == "-fprofile-instr-generate" ]] && continue
    args+=("$a")
  done
  exec "$REAL" "${args[@]}"
else
  exec "$REAL" "$@"
fi
SH

chmod +x /usr/local/bin/clangxx-noprofile
export CC=clang
export CXX=/usr/local/bin/clangxx-noprofile
# export CXX=clang++
# 强制 setuptools/distutils 用 clang 链接共享库
export LDSHARED="clang -shared"
export LDCXXSHARED="/usr/local/bin/clangxx-noprofile -shared"
# 覆盖率（保持你原有设置）这个-fprofile-instr-generate -fcoverage-mapping应该可以不用
export CFLAGS="-O0 -g -fsanitize=fuzzer-no-link"
export CXXFLAGS="-O0 -g -fsanitize=fuzzer-no-link"
# 依然阻止共享库带 profile runtime；可执行文件带
export CMAKE_ARGS="-DCMAKE_SHARED_LINKER_FLAGS='-fno-profile-instr-generate' -DCMAKE_EXE_LINKER_FLAGS='-fprofile-instr-generate'"
# 可选精简（保持不变）
export USE_CUDA=0 BUILD_TEST=0 USE_DISTRIBUTED=0 USE_XNNPACK=0 USE_QNNPACK=0 USE_MKLDNN=0
python3 -m pip install --no-build-isolation -v -e .
export ASAN_OPTIONS=detect_leaks=0
export LD_PRELOAD="$(python3 -c 'import atheris, os; print(os.path.join(atheris.path(), "asan_with_fuzzer.so"))')"
ls -l "$LD_PRELOAD"   # 应该能看到 asan_with_fuzzer.so
pip install atheris
export ASAN_OPTIONS=detect_leaks=0
export LD_PRELOAD="$(python3 -c 'import atheris, os; print(os.path.join(atheris.path(), "asan_with_fuzzer.so"))')"
ls -l "$LD_PRELOAD"   # 应该能看到 asan_with_fuzzer.so
cd ..
ls
python3 fuzz_pytorch.py -atheris_runs=5000
vim fuzz_pytorch.py 
python3 fuzz_pytorch.py -atheris_runs=5000
ls
mkdir crash
ls
cd pytorch
ls
cd torch
ls
cd autograd/
ls
cd ..
ls
cd cpu/
ls
cd amp
ls
cd ..
ls
cd ..
ls
python3 -m coverage run --branch torch_fuzz.py.py corpus -artifact_prefix=crash -atheris_runs=20000
pip install coverage
python3 -m coverage run --branch torch_fuzz.py.py corpus -artifact_prefix=crash -atheris_runs=20000
python3 -m coverage run --branch torch_fuzz.py corpus -artifact_prefix=crash -atheris_runs=20000
python3 py_cov_filter_report.py   --cov .coverage   --include '^.*/torch/_tensor\.py$'   --include '^.*/torch/functional\.py$'   --include '^.*/torch/autograd/.*\.py$'   --include '^.*/torch/nn/functional\.py$'   --term 
python3 -m pip show torch
python3 torch_fuzz.py  -atheris_runs=2000
ls
exit
ls
cd root/
ls
vim cov_overlap.py
exit
ls
cd root/
ls
cd torch-api-fuzz/
ls
cd fft.fft/
ls
cd ..
ls
cd ..
ls
mkdir cov-tool
mv py_cov_filter_report.py cov-tool/
mv python_cov_overlap.py cov-tool/
mv C_cov_overlap.py cov-tool/
ls
cd cov-tool/
ls
cd ..
ls
cd C_1/
ls
cd ..
ls
rm C_1
rm -r C_1
rm -r C_2/
ls
rm run1.json 
rm run2.json 
rm run1.profdata 
rm run2.profdata 
ls
cd pytorch_cov/
ls
cd ..
cd corpus
LS
ls
cd ..
ls
rm -r corpus
rm -r corpus_comparison/
ls
cd crash/
ls
cd ..
rm -r crash/
ls
rm default.profraw 
ls
mkdir harness_test
mv torch_fuzz.py harness_test/
mv fuzz_pytorch.py harness_test/
ls
la
rm .coverage
rm .coverage.run1 
rm .coverage.run2 
ls
rm .bashrc.bak.1761801798 
ls
la
tree torch-api-fuzz/
apt-get update
apt-get install tree
tree torch-api-fuzz/
ls
cd select-tools/
ls
python3 fuzz_harness.py --root /root/torch-api-fuzz --fuzz_rounds 50000 --jobs 8
ls
cd ..
ls
cd torch-api-fuzz/
ls
cd fft.fft/
LS
ls
cd corpus_1
ls
cd ..
tree torch-api-fuzz/
cd torch-api-fuzz/
ls
rm fuzz_summary.json 
ls
cd fft.fft/
ls
python3 harness1.py 
cd ..
ls
source pytorch_fuzz/bin/activate
ls
cd select-tools/
ls
python3 fuzz_harness.py --root /root/torch-api-fuzz --fuzz_rounds 50000 --jobs 8
cd ..
ls
cd torch-api-fuzz/
ls
cd linalg.svd/
ls
cd corpus_1
ls
cd ..
ls
cd logs/
ls
cat harness1.log 
cd ..
ls
cd ..
cd select-tools/
cd ..
ls
mkdir fuzz_output
ls
cd torch-api-fuzz/
ls
rm fuzz_summary.json 
ls
cd fft.fft/
ls
rm -r logs/
cd ..
cd linalg.svd/
ls
rm -r logs/
cd ..
ls
cd matmul/
ls
rm -r logs/
cd ..
cd nn.functional.conv2d/
ls
rm -r logs/
cd ..
ls
cd where/
ls
rm -r logs/
ls
cd ..
ls
cd mm
cd ..
ls
cd torch-api-fuzz/
ls
cd fft.fft/
ls
rm -r corpus_3
rm -r corpus_1
rm -r crash_4
ls
rm -r crash_5
cd ..
ls
cd linalg.svd/
ls
rm -r corpus_5
rm -r corpus_2
rm -r corpus_4
ls
rm -r crash_5
rm -r crash_3
rm -r crash_2
ls
cd ..
cd matmul/
rm -r crash_2
rm -r crash_5
rm -r corpus_3
rm -r corpus_4
rm -r corpus_1
ls
cd ..
ls
cd nn.functional.conv2d/
rm -r corpus_4
rm -r corpus_2
rm -r crash_3
rm -r crash_5
cd ..
cd where/
rm -r corpus_1
rm -r corpus_2
rm -r corpus_5
rm -r crash_4
rm -r crash_2
ls
cd ..
ls
cd ..
tree -L 3 torch-api-fuzz/
tree -L 2 torch-api-fuzz/
cd select-tools/
python3 fuzz_harness.py --root /root/torch-api-fuzz --fuzz_rounds 20000 --jobs 10
ls
cd ..
ls\
ls
cd fuzz_output/
ls
deactivate 
ls
cd ..
ls
dpkg-reconfigure tzdata
exit
date
ls
cd root/
ls
tree -L 2 torch-api-fuzz/
mkdir cov-result
ls
cd select-tools/
ls
vim replay_coverage.py
cd ..
source pytorch_cov/bin/activate
ls
cd select-tools/
python3 replay_coverage.py   --fuzz-root /root/torch-api-fuzz   --out-root /root/cov-result   --workers 8
pip install coverage
python3 replay_coverage.py   --fuzz-root /root/torch-api-fuzz   --out-root /root/cov-result   --workers 8
cd ..
ls
cd cov-result/
ls
cd result_20251106_133400/
ls
cd fft.fft/
ls
la
llvm-profdata merge -sparse harn_1.profraw -o harn_1.profdata
llvm-profdata merge -sparse harn_2.profraw -o harn_2.profdata
ls
PD1=harn_1.profdata
PD2=harn_2.profdata
PRIMARY=/root/pytorch-cov-src/torch/lib/libtorch_cpu.so
OBJ_C10=/root/pytorch-cov-src/torch/lib/libc10.so
OBJ_PY=/root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so
MAP_OPT="-path-equivalence=/root/pytorch-cov-src,."
ATEN_FILES_FFT=$(bash -c '
  cd /root/pytorch-cov-src || exit 1

  # 1) CPU/通用：SpectralOps & 工具
  {
    find aten/src/ATen/native -maxdepth 1 -type f \
      \( -name "SpectralOps.cpp" -o -name "SpectralOpsUtils.h" \)

    # 2) CUDA：fft 实现、工具与计划缓存（cuFFT）
    find aten/src/ATen/native/cuda -type f \
      \( -name "SpectralOps.cu" -o -name "SpectralOps.cpp" \
         -o -name "CuFFTPlanCache.h" -o -name "CuFFTUtils.h" \
         -o -name "*.cuh" \)
  } \
  | sed "s#^#/root/pytorch-cov-src/#"
')
export_cov () {   local prof=$1 out=$2;   llvm-cov export "$PRIMARY"     -object "$OBJ_C10" -object "$OBJ_PY"     -instr-profile "$prof"     $MAP_OPT     -format=text     $ATEN_FILES     > "$out"; }
export_cov "$PD1" run1.json
export_cov "$PD2" run2.json
python3 /root/cov-tools/C_cov_overlap.py run1.json run2.json 
ls
cd ..
ls
la
cd select-tools/
la
cd ..
cd select-tools/
ls
python3 replay_coverage.py   --fuzz-root /root/torch-api-fuzz   --out-root /root/cov-result   --workers 8
cd ..
cd torch-api-fuzz/
ls
cd fft.fft/
/root/pytorch_cov/bin/python3 -m coverage --data-file=/root/cov-result/result_20251106_140512/fft.fft/.coverage.1 run --branch harness1.py corpus_1 -runs=0 -print_final_stats=1
/root/pytorch_cov/bin/python3 -m coverage run  --data-file=/root/cov-result/result_20251106_140512/fft.fft/.coverage.1 --branch harness1.py corpus_1 -runs=0 -print_final_stats=1
ls
la
/root/pytorch_cov/bin/python3 -m coverage run  --branch harness1.py corpus_1 -runs=0 -print_final_stats=1
la
cd ..
cd select-tools/
ls
cp replay_coverage.py relay_coverage.py.bak
ls
python3 replay_coverage.py   --fuzz-root /root/torch-api-fuzz   --out-root /root/cov-result   --workers 8
cd ..
ls
deactivate 
source pytorch_fuzz/bin/activate
ls
cd harness_test/
ls
mkdir corpus
python3 torch_fuzz.py corpus/ -atheris_runs=10000
ls
cd corpus/
ls
cd ..
ls
cd ..
deactivate 
source pytorch_cov/bin/activate
ls
cd harness_test/
ls
python3 -m coverage run --branch torch_fuzz.py corpus/ -runs=0
ls -la
ls
la
cd ..
ls
cd select-tools/
python3 replay_coverage.py   --fuzz-root /root/torch-api-fuzz   --out-root /root/cov-result   --workers 8
python3 replay_coverage.py   --fuzz-root /root/torch-api-fuzz   --out-root /root/cov-result   --workers 16
cd ..
ls
cd cov-result/
ls
cd result_20251106_150928/
ls
cd fft.fft/
ls
la
python3 /root/cov-tools/py_cov_filter_report.py   --cov .coverage.1   --include '^.*/torch/fft/__init__\.py$'   --include '^.*/torch/fft/_fft\.py$'   --include '^.*/torch/fft/_helper.*\.py$'   --include '^.*/torch/_refs/fft\.py$'   --include '^.*/torch/amp/autocast_mode\.py$'   --include '^.*/torch/_tensor\.py$'   --include '^.*/torch/functional\.py$'   --include '^.*/torch/overrides\.py$'   --term
ls
la
cd ..
ls
cd ..
ls
deactivate 
ls
cd fuzz_output/
ls
cd ..
ls
cd cov-result/
ls
cd result_20251106_150928/
ls
cd fft.fft/
ls
cat harn_1.log 
exit
clear
cd pytorch
find . -name native_functions.yaml
ls
cd root/
ls
cd torch-api-fuzz/
ls
cd ..
ls
cd select-tools/
ls
cd ..
ls
mkdir fuzz_api_one
cd fuzz_api_one/
ls
vim param_sampler.py
vim test_sampler_conv2d.py
python3 test_sampler_conv2d.py 
vim conv2d.yaml
vim template.py
vim generate_from_yaml.py
python3 generate_from_yaml.py conv2d.yaml 
python3 generate_from_yaml.py 
python3 auto_torch_fft_fft.py 
ls
vim linalg.yaml
pyhton3 generate_from_yaml.py 
python3 generate_from_yaml.py 
python3 auto_torch_linalg_svd.py 
python3 auto_torch_fft_fft.py 
pythone matmul.yaml 
python3 matmul.yaml 
python3 generate_from_yaml.py 
python3 auto_torch_matmul.py 
python3 generate_from_yaml.py 
python3 auto_torch_where.py 
python3 generate_from_yaml.py 
python3 auto_torch_where.py 
python3 generate_from_yaml.py 
python3 auto_torch_where.py 
python3 generate_from_yaml.py 
python3 auto_torch_where.py 
python3 generate_from_yaml.py 
python3 auto_torch_where.py 
python3 generate_from_yaml.py 
python3 auto_torch_linalg_svd.py 
python3 generate_from_yaml.py 
python3 auto_torch_linalg_svd.py 
python3 generate_from_yaml.py 
python3 auto_torch_linalg_svd.py 
python3 generate_from_yaml.py 
python3 auto_torch_linalg_svd.py 
python3 generate_from_yaml.py 
python3 auto_torch_linalg_svd.py 
python3 generate_from_yaml.py 
python3 auto_torch_matmul.py 
vim fuzz_sampler_fft.py
vim gen_harn_yaml.py
python3 gen_harn_yaml.py 
python3 gen_harn_yaml.py --out matmul_test.py
python3 matmul_test.py 
cd ..
source pytorch_fuzz/bin/activate
cd fuzz_api_one/
python3 matmul_test.py 
python3 gen_harn_yaml.py --out where_test.py
python3 where_test.py 
python3 gen_harn_yaml.py --yaml fft.yaml --out fft_test.py
python3 fft_test.py 
python3 gen_harn_yaml.py --yaml conv2d.yaml --out conv2d_test.py
python3 conv2d_test.py 
exit
ls
cd fuzz_api_one/
ls
python3 generate_from_yaml.py conv2d.yaml 
python3 generate_from_yaml.py 
python3 auto_torch_nn_functional_conv2d.py 
vim fft.yaml
ls
python3 generate_from_yaml.py --yaml fft.yaml 
python3 generate_from_yaml.py 
python3 auto_torch_fft_fft.py 
vim matmul.yaml
vim where.yaml
ls
cd root/
ls
cd pytorch
ls
cd aten
ls
src/
exit
ls
cd root/
cd torch-api-fuzz/
ls
python3 attention_score_block.py 
ls
python3 attention_score_block.py 
cd ..
source pytorch_fuzz/bin/activate
cd torch-api-fuzz/
python3 attention_score_block.py 
ls
python3 mul.py 
exit
ls
cd root/
ls
docker stop ydl_fuzz
mkdir fuzz_api_seq
ls
cd fuzz_api_seq/
ls
vim seq_conv2d.py
python3 seq_conv2d.py 
cd ..
LS
ls
cd pytorch_fuzz/bin/activate
source pytorch_fuzz/bin/activate
cd fuzz_api_seq/
python3 seq_conv2d.py 
vim conv2d.py
python3 conv2d.py 
python3 seq_conv2d.py -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1 -atheris_runs=50000
python3 conv2d.py -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1
ls
vim conv_bn_relu_pool_block.yaml
vim mlp_block.yaml
vim attention_score_block.yaml
vim add_norm_block.yaml
vim elementwise_where_mix_block.yaml
ls
cd ..
ls
cd fuzz_api_one/
ls
vim batch_norm.yaml
vim relu.yaml
vim max_pool2d.yaml
vim linear.yaml
vim dropout.yaml
vim softmax.yaml
vim layer_norm.yaml
vim add.yaml
vim mul.yaml
vim sum.yaml
ls
python3 gen_harn_yaml.py --yaml add.yaml 
python3 auto_torch_add.py 
python3 gen_harn_yaml.py --yaml add.yaml 
python3 auto_torch_add.py 
python3 gen_harn_yaml.py --yaml batch_norm.yaml 
python3 auto_torch_nn_functional_batch_norm.py 
python3 gen_harn_yaml.py --yaml dropout.yaml 
python3 auto_torch_nn_functional_dropout.py 
python3 gen_harn_yaml.py --yaml layer_norm.yaml 
python3 auto_torch_nn_functional_layer_norm.py 
python3 gen_harn_yaml.py --yaml linear.yaml 
python3 auto_torch_nn_functional_linear.py 
python3 gen_harn_yaml.py --yaml max_pool2d.yaml 
python3 auto_torch_nn_functional_max_pool2d.py 
python3 gen_harn_yaml.py --yaml relu.yaml 
python3 auto_torch_nn_functional_relu.py 
python3 gen_harn_yaml.py --yaml softmax.yaml 
python3 auto_torch_nn_functional_softmax.py 
python3 gen_harn_yaml.py --yaml sum.yaml 
python3 auto_torch_sum.py 
ls
vim gen_harness.py
vim param_sampler.py
ls\
ls
mkdir api-yaml
ls
cd api-yaml/
ls
cd ..
ls
cd ..
ls
cd fuzz_api_seq/
ls
vim seq_env.py
ls
vim api_registry.py
ls
vim gen_seq_harness.py
python3 gen_seq_harness.py --seq_yaml add_norm_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/
vim param_sampler.py
python3 auto_seq_add_norm_block.py 
ls
mkdir seq-yaml
ls
mv test/ rl-test
ls
python3 gen_seq_harness.py --seq_yaml seq-yaml/attention_score_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/ --out /root/torch-api-fuzz/
python3 gen_seq_harness.py --seq_yaml seq-yaml/attention_score_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/ --out /root/torch-api-fuzz/attention_score_block.py
cd ..
ls
cd fuzz_api_one
LS
ls
python3 gen_harness.py --yaml api-yaml/mul.yaml --out /root/torch-api-fuzz/mul.py
exit'
exit
cd root/
cd fuzz_api_seq/
ls
cd seq-yaml/
ls
cd ..
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/attention_score_block.yaml     --api_yaml_dir /root/fuzz_api_one/api-yaml     --out /root/torch-api-fuzz/auto_seq_attention_score_block.yaml.py
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/attention_score_block.yaml     --api_yaml_dir /root/fuzz_api_one/api-yaml     --out /root/torch-api-fuzz/auto_seq_attention_score_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/conv_bn_relu_pool_block.yaml     --api_yaml_dir /root/fuzz_api_one/api-yaml     --out /root/torch-api-fuzz/auto_seq_conv_bn_relu_pool_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/elementwise_where_mix_block.yaml     --api_yaml_dir /root/fuzz_api_one/api-yaml     --out /root/torch-api-fuzz/auto_seq_elementwise_where_mix_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/mlp_block.yaml     --api_yaml_dir /root/fuzz_api_one/api-yaml     --out /root/torch-api-fuzz/auto_seq_mlp_block.py
exit
ls
cd root/
ls
source pytorch_fuzz/bin/activate
ls
cd fuzz_api_seq/
ls
cd seq-yaml/
ls
cd ..\
cd ..
ls
vim gen_seq_harness_oracle.py
python3 generate_sequence_harness_oracle.py     --seq_yaml ./seq_yaml/add_norm_block_oracle.yaml     --api_yaml_dir /root/fuzz_api_one/api_yaml     --out /root/torch-api-fuzz/auto_seq_add_norm_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml ./seq_yaml/add_norm_block_oracle.yaml     --api_yaml_dir /root/fuzz_api_one/api_yaml     --out /root/torch-api-fuzz/auto_seq_add_norm_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml ./seq_yaml/add_norm_block.yaml     --api_yaml_dir /root/fuzz_api_one/api_yaml     --out /root/torch-api-fuzz/auto_seq_add_norm_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml seq_yaml/add_norm_block.yaml     --api_yaml_dir /root/fuzz_api_one/api_yaml     --out /root/torch-api-fuzz/auto_seq_add_norm_block.py
ls
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq_yaml/add_norm_block.yaml     --api_yaml_dir /root/fuzz_api_one/api_yaml     --out /root/torch-api-fuzz/auto_seq_add_norm_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/add_norm_block.yaml     --api_yaml_dir /root/fuzz_api_one/api_yaml     --out /root/torch-api-fuzz/auto_seq_add_norm_block.py
python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/add_norm_block.yaml     --api_yaml_dir /root/fuzz_api_one/api-yaml     --out /root/torch-api-fuzz/auto_seq_add_norm_block.py

cd ..
cd torch-api-fuzz/
ls
python3 auto_seq_add_norm_block.py 
python3 auto_seq_add_norm_block.py -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1
python3 auto_seq_attention_score_block.py -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1
python3 auto_seq_conv_bn_relu_pool_block.py -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1
python3 auto_seq_python3 gen_seq_harness_oracle.py     --seq_yaml /root/fuzz_api_seq/seq-yaml/mlp_block.yaml     --api_yaml_dir /root/fuzz_api_one/api-yaml     --out /root/torch-api-fuzz/auto_seq_mlp_block.p -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1
python3 auto_seq_mlp_block.py -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1
exit
cd root/
ls
cd fuzz_api_seq/
ls
python3 gen_seq_harness.py --yaml seq-yaml/mlp_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/ --out /root/torch-api-fuzz/auto_seq_mlp_block.py
python3 gen_seq_harness.py --seq_yaml seq-yaml/mlp_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/ --out /root/torch-api-fuzz/auto_seq_mlp_block.py
python3 gen_seq_harness.py --seq_yaml seq-yaml/conv_bn_relu_pool_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/ --out /root/torch-api-fuzz/auto_seq_conv_bn_relu_pool_block.py
ls
vim oracle_runtime.py
python3 gen_seq_harness.py --seq_yaml seq-yaml/conv_bn_relu_pool_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/ --out /root/torch-api-fuzz/auto_seq_conv_bn_relu_pool_block.py
ls
exit
ls
cd root
ls
ssh-keygen -t ed25519 -C "481918729@qq.com"
cat ~/.ssh/id_ed25519.pub
git init
vim .gitignore
la
git add .
git commit -m "init commit"
git remote add origin git@github.com:yyds1233/Dl_fuzzer.git
git push
git push --set-upstream origin master
git add .
git commit -C "first commit"
git add .
git commit
git push --set-upstream origin master
rm -rf .git
git init
vim .gitignore
git add .
rm -rf .git
git init
git add .
rm -rf .git
git init
git add .
rm -rf .git
git init
git add .
git commit 
git push
git remote add git@github.com:yyds1233/Dl_fuzzer.git
git remote add origin git@github.com:yyds1233/Dl_fuzzer.git
git push
git push --set-upstream origin master
git checkout -b main
git push
git push --set-upstream origin main
git pull origin main --rebase
git add .
git commit
git push
git push --set-upstream origin main
git push --set-upstream origin main -f
git branch -d master
git push origin --delete master
ls
source pytorch_fuzz/bin/activate
ls
cd fuzz_api_seq/
ls
cd ..
ls
cd torch-api-fuzz/
python3 auto_seq_add_norm_block.py 
python3 auto_seq_attention_score_block.py 
python3 auto_seq_conv_bn_relu_pool_block.py 
python3 auto_seq_elementwise_where_mix_block.py 
python3 auto_seq_mlp_block.py 
python3 auto_seq_conv_bn_relu_pool_block.py 
ls
cd ..
ls
cd fuzz_output/
ls
git add .
git commit
git push
ls
vim rl_conv_bn_relu_pool_block.py
python3 rl_conv_bn_relu_pool_block.py 
python3 rl_conv_bn_relu_pool_block.py --atheris_runs=20000
python3 auto_seq__conv_bn_relu_pool_block.py --atheris_runs=20000
python3 auto_seq_conv_bn_relu_pool_block.py --atheris_runs=20000
exit
ls
cd root/
ls
cd fuzz_output/
ls
python3 auto_conv2d_mutation.py 
cd ..
source pytorch_fuzz/bin/activate
cd fuzz_output/
python3 auto_conv2d_mutation.py 
python3 auto_conv2d_mutation.py --atheris_runs=20000
exit
ls
cd root/
ls
source pytorch_fuzz/bin/activate
ls
cd fuzz_api_one/
python3 gen_harness.py --yaml /root/fuzz_api_one/api-yaml/conv2d.yaml --out /root/fuzz_output/
python3 gen_harness.py --yaml /root/fuzz_api_one/api-yaml/conv2d.yaml --out /root/fuzz_output/auto_conv2d.py
cd ..
cd fuzz_output/
ls
python3 auto_conv2d.py 
vim print_value_conv2d.py
python3 print_value_conv2d.py 
ls
cd ..
ls
cd fuzz_api_one/
ls
python3 gen_harness.py --yaml /root/fuzz_api_one/api-yaml/conv2d.yaml --out /root/fuzz_output/auto_conv2d_mutation.py
cd ..
ls
python3 auto_conv2d.py 
fuzz_output/
cd fuzz_output/
python3 auto_conv2d.py 
python3 auto_conv2d.py --atheris_runs=20000
exit
git add .
git commit
git push
git add .
git commit 
ls
cd root/
ls
source pytorch_fuzz/bin/activate
ls
cd fuzz_api_seq/
ls
python3 gen_seq_harness.py --seq_yaml seq-yaml/conv_bn_relu_pool_block.yaml --api_yaml_dir /root/fuzz_api_one/api-yaml/ --out /root/fuzz_output/auto_seq_conv_mutation.py
cd ..
cd fuzz_output/
ls
python3 auto_seq_conv_mutation.py 
exit
cd root/
ls
source pytorch_fuzz/bin/activate
ls
cd fuzz_output/
ls
python3 auto_conv2d_mutation.py 
ls
clear
cd ..
ls
tree -L 2
tree -L 3
tree -L 2
ls
cd fuzz_api_one/
ls
python3 gen_harness.py --yaml api-yaml/add.yaml --out /root/fuzz_output/auto_add.py
python3 gen_harness.py --yaml api-yaml/batch_norm.yaml --out /root/fuzz_output/auto_batch.py
python3 gen_harness.py --yaml api-yaml/dropout.yaml --out /root/fuzz_output/auto_dropout.py
python3 gen_harness.py --yaml api-yaml/fft.yaml --out /root/fuzz_output/auto_dropout.py
python3 gen_harness.py --yaml api-yaml/dropout.yaml --out /root/fuzz_output/auto_dropout.py
python3 gen_harness.py --yaml api-yaml/fft.yaml --out /root/fuzz_output/auto_fft.py
python3 gen_harness.py --yaml api-yaml/layer_norm.yaml --out /root/fuzz_output/auto_layer.py
python3 gen_harness.py --yaml api-yaml/linalg.yaml --out /root/fuzz_output/auto_linalg.py
python3 gen_harness.py --yaml api-yaml/linear.yaml --out /root/fuzz_output/auto_linear.py
python3 gen_harness.py --yaml api-yaml/matmul.yaml --out /root/fuzz_output/auto_matmul.py
python3 gen_harness.py --yaml api-yaml/max_pool2d.yaml --out /root/fuzz_output/auto_maxpool2d.py
python3 gen_harness.py --yaml api-yaml/mul.yaml --out /root/fuzz_output/auto_mul.py
python3 gen_harness.py --yaml api-yaml/relu.yaml --out /root/fuzz_output/relu.py
python3 gen_harness.py --yaml api-yaml/relu.yaml --out /root/fuzz_output/auto_relu.py
python3 gen_harness.py --yaml api-yaml/softmax.yaml --out /root/fuzz_output/auto_softmax.py
python3 gen_harness.py --yaml api-yaml/sum.yaml --out /root/fuzz_output/auto_sum.py
python3 gen_harness.py --yaml api-yaml/where.yaml --out /root/fuzz_output/auto_where.py
cd ..
tree -L 2
cd fuzz_output/
ls
mkdir Corpus
mkdir Crash
tree -L 2
cd ..
tree -L 2
ls
cd fuzz_output/
ls
vim screen_single_api.py
python3 screen_single_api.py   --root fuzz_output   --harness fuzz_output/auto_conv2d.py   --epoch 60   --n_profiles 60   --score_mode ft
python3 screen_single_api.py   --root fuzz_output   --harness auto_conv2d.py   --epoch 60   --n_profiles 60   --score_mode ft
ls
vim bandit_audit_driver.py
vim cov_global_union_audit.py
ls
mkdir bandit_corpus
cd bandit_corpus/
ls
mkdir 3449d6ea62
ls
cd ..
ls
python3 auto_conv2d.py bandit_corpus/3449d6ea62 -artifact_prefix=crash_conv2d -ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1
ls
cd bandit_corpus/
ls
cd 3449d6ea62/
ls
cd ..
ls
pwd
cd bandit_corpus/
ls
pwd
cd ..
ls
cd ..
source pytorch_cov/bin/activate
ls
cd fuzz_output/
ls
python3 /root/fuzz_output/cov_global_union_audit.py   --python python3   --harness /root/fuzz_output/auto_conv2d.py   --corpus  /root/fuzz_output/bandit_corpus/3449d6ea62   --work_dir /root/fuzz_output/audits/conv2d_smoke   --global_dir /root/fuzz_output/global_union   --primary_object /root/pytorch-cov-src/torch/lib/libtorch_cpu.so   --extra_object /root/pytorch-cov-src/torch/lib/libc10.so   --extra_object /root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so   --ignore_filename_regex ".*(site-packages|third_party|build).*"
ls
cd audits/
ls
cd conv2d_smoke/
ls
cd ..
ls
rm -rf audits/
ls
rm -rf global_union/
ls
python3 /root/fuzz_output/cov_global_union_audit.py   --python python3   --harness /root/fuzz_output/auto_conv2d.py   --corpus  /root/fuzz_output/bandit_corpus/3449d6ea62   --work_dir /root/fuzz_output/audits/conv2d_smoke   --global_dir /root/fuzz_output/global_union   --primary_object /root/pytorch-cov-src/torch/lib/libtorch_cpu.so   --extra_object /root/pytorch-cov-src/torch/lib/libc10.so   --extra_object /root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so   --ignore_filename_regex ".*(site-packages|third_party|build).*"
ls
cd fuzz_output/
ls
pwd
ls
cd ..
ls
rm -rf audits/
ls
rm -rf global_union/
ls
python3 /root/fuzz_output/bandit_audit_driver.py   --harness /root/fuzz_output/auto_conv2d.py   --top_json /root/fuzz_output/fuzz_output/screen_runs/conv2d/round1_top_results.json   --root /root/fuzz_output_test   --epoch 10   --steps 6   --audit_every 2   --cov_audit_script /root/fuzz_output/cov_global_union_audit.py   --cov_venv_activate /root/pytorch_cov/bin/activate   --primary_object /root/pytorch-cov-src/torch/lib/libtorch_cpu.so   --extra_object /root/pytorch-cov-src/torch/lib/libc10.so   --extra_object /root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so   --ignore_filename_regex ".*(site-packages|third_party|build).*"   --audit_max_inputs 200   --slow_metric BRH
git add .
git commit
git push
eixt
exit
ls
cd root/
ls
source pytorch_fuzz/
source pytorch_fuzz/bin/activate
ls
cd fuzz_output/
ls
cd Corpus/
ls
cd ..
cd fuzz_output/
ls
cd Corpus/
ls
cd conv2d/
ls
cd d868080f8b/
ls
cd ..
ls
cd ..
ls
cd screen_runs/
ls
cd conv2d/
ls
cd 1757c1a18f/
ls
cat round1_fuzzer.log 
cd ..
ls
cd .
cd ..
ls
cd ..
ls
cd C
cd Crash/
ls
cd conv2d/
ls
cd ..
ls
cd Corpus/
ls
cd conv2d/
ls
exit
ls
cd root/
ls
source pytorch_fuzz/bin/activate
cd screen/
ls
python3 /root/screen/bandit_audit_driver_hier.py   --harnesses_json /root/screen/harnesses.json   --root /root/fuzz_output_test   --epoch 10   --steps 0   --audit_every 3   --cov_audit_script /root/screen/cov_global_union_audit.py   --cov_venv_activate /root/pytorch_cov/bin/activate   --primary_object /root/pytorch-cov-src/torch/lib/libtorch_cpu.so   --extra_object /root/pytorch-cov-src/torch/lib/libc10.so   --extra_object /root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so   --ignore_filename_regex ".*(site-packages|third_party|build).*"   --audit_max_inputs 200   --slow_metric BRH   --audit_profile_topk 5   --min_credit_inputs 30   --zero_slow_penalty 2.0
vim build_json.py
python3 /root/screen/make_harnesses_json.py   --out /root/fuzz_output/screen_runs/auto_harnesses.json   --harness seq_EWMB /root/fuzz_output/auto_seq_EWMB.py /root/fuzz_output/screen_runs/seq_EWMB/round1_top_results.json   --harness conv2d   /root/fuzz_output/auto_conv2d.py   /root/fuzz_output/screen_runs/conv2d/round1_top_results.json
python3 /root/screen/build_json.py   --out /root/fuzz_output/screen_runs/harnesses.json   --harness seq_EWMB /root/fuzz_output/auto_seq_EWMB.py /root/fuzz_output/screen_runs/seq_EWMB/round1_top_results.json   --harness conv2d   /root/fuzz_output/auto_conv2d.py   /root/fuzz_output/screen_runs/conv2d/round1_top_results.json
python3 /root/screen/build_json.py   --out /root/screen/auto_harnesses.json   --harness seq_EWMB /root/fuzz_output/auto_seq_EWMB.py /root/fuzz_output/screen_runs/seq_EWMB/round1_top_results.json   --harness conv2d   /root/fuzz_output/auto_conv2d.py   /root/fuzz_output/screen_runs/conv2d/round1_top_results.json
python3 /root/screen/bandit_audit_driver_hier.py   --harnesses_json /root/screen/auto_harnesses.json   --root /root/fuzz_output_test   --epoch 10   --steps 0   --audit_every 3   --cov_audit_script /root/screen/cov_global_union_audit.py   --cov_venv_activate /root/pytorch_cov/bin/activate   --primary_object /root/pytorch-cov-src/torch/lib/libtorch_cpu.so   --extra_object /root/pytorch-cov-src/torch/lib/libc10.so   --extra_object /root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so   --ignore_filename_regex ".*(site-packages|third_party|build).*"   --audit_max_inputs 200   --slow_metric BRH   --audit_profile_topk 5   --min_credit_inputs 30   --zero_slow_penalty 2.0
cd ..
git add .
git commit
git push
ls
cd build_yaml/
ls
vim doc_rank_extractor.py
python3 doc_rank_extractor.py   --api_name torch.nn.functional.batch_norm   --doc_txt api_txt/batch_norm.txt   --out_json api_txt/multi_rank_index.json   --debug
python3 doc_rank_extractor.py   --api_name torch.nn.functional.dropout2d   --doc_txt api_txt/dropout2d.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.group_norm   --doc_txt api_txt/group_norm.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.max_pool2d   --doc_txt api_txt/maxpool2d.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.group_norm   --doc_txt api_txt/group_norm.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.max_pool2d   --doc_txt api_txt/maxpool2d.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.batch_norm   --doc_txt api_txt/batch_norm.txt   --out_json api_txt/multi_rank_index.json   --debug
python3 doc_rank_extractor.py   --api_name torch.nn.functional.max_pool2d   --doc_txt api_txt/maxpool2d.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.dropout2d   --doc_txt api_txt/dropout2d.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.max_pool2d   --doc_txt api_txt/maxpool2d.txt   --out_json api_txt/multi_rank_index.json   --debug  --merge
git
python3 schema2yaml.py --schema_json api_schema/torch_nn_functional_batch_norm_schema.json --out_dir api_yaml_skeleton/ --rank_index_json api_txt/multi_rank_index.json 
python3 schema2yaml.py --schema_json api_schema/torch_sparse_csc_tensor_schema.json --out_dir api_yaml_skeleton/ --rank_index_json api_txt/multi_rank_index.json 
ls
python3 doc_rank_extractor.py   --api_name torch.nn.functional.batch_norm   --doc_txt api_txt/batch_norm.txt   --out_json api_txt/multi_rank_index.json   --debug   --merge
python3 doc_rank_extractor.py   --api_name torch.nn.functional.conv2d   --doc_txt api_txt/conv2d.txt   --out_json api_txt/multi_rank_index.json   --debug   --merge
exit
ls
cd root/
ls
cd fuzz_api_one/
ls
python3 gen_harness.py   --yaml /root/batch_norm_cons.yaml   --out_dir /root/fuzz_output
python3 gen_harness.py   --yaml /root/yaml/batch_norm_cons.yaml   --out_dir /root/fuzz_output
cd ..
source pytorch_fuzz/bin/activate
cd fuzz_output
python3 auto_torch.nn.functional.batch_norm__ov_default__rank2.py 
python3 auto_torch.nn.functional.batch_norm__ov_default__rank3
python3 auto_torch.nn.functional.batch_norm__ov_default__rank3.py 
python3 auto_torch.nn.functional.batch_norm__ov_default__rank4.py 
python3 auto_torch.nn.functional.batch_norm__ov_default__rank5.py 
cd ..
cd fuzz_api_one/
python3 gen_harness.py   --yaml /root/yaml/conv2d_default_cons.yaml   --out_dir /root/fuzz_output
cd ..
cd fuzz_output
python3 auto_torch.nn.functional.conv2d__ov_default__rank4.py 
cd ..
ls
git add .
git commit
git push
exit
ls 
ls
cd screen/
ls
tree
ls
cd root/
ls
source pytorch_fuzz/bin/activate
ls
cd screen/
ls
cd config/
ls
api_group.json
vim api_group.json
ls
cd ..
vim profile_space.py
ls
mkdir prior
cd prior/
vim __init__.py
vim group_prior.py
cd ..
ls
mkdir pool
cd pool/
ls
vim __init__.py
vim profile_pool.py
exit
ls
cd root/
source pytorch_cov/bin/activate
exit
ls
cd root
ls
source pytorch_cov/bin/activate
cd cov-tools/
ls
vim C_cov_overlap.py 
ls
cd ..
ls
exit
ls
cd screen/
tree
ls
cd root/
ls
git add .
git commit
git push
ssh -T git@github.com
git push
ls
exit
ls
cd root/
ls
source pytorch_fuzz/bin/activate
ls
vim test_issue.py
python3 test_issue.py 
vim test_issue2.py
python3 test_issue2.py 
cd pytorch
ls
cd main/torch/utils/
cd ..
curl -OL https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py
apt-get update
apt-get install vim
curl -OL https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py
apt-get install curl
curl -OL https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py
ls
python3 collect_env.py 
exit
ls
cd opt/
cd ..
cd root/
vim test_poc.py
source pytorch_fuzz/bin/activate
python3 test_poc.py 
rm test_poc.py 
vim test_poc.py
python3 test_poc.py 
exit
ls
cd root/
ls
exit
dos2unix /root/.bashrc
apt-get update
apt-get install dos2unix
dos2unix /root/.bashrc
exit
