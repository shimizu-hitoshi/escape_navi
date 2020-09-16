import subprocess
subprocess.call(
"nvcc -shared -Xcompiler -fPIC  twd.cpp simulator.cpp main.cpp -o libsim.so -DDEBUG"
,shell=True)
subprocess.call(
"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./"
,shell=True)

