@echo off

rem Step 1: Compile the Cython code
echo Compiling Cython code...
C:\path\to\cythonize.exe -i main.pyx
rem Replace "C:\path\to\cythonize.exe" with the actual path to the cythonize executable

rem Step 2: Run the Python script
echo Running Python script...
python main.py

rem Pause to keep the command prompt window open (optional)
pause
