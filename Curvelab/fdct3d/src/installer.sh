if [ "$1" == "swig" ]; then
	rm -r build/
	rm _*.so
	rm *wrap.cpp
	CC=g++ python3 setup.py build
	scp build/lib.*/* ./
	CC=g++ python3 setup.py install
fi

if [ "$1" == "cython" ]; then
	rm -r build/
	rm pycfdct3d*.so
	rm pycfdct3d.c
	rm pycfdct3d.cpp
	CC=g++ python3 setup.py build_ext --inplace
fi
