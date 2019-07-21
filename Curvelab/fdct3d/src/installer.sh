rm -r build/
rm _*.so
rm *wrap.cpp
python3 setup.py build
scp build/lib.*/* ./
python3 setup.py install
