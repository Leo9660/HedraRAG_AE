make -C build -j faiss
make -C build -j faiss_avx2
make -C build -j swigfaiss
cd build/faiss/python
pip install -e .