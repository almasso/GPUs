CXX=icpx
CXXFLAGS=-fsycl -I. -fsycl-targets=nvptx64-nvidia-cuda,spir64
LIBS=

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: vector_add.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o
