#include <CL/sycl.hpp>

using  namespace  cl::sycl;

int main(int argc, char **argv) {

	sycl::queue Q(sycl::gpu_selector_v); // Coge la gráfica (CUDA pilla la de NVIDIA y sin CUDA pilla la de Intel)
	//sycl::queue Q(sycle::cpu_selector_v); // Coge el procesador
	//sycl::queue Q(sycl::default_selector_v); // Coge la gráfica por defecto (CUDA pilla la de NVIDIA y sin CUDA pilla la de Intel)
	//sycl::queue Q(sycl::accelerator_selector_v); // No hay acelerador y da error


	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	Q.submit([&](handler &cgh) {
		// Create a output stream
		sycl::stream sout(1024, 256, cgh);
		// Submit a unique task, using a lambda
		cgh.single_task([=]() {
			sout << "Hello, World!" << sycl::endl;
		}); // End of the kernel function
	});   // End of the queue commands. The kernel is now submited

	// wait for all queue submissions to complete
	Q.wait();


  return 0;
}
