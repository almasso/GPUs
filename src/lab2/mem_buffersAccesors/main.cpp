#include <CL/sycl.hpp>

using  namespace  cl::sycl;

#define N 10

int main(int argc, char** argv) {
    sycl::queue Q(sycl::gpu_selector_v);

    std::cout << "Running on "
        << Q.get_device().get_info<sycl::info::device::name>()
        << std::endl;


    std::vector<float> a(N);

    for(int i=0; i<N; i++)
        a[i] = i; // Init a

    // Alternativa 1 -> Usamos un host_accesor
    //Create a submit a kernel
    buffer buffer_a{a}; //Create a buffer with values of array a

    // Create a command_group to issue command to the group
    Q.submit([&](handler &h) {
        accessor acc_a{buffer_a, h, read_write}; // Accessor to buffer_a

        // Submit the kernel
        h.parallel_for(N, [=](id<1> i) {
            acc_a[i]*=2.0f;
        }); // End of the kernel function
    }).wait();       // End of the queue commands we waint on the event reported.

    host_accessor a_(buffer_a, read_only);

    for(int i=0; i<N; i++)
        std::cout << "a[" << i << "] = " << a_[i] << std::endl;


    // Alternativa 2 -> Ponemos el buffer en un scope
    /*{
        //Create a submit a kernel
        buffer buffer_a{a}; //Create a buffer with values of array a

        // Create a command_group to issue command to the group
        Q.submit([&](handler &h) {
            accessor acc_a{buffer_a, h, read_write}; // Accessor to buffer_a

            // Submit the kernel
            h.parallel_for(N, [=](id<1> i) {
                acc_a[i]*=2.0f;
            }); // End of the kernel function
        }).wait();       // End of the queue commands we waint on the event reported.
    }

    for(int i=0; i<N; i++)
        std::cout << "a[" << i << "] = " << a[i] << std::endl;
    */
        
    return 0;
}
