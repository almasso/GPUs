/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".

    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <CL/sycl.hpp>

#include "GSimulation.hpp"

using namespace cl::sycl;

int main(int argc, char** argv)
{
  int N;			//number of particles
  int nstep; 		//number ot integration steps

  GSimulation sim;

  sycl::device dev;
  sycl::queue Q;

  if(argc>1)
  {
    N=atoi(argv[1]);
    sim.set_number_of_particles(N);
    if(argc>3)
    {
      nstep=atoi(argv[2]);
      sim.set_number_of_steps(nstep);

      if(argv[3][0] == 'c') {
          dev = sycl::device(sycl::cpu_selector_v);
          Q = sycl::queue(dev);
          std::cout << "Running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
          sim.setSyclQueue(Q);
          sim.start(true);
      }
      else if(argv[3][0] == 'g') {
          dev = sycl::device(sycl::gpu_selector_v);
          Q = sycl::queue(dev);
          std::cout << "Running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
          sim.setSyclQueue(Q);
          sim.start(true);
      }
      else {
          dev = sycl::device(sycl::cpu_selector_v);
         	Q = sycl::queue(dev);
          std::cout << "Running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
          sim.start();
      }
    }
  }
  return 0;
}
