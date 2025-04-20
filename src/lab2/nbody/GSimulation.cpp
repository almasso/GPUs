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

#include "GSimulation.hpp"
#include "cpu_time.hpp"
#include <CL/sycl.hpp>

using  namespace  cl::sycl;

#pragma region reducciones
constexpr int warm_up_token = -1;

class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

void flush_cache(sycl::queue &q, sycl::buffer<real_type> &flush_buf) {
  auto flush_size = flush_buf.get_size()/sizeof(real_type);
  auto ev = q.submit([&](auto &h) {
    sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::noinit);
    h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
  });
  ev.wait_and_throw();
}

// reductionAtomics1 de
// https://www.intel.com/content/www/us/en/developer/articles/technical/analyzing-performance-reduction-operations-dpc.html#gs.pplz86
void reductionAtomics1(sycl::queue &q, sycl::buffer<real_type> inbuf,
                     sycl::buffer<real_type> flush_buf, real_type &res, int iter) {
  const size_t data_size = inbuf.get_size()/sizeof(real_type);

  sycl::buffer<real_type> sum_buf(&res, 1);

  double elapsed = 0;
  for (int i = warm_up_token; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);

      h.parallel_for(1, [=](auto index) {
        sum_acc[0] = 0;
      });
    });

    flush_cache(q, flush_buf);
    Timer timer;
    // reductionAtomics1 main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::noinit);

      h.parallel_for(data_size, [=](auto index) {
        size_t glob_id = index[0];
        auto v =
            sycl::atomic_ref<real_type, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
                sum_acc[0]);
        v.fetch_add(buf_acc[glob_id]);
      });
      // reductionAtomics1 main end
    });
    q.wait();
    {
      // ensure limited life-time of host accessor since it blocks the queue
      sycl::host_accessor h_acc(sum_buf);
      res = h_acc[0];
    }
    // do not measure time of warm-up iteration to exclude JIT compilation time
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
} // end reductionAtomics1

#pragma endregion

GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

void GSimulation::setSyclQueue(sycl::queue Q) {
    _syclQueue = Q;
}

void GSimulation :: set_number_of_particles(int N)
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)
{
  set_nsteps(N);
}

void GSimulation :: init_pos()
{
  std::random_device rd;	//random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].pos[0] = unif_d(gen);
    particles[i].pos[1] = unif_d(gen);
    particles[i].pos[2] = unif_d(gen);
  }
}

void GSimulation :: init_vel()
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[2] = unif_d(gen) * 1.0e-3f;
  }
}

void GSimulation :: init_acc()
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].acc[0] = 0.f;
    particles[i].acc[1] = 0.f;
    particles[i].acc[2] = 0.f;
  }
}

void GSimulation :: init_mass()
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].mass = n * unif_d(gen);
  }
}

void GSimulation :: get_acceleration(int n)
{
   int i,j;

   const float softeningSquared = 1e-3f;
   const float G = 6.67259e-11f;

   for (i = 0; i < n; i++)// update acceleration
   {
     real_type ax_i = particles[i].acc[0];
     real_type ay_i = particles[i].acc[1];
     real_type az_i = particles[i].acc[2];
     for (j = 0; j < n; j++)
     {
         real_type dx, dy, dz;
	 real_type distanceSqr = 0.0f;
	 real_type distanceInv = 0.0f;

	 dx = particles[j].pos[0] - particles[i].pos[0];	//1flop
	 dy = particles[j].pos[1] - particles[i].pos[1];	//1flop
	 dz = particles[j].pos[2] - particles[i].pos[2];	//1flop

 	 distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	 distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

	 ax_i += dx * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
	 ay_i += dy * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
	 az_i += dz * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
     }
     particles[i].acc[0] = ax_i;
     particles[i].acc[1] = ay_i;
     particles[i].acc[2] = az_i;
   }
}

void GSimulation::get_acceleration_SYCL(int n) {
    float softeningSquared = 1e-3f;
    float G = 6.67259e-11f;

    _syclQueue.submit([&](handler &h) {
        ParticleAoS* local_particles = particles;
        h.parallel_for(range<2>(n, n), [=](id<2> item) {
            int i = item[0], j = item[1];
            real_type ax_i = local_particles[i].acc[0];
            real_type ay_i = local_particles[i].acc[1];
            real_type az_i = local_particles[i].acc[2];
            real_type distanceSqr = 0.0f;
            real_type distanceInv = 0.0f;
            real_type dx = local_particles[j].pos[0] - local_particles[i].pos[0];
            real_type dy = local_particles[j].pos[1] - local_particles[i].pos[1];
            real_type dz = local_particles[j].pos[2] - local_particles[i].pos[2];
            distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared;
            distanceInv = 1.0f / sqrtf(distanceSqr);
            ax_i += dx * G * local_particles[j].mass * distanceInv * distanceInv * distanceInv;
            ay_i += dy * G * local_particles[j].mass * distanceInv * distanceInv * distanceInv;
            az_i += dz * G * local_particles[j].mass * distanceInv * distanceInv * distanceInv;
            local_particles[i].acc[0] = ax_i;
            local_particles[i].acc[1] = ay_i;
            local_particles[i].acc[2] = az_i;
        });
    }).wait();
}

real_type GSimulation :: updateParticles(int n, real_type dt)
{
   int i;
   real_type energy = 0;

   for (i = 0; i < n; ++i)// update position
   {
     particles[i].vel[0] += particles[i].acc[0] * dt; //2flops
     particles[i].vel[1] += particles[i].acc[1] * dt; //2flops
     particles[i].vel[2] += particles[i].acc[2] * dt; //2flops

     particles[i].pos[0] += particles[i].vel[0] * dt; //2flops
     particles[i].pos[1] += particles[i].vel[1] * dt; //2flops
     particles[i].pos[2] += particles[i].vel[2] * dt; //2flops

     particles[i].acc[0] = 0.;
     particles[i].acc[1] = 0.;
     particles[i].acc[2] = 0.;

     energy += 0.5f * particles[i].mass * (
	      particles[i].vel[0]*particles[i].vel[0] +
               particles[i].vel[1]*particles[i].vel[1] +
               particles[i].vel[2]*particles[i].vel[2]); //7flops
   }
   return energy;
}

real_type GSimulation::updateParticles_SYCL(int n, real_type dt) {
    sycl::buffer<real_type> energyBuffer(n);
    _syclQueue.submit([&](handler &h) {
        auto energy = energyBuffer.get_access<sycl::access::mode::write>(h);
        ParticleAoS* local_particles = particles;
        h.parallel_for(n, [=](id<1> i) {
            local_particles[i].vel[0] += local_particles[i].acc[0] * dt; //2flops
            local_particles[i].vel[1] += local_particles[i].acc[1] * dt; //2flops
            local_particles[i].vel[2] += local_particles[i].acc[2] * dt; //2flops

            local_particles[i].pos[0] += local_particles[i].vel[0] * dt; //2flops
            local_particles[i].pos[1] += local_particles[i].vel[1] * dt; //2flops
            local_particles[i].pos[2] += local_particles[i].vel[2] * dt; //2flops

            local_particles[i].acc[0] = 0.;
            local_particles[i].acc[1] = 0.;
            local_particles[i].acc[2] = 0.;

            real_type v2 =  local_particles[i].vel[0]*local_particles[i].vel[0] +
                    local_particles[i].vel[1]*local_particles[i].vel[1] +
                    local_particles[i].vel[2]*local_particles[i].vel[2];
            energy[i] = (0.5f * v2 * local_particles[i].mass);
        });
    }).wait();

    real_type resultEnergy = 0.0f;
    sycl::buffer<real_type> flush(n);
    if(_syclQueue.get_device().is_gpu()) reductionAtomics1(_syclQueue, energyBuffer, flush, resultEnergy, 16);
    else {
        auto energy = energyBuffer.get_access<sycl::access::mode::read>();
        for(int i = 0; i < n; ++i) {
            resultEnergy += energy[i];
        }
    }

    return resultEnergy;
}

void GSimulation :: start(bool useSycl)
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  //allocate particles
  particles = malloc_shared<ParticleAoS>(n, _syclQueue);
  for(int i = 0; i < n; ++i) {
      particles[i].init();
  }

  init_pos();
  init_vel();
  init_acc();
  init_mass();

  print_header();

  _totTime = 0.;


  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;

  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {
   ts0 += time.start();

    useSycl ? get_acceleration_SYCL(n) : get_acceleration(n);

    energy = useSycl ? updateParticles_SYCL(n, dt) : updateParticles(n, dt);
    _kenergy = 0.5 * energy;

    ts1 += time.stop();
    if(!(s%get_sfreq()) )
    {
      nf += 1;
      std::cout << " "
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2)
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }

      ts0 = 0;
      ts1 = 0;
    }

  } //end of the time step loop

  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();

  av/=(double)(nf-2);
  dev=std::sqrt((double)(dev/(double)(nf-2)-av*av));


  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void GSimulation :: print_header()
{

  std::cout << " nPart = " << get_npart()  << "; "
	    << "nSteps = " << get_nsteps() << "; "
	    << "dt = "     << get_tstep()  << std::endl;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " "
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
  free(particles, _syclQueue);
}
