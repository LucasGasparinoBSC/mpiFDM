#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#define NDIV 100 // Number of divisions

void erk4(int mpi_size, int mpi_rank, int ncomms, std::vector<int> nodes2comm, int np, int listDofBc, int *local_to_global, float h, float k, float dt, float *u);
void laplacian1d(int np, float h, float k, float *u, float *R);
void data_exchange(int mpi_size, int mpi_rank, int np, float fact, float *u, float *R);

void erk4(int mpi_size, int mpi_rank, int ncomms, std::vector<int> nodes2comm, int np, int listDofBc, int *local_to_global, float h, float k, float dt, float *u)
{
    // RK4 time integration:
    // Coefficients:
    const float a[4] = {0.0f, 0.5f, 0.5f, 1.0f};
    const float b[4] = {1.0f / 6.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 6.0f};

    // Allocate memory for the rhs residual
    float *u_aux = (float *)malloc(np * sizeof(float));
    float *R = (float *)malloc(np * sizeof(float));
    float *Rsum = (float *)malloc(np * sizeof(float));
    memset(R, 0, np * sizeof(float));
    memset(u_aux, 0, np * sizeof(float));
    memset(Rsum, 0, np * sizeof(float));

    // Factor: k/(h*h)
    float fact = k / (h * h);

    // Substeps
    for (int s = 0; s < 4; s++)
    {
        for (int i = 0; i < np; i++)
        {
            u_aux[i] = u[i] + a[s]*dt*R[i];
        }

        // Compute residual R^s
        laplacian1d(np, h, k, u_aux, R);

        // Handle comms
        if (mpi_size > 1)
        {
            data_exchange(mpi_size, mpi_rank, np, fact, u_aux, R);
        }

        // Accumulate R^s into Rsum
        for (int i = 0; i < np; i++)
        {
            Rsum[i] = Rsum[i] + b[s]*dt*R[i];
        }
    }

    // Last update
    for (int i = 0; i < np; i++)
    {
        u[i] = u[i] + Rsum[i];
    }

    // Free memory
    free(u_aux);
    free(R);
    free(Rsum);
}

void laplacian1d(int np, float h, float k, float *u, float *R)
{
    // Laplacian operator 2nd order: (k/h) * [u(i+1) - 2u(i) + u(i-1)]
    for (int i = 1; i < np-1; i++)
    {
        R[i]  = (k/(h*h)) * (u[i+1] - 2.0f*u[i] + u[i-1]);
    }
}

void data_exchange(int mpi_size, int mpi_rank, int np, float fact, float *u, float *R)
{
    // Aux scalars for receiving data
    float win_buf[2] = {0.0f, 0.0f};

    // Form the window with win_buf
    MPI_Win win;
    MPI_Win_create(win_buf, 2*sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Ranks put info on the window
    win_buf[0] = u[0];    // leftmost node
    win_buf[1] = u[np-1]; // rightmost node

    // Each rank gets needed info from the window
    float aux_left, aux_right;
    if (mpi_rank == 0)
    {
        MPI_Win_fence(0, win);
        MPI_Get(&aux_right, 1, MPI_FLOAT, mpi_size-1, 0, 1, MPI_FLOAT, win);
        MPI_Win_fence(0, win);
        R[np-1] = fact * (aux_right - 2.0f*u[np-1] + u[np-2]);
    }
    else if (mpi_rank == mpi_size - 1)
    {
        MPI_Win_fence(0, win);
        MPI_Get(&aux_left, 1, MPI_FLOAT, 0, 1, 1, MPI_FLOAT, win);
        MPI_Win_fence(0, win);
        R[0] = fact * (u[1] - 2.0f*u[0] + aux_left);
    }
    else
    {
        MPI_Win_fence(0, win);
        MPI_Get(&aux_left, 1, MPI_FLOAT, mpi_rank-1, 1, 1, MPI_FLOAT, win);
        MPI_Get(&aux_right, 1, MPI_FLOAT, mpi_rank+1, 0, 1, MPI_FLOAT, win);
        MPI_Win_fence(0, win);
        R[0] = fact * (u[1] - 2.0f*u[0] + aux_left);
        R[np-1] = fact * (aux_right - 2.0f*u[np-1] + u[np-2]);
    }

    // Free window
    MPI_Win_free(&win);
}

int main()
{
    // Init MPI
    int nranks, irank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);

    // Check that (NDIV)/nranks >= 1
    float ratio = (float)(NDIV+1) / (float)nranks;
    if (ratio < 1.0)
    {
        if (irank == 0)
        {
            std::cerr << "Error: (NDIV+1)/nranks < 1" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Generate the 1D grid and IC
    const double xLeft = 0.0;
    const double xRight = 1.0;
    double h = (xRight - xLeft) / NDIV;
    int ndof = NDIV + 1;
    double *xGrid;
    float *phi;

    if (irank == 0)
    {
        xGrid = (double *)malloc(ndof * sizeof(double));
        phi = (float *)malloc(ndof * sizeof(float));
        for (int i = 0; i < ndof; i++)
        {
            xGrid[i] = xLeft + i * h;
            phi[i] = 1.0f;
        }

        // Impose DBCs
        phi[0] = 0.0;
        phi[ndof-1] = 0.0;
    }

    // Create a list of boundary nodes
    int listDofBc[2];
    listDofBc[0] = 0;
    listDofBc[1] = NDIV;

    // Partition the grid and solution arrays:

    // If decimal part of ratio is > 0.5, round it up, otherwise round it down
    int ndof_part = (int)ratio;
    if (ratio - (float)ndof_part > 0.5f)
    {
        ndof_part++;
    }

    // Array of chunk size per rank
    int chunk_sizes[nranks];
    for (int i = 0; i < nranks - 1; i++)
    {
        chunk_sizes[i] = ndof_part;
    }
    chunk_sizes[nranks - 1] = ndof - (nranks - 1) * ndof_part;

    // Create the local arrays
    double *xGrid_local = (double *)malloc(chunk_sizes[irank] * sizeof(double));
    float *phi_local = (float *)malloc(chunk_sizes[irank] * sizeof(float));

    // Rank 0 sends chunks of xGrid and phi to all ranks
    if (irank == 0)
    {
        for (int i = 0; i < nranks; i++)
        {
            if (i == 0)
            {
                for (int j = 0; j < chunk_sizes[i]; j++)
                {
                    xGrid_local[j] = xGrid[j];
                    phi_local[j] = phi[j];
                }
            }
            else
            {
                MPI_Send(&xGrid[i * ndof_part], chunk_sizes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&phi[i * ndof_part], chunk_sizes[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        MPI_Recv(xGrid_local, chunk_sizes[irank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(phi_local, chunk_sizes[irank], MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Each rank crerates a list relating local to global indexes
    int *local_to_global = (int *)malloc(chunk_sizes[irank] * sizeof(int));
    for (int i = 0; i < chunk_sizes[irank]; i++)
    {
        local_to_global[i] = irank * ndof_part + i;
    }

    // Comms list:
    // Each rank has 2 comms (left and rightmost nodes). Ranks with BC nodes have 1 comm
    int ncomms;
    std::vector<int> nodes2comm;
    if (irank == 0 || irank == nranks - 1)
    {
        ncomms = 1;
        if (irank == 0)
        {
            nodes2comm.push_back(local_to_global[chunk_sizes[irank] - 1]);
        }
        else
        {
            nodes2comm.push_back(local_to_global[0]);
        }
    }
    else
    {
        ncomms = 2;
        nodes2comm.push_back(local_to_global[0]);
        nodes2comm.push_back(local_to_global[chunk_sizes[irank] - 1]);
    }

    // Compute dt
    float k = 0.1f;
    float cfl = 0.25f;\
    float dt = cfl * (h * h) / k;

    // Call the time integration routine
    const int nsteps = 1000;
    for (int i = 0; i < nsteps; i++)
    {
        erk4(nranks, irank, ncomms, nodes2comm, chunk_sizes[irank], ncomms, local_to_global, h, k, dt, phi_local);
    }

    // Print the solution iin correct order
    for (int i = 0; i < nranks; i++)
    {
        if (irank == i)
        {
            for (int j = 0; j < chunk_sizes[irank]; j++)
            {
                std::cout << "Rank " << irank << " - Local Node = " << j << " - Node " << local_to_global[j] << " - x = " << xGrid_local[j] << " - phi = " << phi_local[j] << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}