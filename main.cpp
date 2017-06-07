#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

using namespace std;


// Neodnorodnost'
double F(double x, double y, double z) {
    return 3 * exp(x + y + z);
}

// Granichnoe uslovie pri x=0
double A0(double y, double z) {
    return exp(y + z);
}

// Granichnoe uslovie pri x=X
double A1(double y, double z, double X) {
    return exp(X + y + z);
}

// Granichnoe uslovie pri y=0
double B0(double x, double z) {
    return exp(x + z);
}

// Granichnoe uslovie pri y=Y
double B1(double x, double z, double Y) {
    return exp(x + Y + z);
}

// Granichnoe uslovie pri z=0
double C0(double x, double y) {
    return exp(x + y);
}

// Granichnoe uslovie pri z=Z
double C1(double x, double y, double Z) {
    return exp(x + y + Z);
}

bool calculationsNeeded(int m, int igl1, int Nx, int r1) {
    return max(0, m - igl1 * r1) < min(r1, m + Nx - igl1 * r1);
}

int main(int argc, char* argv[]) {
    int rank, size;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Vichislenie osnovnih parametrov oblasti reshenija
    int rit = 600, tag = 31;
    double h1 = 0.01, h2 = 0.01, h3 = 0.01;
    double X = 1, Y = 1, Z = 1;
    double w = 1.7;

    int Nx = (int)(X / h1) - 1, Ny = (int)(Y / h2) - 1, Nz = (int)(Z / h3) - 1;

    int r2 = 20, r3 = 20;
    int Q2 = (int)ceil((double)Ny / r2);
    int Q3 = (int)ceil((double)Nz / r3);

    int Q1 = size;
    int r1 = (int)ceil((double)(rit + Nx) / Q1);

    double *U = (double *)calloc((size_t)2 * r1 * r2 * Q2 * r3 * Q3, sizeof(double));
    double *preLeft = (double *)calloc((size_t)2 * r2 * Q2 * r3 * Q3, sizeof(double));

    int igl1 = rank;
    int left = rank - 1;
    int right = (size - 1 == rank) ? -1 : rank + 1;

    MPI_Status status;

    MPI_Datatype ujk_t;
    MPI_Type_vector(r2, r3, r3 * Q3, MPI_DOUBLE, &ujk_t);
    MPI_Type_commit(&ujk_t);

    for (int m = 1; m <= rit; ++m) {
        for (int igl2 = 0; igl2 < Q2; ++igl2) {
            for (int igl3 = 0; igl3 < Q3; ++igl3) {
                if (left != -1) {
                    bool leftRecvNeeded = calculationsNeeded(m, left, Nx, r1);
                    if (leftRecvNeeded) {
                        MPI_Recv(preLeft + ((Q2 + igl2) * r2 * Q3 + igl3) * r3, 1, ujk_t,
                                 left, (tag * m + igl2) * tag + igl3, MPI_COMM_WORLD, &status);
                    }
                }

//                printf("%d enters tile %d %d %d\n", rank, m, igl2, igl3);
                bool sendNeeded = false;
                for (int i1 = max(0, m - igl1 * r1); i1 < min(r1, m + Nx - igl1 * r1); ++i1) {
                    for (int i2 = igl2 * r2; i2 < min((igl2 + 1) * r2, Ny); ++i2) {
                        for (int i3 = igl3 * r3; i3 < min((igl3 + 1) * r3, Nz); ++i3) {
                            sendNeeded = true;

                            double uip, uim, ujp, ujm, ukp, ukm, u;
                            int i = igl1 * r1 + i1 - m, j = i2, k = i3;

                            if (i == 0) {
                                uim = A0((j + 1) * h2, (k + 1) * h3);
                            } else if (i1 == 0) {
                                uim = preLeft[(r2 * Q2 + i2) * r3 * Q3 + i3];
                            } else {
                                uim = U[((r1 + i1 - 1) * r2 * Q2 + i2) * r3 * Q3 + i3];
                            }

                            if (i == Nx - 1) {
                                uip = A1((j + 1) * h2, (k + 1) * h3, X);
                            } else {
                                uip = U[(i1 * r2 * Q2 + i2) * r3 * Q3 + i3];
                            }

                            if (j == 0) {
                                ujm = B0((i + 1) * h1, (k + 1) * h3);
                            } else {
                                ujm = U[((r1 + i1) * r2 * Q2 + i2 - 1) * r3 * Q3 + i3];
                            }

                            if (j == Ny - 1) {
                                ujp = B1((i + 1) * h1, (k + 1) * h3, Y);
                            } else if (i1 == 0) {
                                ujp = preLeft[(i2 + 1) * r3 * Q3 + i3];
                            } else {
                                ujp = U[((i1 - 1) * r2 * Q2 + i2 + 1) * r3 * Q3 + i3];
                            }

                            if (k == 0) {
                                ukm = C0((i + 1) * h1, (j + 1) * h2);
                            } else {
                                ukm = U[((r1 + i1) * r2 * Q2 + i2) * r3 * Q3 + i3 - 1];
                            }

                            if (k == Nz - 1) {
                                ukp = C1((i + 1) * h1, (j + 1) * h2, Z);
                            } else if (i1 == 0) {
                                ukp = preLeft[i2 * r3 * Q3 + i3 + 1];
                            } else {
                                ukp = U[((i1 - 1) * r2 * Q2 + i2) * r3 * Q3 + i3 + 1];
                            }

                            if (i1 == 0) {
                                u = preLeft[i2 * r3 * Q3 + i3];
                            } else {
                                u = U[((i1 - 1) * r2 * Q2 + i2) * r3 * Q3 + i3];
                            }

                            U[((r1 + i1) * r2 * Q2 + i2) * r3 * Q3 + i3] =
                                    w * ((uip + uim) / (h1 * h1)
                                         + (ujp + ujm) / (h2 * h2)
                                         + (ukp + ukm) / (h3 * h3) - F((i + 1) * h1,
                                                                       (j + 1) * h2,
                                                                       (k + 1) * h3)) /
                                    (2 / (h1 * h1) + 2 / (h2 * h2) + 2 / (h3 * h3)) + (1 - w) * u;
                        }
                    }
                }
//                printf("%d leaves tile %d %d %d\n", rank, m, igl2, igl3);

                if (sendNeeded && right != -1) {
                    MPI_Send(U + (((2 * r1 - 1) * Q2 + igl2) * r2 * Q3 + igl3) * r3, 1, ujk_t,
                             right, (tag * m + igl2) * tag + igl3, MPI_COMM_WORLD);
                }
            }
        }
        memcpy(U, U + r1 * r2 * Q2 * r3 * Q3, r1 * r2 * Q2 * r3 * Q3 * sizeof(double));
        memcpy(preLeft, preLeft + r2 * Q2 * r3 * Q3, r2 * Q2 * r3 * Q3 * sizeof(double));
    }

    double *R;
    if (rank == 0) {
        R = (double *) malloc(sizeof(double) * r1 * Q1 * r2 * Q2 * r3 * Q3);
    }

    MPI_Gather(U, r1 * r2 * Q2 * r3 * Q3, MPI_DOUBLE,
               R, r1 * r2 * Q2 * r3 * Q3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    free(U);
    free(preLeft);

    MPI_Finalize();

    if (rank == 0) {
        printf("Time: %f\n", end - start);

        FILE *f = fopen("output.txt", "w");
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    fprintf(f, "%f ", R[((rit + i) * r2 * Q2 + j) * r3 * Q3 + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "-----------------------------------------------\n");
        }
        fclose(f);
    }

    return 0;
}