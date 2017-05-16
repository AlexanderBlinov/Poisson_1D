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

bool calculationsNeeded(int igl1, int igl2, int igl3, int igl4,
                        int rit, int Nx, int Ny, int Nz,
                        int r1, int r2, int r3, int r4) {
    for (int i1 = igl1 * r1 + 1; i1 < min((igl1 + 1) * r1 + 1, rit + 1); ++i1) {
        if (max(0, i1 - igl2 * r2) < min(r2, i1 + Nx - igl2 * r2)
            && max(igl3 * r3, i1) < min((igl3 + 1) * r3, i1 + Ny)
            && max(igl4 * r4, i1) < min((igl4 + 1) * r4, i1 + Nz)) {
            return true;
        }
    }
    return false;
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
    int rit = 300, tag = 1000;
    double h1 = 0.01, h2 = 0.01, h3 = 0.01;
    double X = 1, Y = 1, Z = 1;
    double w = 1.7;

    int Nx = (int)(X / h1) - 1, Ny = (int)(Y / h2) - 1, Nz = (int)(Z / h3) - 1;

    int r1 = 10, r3 = 20, r4 = 20;
    int Q1 = (int)ceil((double)rit / r1);
    int Q3 = (int)ceil((double)(rit + Ny) / r3);
    int Q4 = (int)ceil((double)(rit + Nz) / r4);

    int Q2 = size;
    int r2 = (int)ceil((double)(rit + Nx) / Q2);

    double *U = (double *)calloc((size_t)(r1 + 1) * r2 * r3 * Q3 * r4 * Q4, sizeof(double));
    double *preLeft = (double *)calloc((size_t)(r1 + 1) * r3 * Q3 * r4 * Q4, sizeof(double));

    int igl2 = rank;
    int left = rank - 1;
    int right = (size - 1 == rank) ? -1 : rank + 1;

    MPI_Status status;

    MPI_Datatype ujk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &ujk_t);
    MPI_Type_commit(&ujk_t);
    MPI_Type_create_resized(ujk_t, 0, sizeof(double) * r2 * r3 * Q3 * r4 * Q4, &ujk_t);
    MPI_Type_commit(&ujk_t);

    MPI_Datatype  prejk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &prejk_t);
    MPI_Type_commit(&prejk_t);
    MPI_Type_create_resized(prejk_t, 0, sizeof(double) * r3 * Q3 * r4 * Q4, &prejk_t);
    MPI_Type_commit(&prejk_t);

    for (int igl1 = 0; igl1 < Q1; ++igl1) {
        for (int igl3 = 0; igl3 < Q3; ++igl3) {
            for (int igl4 = 0; igl4 < Q4; ++igl4) {
                if (left != -1) {
                    bool leftRecvNeeded = calculationsNeeded(igl1, left, igl3, igl4,
                                                             rit, Nx, Ny, Nz, r1, r2, r3, r4);
                    if (leftRecvNeeded) {
                        MPI_Recv(preLeft + ((Q3 + igl3) * r3 * Q4 + igl4) * r4, r1, prejk_t,
                                 left, tag * igl1 + igl4, MPI_COMM_WORLD, &status);
                    }
                }

//                printf("%d enters tile %d %d %d\n", rank, igl1, igl3, igl4);
                bool sendNeeded = false;
                for (int i1 = igl1 * r1 + 1, ii1 = 1; i1 < min((igl1 + 1) * r1 + 1, rit + 1); ++i1, ++ii1) {
                    for (int i2 = max(0, i1 - igl2 * r2); i2 < min(r2, i1 + Nx - igl2 * r2); ++i2) {
                        for (int i3 = max(igl3 * r3, i1); i3 < min((igl3 + 1) * r3, i1 + Ny); ++i3) {
                            for (int i4 = max(igl4 * r4, i1); i4 < min((igl4 + 1) * r4, i1 + Nz); ++i4) {
                                sendNeeded = true;

                                double uip, uim, ujp, ujm, ukp, ukm, u;
                                int i = igl2 * r2 + i2 - i1, j = i3 - i1, k = i4 - i1;

                                if (i == 0) {
                                    uim = A0((j + 1) * h2, (k + 1) * h3);
                                } else if (i2 == 0) {
                                    uim = preLeft[(ii1 * r3 * Q3 + i3) * r4 * Q4 + i4];
                                } else {
                                    uim = U[((ii1 * r2 + i2 - 1) * r3 * Q3 + i3) * r4 * Q4 + i4];
                                }

                                if (i == Nx - 1) {
                                    uip = A1((j + 1) * h2, (k + 1) * h3, X);
                                } else {
                                    uip = U[(((ii1 - 1) * r2 + i2) * r3 * Q3 + i3 - 1) * r4 * Q4 + i4 - 1];
                                }

                                if (j == 0) {
                                    ujm = B0((i + 1) * h1, (k + 1) * h3);
                                } else {
                                    ujm = U[((ii1 * r2 + i2) * r3 * Q3 + i3 - 1) * r4 * Q4 + i4];
                                }

                                if (j == Ny - 1) {
                                    ujp = B1((i + 1) * h1, (k + 1) * h3, Y);
                                } else if (i2 == 0) {
                                    ujp = preLeft[((ii1 - 1) * r3 * Q3 + i3) * r4 * Q4 + i4 - 1];
                                } else {
                                    ujp = U[(((ii1 - 1) * r2 + i2 - 1) * r3 * Q3 + i3) * r4 * Q4 + i4 - 1];
                                }

                                if (k == 0) {
                                    ukm = C0((i + 1) * h1, (j + 1) * h2);
                                } else {
                                    ukm = U[((ii1 * r2 + i2) * r3 * Q3 + i3) * r4 * Q4 + i4 - 1];
                                }

                                if (k == Nz - 1) {
                                    ukp = C1((i + 1) * h1, (j + 1) * h2, Z);
                                } else if (i2 == 0) {
                                    ukp = preLeft[((ii1 - 1) * r3 * Q3 + i3 - 1) * r4 * Q4 + i4];
                                } else {
                                    ukp = U[(((ii1 - 1) * r2 + i2 - 1) * r3 * Q3 + i3 - 1) * r4 * Q4 + i4];
                                }

                                if (i2 == 0) {
                                    u = preLeft[((ii1 - 1) * r3 * Q3 + i3 - 1) * r4 * Q4 + i4 - 1];
                                } else {
                                    u = U[(((ii1 - 1) * r2 + i2 - 1) * r3 * Q3 + i3 - 1) * r4 * Q4 + i4 - 1];
                                }

                                U[((ii1 * r2 + i2) * r3 * Q3 + i3) * r4 * Q4 + i4] =
                                        w * ((uip + uim) / (h1 * h1)
                                             + (ujp + ujm) / (h2 * h2)
                                             + (ukp + ukm) / (h3 * h3) - F((i + 1) * h1,
                                                                           (j + 1) * h2,
                                                                           (k + 1) * h3)) /
                                        (2 / (h1 * h1) + 2 / (h2 * h2) + 2 / (h3 * h3)) + (1 - w) * u;
                            }
                        }
                    }
                }
//                printf("%d leaves tile %d %d %d\n", rank, igl1, igl3, igl4);

                if (sendNeeded && right != -1) {
                    MPI_Send(U + (((2 * r2 - 1) * Q3 + igl3) * r3 * Q4 + igl4) * r4, r1, ujk_t,
                             right, tag * igl1 + igl4, MPI_COMM_WORLD);
                }
            }
        }

        if (igl1 != Q1 - 1) {
            memcpy(U, U + r1 * r2 * r3 * Q3 * r4 * Q4, r2 * r3 * Q3 * r4 * Q4 * sizeof(double));
            memcpy(preLeft, preLeft + r1 * r3 * Q3 * r4 * Q4, r3 * Q3 * r4 * Q4 * sizeof(double));
        }
    }

    double *R;
    if (rank == 0) {
        R = (double *) malloc(sizeof(double) * r2 * Q2 * r3 * Q3 * r4 * Q4);
    }

    int l = rit % r1;
    if (l == 0) {
        l = r1;
    }

    MPI_Gather(U + l * r2 * r3 * Q3 * r4 * Q4, r2 * r3 * Q3 * r4 * Q4, MPI_DOUBLE,
               R, r2 * r3 * Q3 * r4 * Q4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Finalize();

    if (rank == 0) {
        printf("Time: %f\n", end - start);

        FILE *f = fopen("output.txt", "w");
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    fprintf(f, "%f ", R[((rit + i) * r3 * Q3 + rit + j) * r4 * Q4 + rit + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "-----------------------------------------------\n");
        }
        fclose(f);
    }

    return 0;
}