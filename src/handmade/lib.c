#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib.h"

float* init_matrix(const int rows, const int cols, const float init_value) {
  const unsigned matrix_size = rows * cols;
  //printf("Allocating %u cells\n", matrix_size);
  float* matrix = (float*) malloc(matrix_size * sizeof(float));
  for (int i = 0; i < matrix_size; i++) {
    matrix[i] = init_value * (float)rand()/(float)RAND_MAX;
  }
  return matrix;
}

void print_matrix(const float* matrix, const int rows, const int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%8.2f", matrix[i * cols + j]);
    }
    printf("\n");
  }
}