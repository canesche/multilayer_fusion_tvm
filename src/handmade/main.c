#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_SAMPLES 2
#define NUM_CHANNELS 3
#define HEIGHT 5
#define WIDTH 5
#define KERNEL_SIZE 3
#define OUTPUT_HEIGHT (HEIGHT - KERNEL_SIZE + 1)
#define OUTPUT_WIDTH (WIDTH - KERNEL_SIZE + 1)


void init_matrix(float ****input) {
    for (int s = 0; s < NUM_SAMPLES; s++) {
        for (int c = 0; c < NUM_CHANNELS; c++) {
            for (int h = 0; h < OUTPUT_HEIGHT; h++) {
                for (int w = 0; w < OUTPUT_WIDTH; w++) {
                    input[s][c][h][w] = 
                }
            }
        }
    }
}

// Function to perform 2D convolution
void conv2d(float input[NUM_SAMPLES][NUM_CHANNELS][HEIGHT][WIDTH], float kernel[KERNEL_SIZE][KERNEL_SIZE][NUM_CHANNELS], float output[NUM_SAMPLES][NUM_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH]) {
    for (int s = 0; s < NUM_SAMPLES; s++) {
        for (int c = 0; c < NUM_CHANNELS; c++) {
            for (int i = 0; i < OUTPUT_HEIGHT; i++) {
                for (int j = 0; j < OUTPUT_WIDTH; j++) {
                    output[s][c][i][j] = 0.0;
                    for (int k = 0; k < KERNEL_SIZE; k++) {
                        for (int l = 0; l < KERNEL_SIZE; l++) {
                            output[s][c][i][j] += input[s][c][i + k][j + l] * kernel[k][l][c];
                        }
                    }
                }
            }
        }
    }
}

// Function to apply ReLU activation function
void relu(float input[NUM_SAMPLES][NUM_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH], float output[NUM_SAMPLES][NUM_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH]) {
    for (int s = 0; s < NUM_SAMPLES; s++) {
        for (int c = 0; c < NUM_CHANNELS; c++) {
            for (int i = 0; i < OUTPUT_HEIGHT; i++) {
                for (int j = 0; j < OUTPUT_WIDTH; j++) {
                    output[s][c][i][j] = (input[s][c][i][j] > 0) ? input[s][c][i][j] : 0;
                }
            }
        }
    }
}

int main() {

    init_matrix();

    float input[NUM_SAMPLES][NUM_CHANNELS][HEIGHT][WIDTH] = {
        {
            {
                {1.0, 2.0, 3.0, 4.0, 5.0},
                {6.0, 7.0, 8.0, 9.0, 10.0},
                {11.0, 12.0, 13.0, 14.0, 15.0},
                {16.0, 17.0, 18.0, 19.0, 20.0},
                {21.0, 22.0, 23.0, 24.0, 25.0}
            },
            {
                {0.1, 0.2, 0.3, 0.4, 0.5},
                {0.6, 0.7, 0.8, 0.9, 1.0},
                {1.1, 1.2, 1.3, 1.4, 1.5},
                {1.6, 1.7, 1.8, 1.9, 2.0},
                {2.1, 2.2, 2.3, 2.4, 2.5}
            },
            {
                {0.01, 0.02, 0.03, 0.04, 0.05},
                {0.06, 0.07, 0.08, 0.09, 0.1},
                {0.11, 0.12, 0.13, 0.14, 0.15},
                {0.16, 0.17, 0.18, 0.19, 0.20},
                {0.21, 0.22, 0.23, 0.24, 0.25}
            }
        },
        {
            {
                {0.5, 0.6, 0.7, 0.8, 0.9},
                {1.0, 1.1, 1.2, 1.3, 1.4},
                {1.5, 1.6, 1.7, 1.8, 1.9},
                {2.0, 2.1, 2.2, 2.3, 2.4},
                {2.5, 2.6, 2.7, 2.8, 2.9}
            },
            {
                {0.01, 0.02, 0.03, 0.04, 0.05},
                {0.06, 0.07, 0.08, 0.09, 0.10},
                {0.11, 0.12, 0.13, 0.14, 0.15},
                {0.16, 0.17, 0.18, 0.19, 0.20},
                {0.21, 0.22, 0.23, 0.24, 0.25}
            },
            {
                {1.0, 2.0, 3.0, 4.0, 5.0},
                {6.0, 7.0, 8.0, 9.0, 10.0},
                {11.0, 12.0, 13.0, 14.0, 15.0},
                {16.0, 17.0, 18.0, 19.0, 20.0},
                {21.0, 22.0, 23.0, 24.0, 25.0}
            }
        }
    };

    float kernel[KERNEL_SIZE][KERNEL_SIZE][NUM_CHANNELS] = {
        {
            {0.5, 0.5, 0.5},
            {0.5, 0.5, 0.5},
            {0.5, 0.5, 0.5}
        },
        {
            {0.1, 0.1, 0.1},
            {0.1, 0.1, 0.1},
            {0.1, 0.1, 0.1}
        },
        {
            {0.01, 0.01, 0.01},
            {0.01, 0.01, 0.01},
            {0.01, 0.01, 0.01}
        }
    };

    float conv_output[NUM_SAMPLES][NUM_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH];
    float relu_output[NUM_SAMPLES][NUM_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH];

    conv2d(input, kernel, conv_output);
    relu(conv_output, relu_output);

    printf("Convolution Output:\n");
    for (int s = 0; s < NUM_SAMPLES; s++) {
        for (int c = 0; c < NUM_CHANNELS; c++) {
            printf("Sample %d, Channel %d:\n", s, c);
            for (int i = 0; i < OUTPUT_HEIGHT; i++) {
                for (int j = 0; j < OUTPUT_WIDTH; j++) {
                    printf("%.2f ", conv_output[s][c][i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    printf("\nReLU Output:\n");
    for (int s = 0; s < NUM_SAMPLES; s++) {
        for (int c = 0; c < NUM_CHANNELS; c++) {
            printf("Sample %d, Channel %d:\n", s, c);
            for (int i = 0; i < OUTPUT_HEIGHT; i++) {
                for (int j = 0; j < OUTPUT_WIDTH; j++) {
                    printf("%.2f ", relu_output[s][c][i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    return 0;
}